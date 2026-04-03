"""Tests for the four agent types: LoopAgent, PipelineAgent,
SupervisorAgent, and MultiAgent."""

import pytest

from supervisor._core import Message, Supervisor
from supervisor.agent import Agent
from supervisor.loop_agent import LoopAgent
from supervisor.pipeline import PipelineAgent
from supervisor.supervisor_agent import SupervisorAgent
from supervisor.multi_agent import MultiAgent


# ---------------------------------------------------------------------------
# LoopAgent
# ---------------------------------------------------------------------------

class TestLoopAgent:
    def test_basic_loop(self):
        """Loop runs step() until done flag is set."""
        class Counter(LoopAgent):
            def step(self, state):
                state["count"] = state.get("count", 0) + 1
                if state["count"] >= 3:
                    state["done"] = True
                return state

        sup = Supervisor()
        agent = Counter("counter", max_iterations=10)
        agent.register(sup)
        sup.send(Message("user", "counter", "go"))
        sup.run_once()

    def test_loop_max_iterations(self):
        """Loop stops at max_iterations even without done flag."""
        iterations_run = []

        class Infinite(LoopAgent):
            def step(self, state):
                state["i"] = state.get("i", 0) + 1
                iterations_run.append(state["i"])
                return state

        sup = Supervisor()
        agent = Infinite("inf", max_iterations=5)
        agent.register(sup)
        sup.send(Message("user", "inf", "run"))
        sup.run_once()
        assert len(iterations_run) == 5

    def test_loop_hooks_called(self):
        """on_loop_start and on_loop_end are called."""
        events = []

        class Hooked(LoopAgent):
            def on_loop_start(self, msg, state):
                events.append("start")

            def on_loop_end(self, msg, state, iterations):
                events.append(f"end:{iterations}")

            def step(self, state):
                state["done"] = True
                return state

        sup = Supervisor()
        agent = Hooked("hooked", max_iterations=10)
        agent.register(sup)
        sup.send(Message("user", "hooked", "test"))
        sup.run_once()
        assert events == ["start", "end:1"]

    def test_loop_state_factory(self):
        """Custom state_factory initialises state from the message."""
        final_states = []

        class Custom(LoopAgent):
            def step(self, state):
                state["done"] = True
                return state

            def on_loop_end(self, msg, state, iterations):
                final_states.append(state)

        factory = lambda msg: {"custom_key": msg.content.upper()}
        sup = Supervisor()
        agent = Custom("custom", max_iterations=5, state_factory=factory)
        agent.register(sup)
        sup.send(Message("user", "custom", "hello"))
        sup.run_once()
        assert final_states[0]["custom_key"] == "HELLO"

    def test_loop_run_loop_returns_state(self):
        """run_loop() returns the final state dict."""
        class Simple(LoopAgent):
            def step(self, state):
                state["result"] = "computed"
                state["done"] = True
                return state

        agent = Simple("s", max_iterations=5)
        msg = Message("user", "s", "input")
        state = agent.run_loop(msg)
        assert state["result"] == "computed"
        assert state["input"] == "input"

    def test_loop_repr(self):
        agent = LoopAgent("test", max_iterations=5)
        r = repr(agent)
        assert "test" in r
        assert "max_iterations=5" in r

    def test_loop_default_step_noop(self):
        """Default step returns state unchanged (stops due to max_iterations=1)."""
        agent = LoopAgent("noop", max_iterations=1)
        msg = Message("user", "noop", "test")
        state = agent.run_loop(msg)
        assert state["input"] == "test"


# ---------------------------------------------------------------------------
# PipelineAgent
# ---------------------------------------------------------------------------

class TestPipelineAgent:
    def test_basic_pipeline(self):
        """Stages run in order."""
        results = []

        def stage_a(ctx):
            ctx["a"] = True
            results.append("a")
            return ctx

        def stage_b(ctx):
            ctx["b"] = True
            results.append("b")
            return ctx

        sup = Supervisor()
        agent = PipelineAgent("pipe", stages=[stage_a, stage_b])
        agent.register(sup)
        sup.send(Message("user", "pipe", "data"))
        sup.run_once()
        assert results == ["a", "b"]

    def test_pipeline_add_stage_chaining(self):
        """add_stage returns self for chaining."""
        def s1(ctx):
            return ctx

        def s2(ctx):
            return ctx

        agent = PipelineAgent("p")
        result = agent.add_stage(s1).add_stage(s2)
        assert result is agent
        assert agent.stage_count == 2

    def test_pipeline_stage_decorator(self):
        """@agent.stage decorator registers a stage."""
        agent = PipelineAgent("p")

        @agent.stage
        def my_stage(ctx):
            ctx["decorated"] = True
            return ctx

        assert agent.stage_count == 1

    def test_pipeline_context_passes_through(self):
        """Context is passed from stage to stage."""
        def s1(ctx):
            ctx["step1"] = "done"
            return ctx

        def s2(ctx):
            ctx["step2"] = ctx.get("step1", "missing") + "_extended"
            return ctx

        agent = PipelineAgent("p", stages=[s1, s2])
        msg = Message("user", "p", "input")
        ctx = agent.run_pipeline(msg)
        assert ctx["step1"] == "done"
        assert ctx["step2"] == "done_extended"
        assert ctx["input"] == "input"

    def test_pipeline_hooks(self):
        """on_pipeline_start, on_stage_complete, on_pipeline_end are called."""
        events = []

        class Hooked(PipelineAgent):
            def on_pipeline_start(self, msg, ctx):
                events.append("start")

            def on_stage_complete(self, index, name, ctx):
                events.append(f"stage:{index}:{name}")

            def on_pipeline_end(self, msg, ctx):
                events.append("end")

        def step_a(ctx):
            return ctx

        def step_b(ctx):
            return ctx

        sup = Supervisor()
        agent = Hooked("hooked", stages=[step_a, step_b])
        agent.register(sup)
        sup.send(Message("user", "hooked", "test"))
        sup.run_once()
        assert events == ["start", "stage:0:step_a", "stage:1:step_b", "end"]

    def test_pipeline_empty(self):
        """Empty pipeline still works, returns initial context."""
        agent = PipelineAgent("empty")
        msg = Message("user", "empty", "hello")
        ctx = agent.run_pipeline(msg)
        assert ctx["input"] == "hello"

    def test_pipeline_repr(self):
        agent = PipelineAgent("test", stages=[lambda ctx: ctx])
        r = repr(agent)
        assert "test" in r
        assert "stages=1" in r

    def test_pipeline_stages_property(self):
        """stages property returns a copy."""
        def s(ctx):
            return ctx
        agent = PipelineAgent("p", stages=[s])
        stages = agent.stages
        stages.append(s)  # modify copy
        assert agent.stage_count == 1  # original unchanged


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_basic_delegation(self):
        """Supervisor routes message to a specific sub-agent."""
        received = []

        class Worker(Agent):
            def handle_message(self, msg):
                received.append((self.name, msg.content))

        def router(msg):
            return "worker_a"

        sup = Supervisor()
        manager = SupervisorAgent("manager", router=router)
        manager.add_sub_agent(Worker("worker_a"))
        manager.add_sub_agent(Worker("worker_b"))
        manager.register(sup)

        sup.send(Message("user", "manager", "task"))
        sup.run_once()
        assert ("worker_a", "task") in received
        assert all(name != "worker_b" for name, _ in received)

    def test_broadcast_to_subs(self):
        """Without router, messages are broadcast to all sub-agents."""
        received = []

        class Worker(Agent):
            def handle_message(self, msg):
                received.append(self.name)

        sup = Supervisor()
        manager = SupervisorAgent("manager")
        manager.add_sub_agent(Worker("w1"))
        manager.add_sub_agent(Worker("w2"))
        manager.register(sup)

        sup.send(Message("user", "manager", "task"))
        sup.run_once()
        assert set(received) == {"w1", "w2"}

    def test_add_remove_sub_agent(self):
        manager = SupervisorAgent("m")
        w = Agent("worker")
        manager.add_sub_agent(w)
        assert "worker" in manager.sub_agent_names
        assert manager.sub_agent_count == 1

        result = manager.remove_sub_agent("worker")
        assert result is True
        assert manager.sub_agent_count == 0

    def test_remove_nonexistent_sub_agent(self):
        manager = SupervisorAgent("m")
        assert manager.remove_sub_agent("ghost") is False

    def test_get_sub_agent(self):
        manager = SupervisorAgent("m")
        w = Agent("worker")
        manager.add_sub_agent(w)
        assert manager.get_sub_agent("worker") is w
        assert manager.get_sub_agent("ghost") is None

    def test_delegate_unknown_raises(self):
        manager = SupervisorAgent("m")
        with pytest.raises(KeyError, match="ghost"):
            manager.delegate("ghost", "content")

    def test_chaining(self):
        """add_sub_agent returns self for chaining."""
        manager = SupervisorAgent("m")
        result = manager.add_sub_agent(Agent("a")).add_sub_agent(Agent("b"))
        assert result is manager
        assert manager.sub_agent_count == 2

    def test_hooks(self):
        """on_delegate and on_sub_agents_complete are called."""
        events = []

        class Tracked(SupervisorAgent):
            def on_delegate(self, msg, target):
                events.append(f"delegate:{target}")

            def on_sub_agents_complete(self, processed):
                events.append(f"complete:{processed}")

        sup = Supervisor()
        manager = Tracked("m", router=lambda msg: "w")
        manager.add_sub_agent(Agent("w"))
        manager.register(sup)

        sup.send(Message("user", "m", "task"))
        sup.run_once()
        assert "delegate:w" in events
        assert any(e.startswith("complete:") for e in events)

    def test_repr(self):
        manager = SupervisorAgent("test")
        manager.add_sub_agent(Agent("w"))
        r = repr(manager)
        assert "test" in r
        assert "w" in r


# ---------------------------------------------------------------------------
# MultiAgent
# ---------------------------------------------------------------------------

class TestMultiAgent:
    def test_broadcast_to_all_members(self):
        """Default strategy broadcasts to all members."""
        received = []

        class Worker(Agent):
            def handle_message(self, msg):
                received.append(self.name)

        group = MultiAgent("team")
        group.add_member(Worker("alice"))
        group.add_member(Worker("bob"))

        sup = Supervisor()
        group.register(sup)
        sup.send(Message("user", "team", "task"))
        sup.run_once()
        assert set(received) == {"alice", "bob"}

    def test_custom_strategy(self):
        """Custom strategy controls which members receive messages."""
        received = []

        class Worker(Agent):
            def handle_message(self, msg):
                received.append(self.name)

        def first_only(msg, members):
            return [members[0]] if members else []

        group = MultiAgent("team", strategy=first_only)
        group.add_member(Worker("alice"))
        group.add_member(Worker("bob"))

        sup = Supervisor()
        group.register(sup)
        sup.send(Message("user", "team", "task"))
        sup.run_once()
        assert received == ["alice"]

    def test_add_remove_member(self):
        group = MultiAgent("g")
        w = Agent("worker")
        group.add_member(w)
        assert "worker" in group.member_names
        assert group.member_count == 1

        assert group.remove_member("worker") is True
        assert group.member_count == 0

    def test_remove_nonexistent_member(self):
        group = MultiAgent("g")
        assert group.remove_member("ghost") is False

    def test_get_member(self):
        group = MultiAgent("g")
        w = Agent("worker")
        group.add_member(w)
        assert group.get_member("worker") is w
        assert group.get_member("ghost") is None

    def test_members_constructor(self):
        """Members can be passed in the constructor."""
        group = MultiAgent("g", members=[Agent("a"), Agent("b")])
        assert group.member_count == 2

    def test_multi_round_communication(self):
        """Members receive messages within the group."""
        messages_seen = []

        class Worker(Agent):
            def handle_message(self, msg):
                messages_seen.append((self.name, msg.content))

        group = MultiAgent("g", max_rounds=5)
        group.add_member(Worker("alice"))
        group.add_member(Worker("bob"))

        sup = Supervisor()
        group.register(sup)
        sup.send(Message("user", "g", "collaborate"))
        sup.run_once()
        # Both members should receive the initial broadcast
        assert ("alice", "collaborate") in messages_seen
        assert ("bob", "collaborate") in messages_seen

    def test_max_rounds_limit(self):
        """max_rounds prevents infinite loops."""
        round_count = [0]

        class Looper(Agent):
            def handle_message(self, msg):
                round_count[0] += 1
                if self.supervisor:
                    try:
                        self.send("looper", "again")
                    except Exception:
                        pass

        group = MultiAgent("g", max_rounds=3)
        group.add_member(Looper("looper"))

        sup = Supervisor()
        group.register(sup)
        sup.send(Message("user", "g", "start"))
        sup.run_once()
        # Should not run more than 3 rounds worth
        assert round_count[0] <= 4  # 1 initial + up to 3 rounds

    def test_hooks(self):
        """on_group_start and on_group_end are called."""
        events = []

        class Tracked(MultiAgent):
            def on_group_start(self, msg):
                events.append("start")

            def on_group_end(self, msg, total):
                events.append(f"end:{total}")

        group = Tracked("g")
        group.add_member(Agent("w"))

        sup = Supervisor()
        group.register(sup)
        sup.send(Message("user", "g", "task"))
        sup.run_once()
        assert events[0] == "start"
        assert events[1].startswith("end:")

    def test_chaining(self):
        """add_member returns self for chaining."""
        group = MultiAgent("g")
        result = group.add_member(Agent("a")).add_member(Agent("b"))
        assert result is group
        assert group.member_count == 2

    def test_repr(self):
        group = MultiAgent("test", members=[Agent("a")])
        r = repr(group)
        assert "test" in r
        assert "a" in r


# ---------------------------------------------------------------------------
# Composition: SupervisorAgent + MultiAgent
# ---------------------------------------------------------------------------

class TestComposition:
    def test_supervisor_with_multi_agent_sub(self):
        """A SupervisorAgent can use a MultiAgent as a sub-agent."""
        received = []

        class Worker(Agent):
            def handle_message(self, msg):
                received.append((self.name, msg.content))

        team = MultiAgent("team")
        team.add_member(Worker("w1"))
        team.add_member(Worker("w2"))

        manager = SupervisorAgent("manager", router=lambda msg: "team")
        manager.add_sub_agent(team)

        sup = Supervisor()
        manager.register(sup)
        sup.send(Message("user", "manager", "collaborate"))
        sup.run_once()

        # Both workers in the team should have received the message
        names = [name for name, _ in received]
        assert "w1" in names
        assert "w2" in names

    def test_pipeline_in_loop(self):
        """A LoopAgent can contain pipeline-style logic."""
        results = []

        class PipelineLoop(LoopAgent):
            def step(self, state):
                iteration = state.get("iteration", 0)
                state["iteration"] = iteration + 1
                results.append(state["iteration"])
                if state["iteration"] >= 2:
                    state["done"] = True
                return state

        sup = Supervisor()
        agent = PipelineLoop("pl", max_iterations=5)
        agent.register(sup)
        sup.send(Message("user", "pl", "go"))
        sup.run_once()
        assert results == [1, 2]

    def test_all_types_on_same_supervisor(self):
        """All four agent types can coexist on one Supervisor."""
        sup = Supervisor()

        loop = LoopAgent("loop", max_iterations=1)
        pipe = PipelineAgent("pipe")
        supervisor_agent = SupervisorAgent("sup_agent")
        multi = MultiAgent("multi")

        loop.register(sup)
        pipe.register(sup)
        supervisor_agent.register(sup)
        multi.register(sup)

        assert sup.agent_count() == 4
        names = set(sup.agent_names())
        assert names == {"loop", "pipe", "sup_agent", "multi"}

    def test_imports_from_top_level(self):
        """All new types are importable from the top-level package."""
        from supervisor import LoopAgent, PipelineAgent
        from supervisor import SupervisorAgent, MultiAgent
        assert LoopAgent is not None
        assert PipelineAgent is not None
        assert SupervisorAgent is not None
        assert MultiAgent is not None
