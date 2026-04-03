from supervisors import LoopAgent, Message


class ReasoningAgent(LoopAgent):
    def step(self, state):
        state["count"] = state.get("count", 0) + 1
        if state["count"] >= 3:
            state["done"] = True
            state["result"] = "Reasoning complete."
        return state


agent = ReasoningAgent("reasoner", max_iterations=10)
state = agent.run_loop(Message("user", "reasoner", "Think about X"))
print(state["result"])
