"""Pipeline Agent -- sequential stage-based processing.

A :class:`PipelineAgent` processes messages through an ordered sequence of
stages.  Each stage is a callable that receives the current context dict
and returns an updated context.  This is useful for ETL workflows,
multi-step transformations, or any sequential processing pipeline.

Example::

    from supervisor import PipelineAgent, Message, Supervisor

    def parse(ctx):
        ctx["parsed"] = ctx["input"].split(",")
        return ctx

    def transform(ctx):
        ctx["transformed"] = [x.strip().upper() for x in ctx["parsed"]]
        return ctx

    def output(ctx):
        ctx["result"] = " | ".join(ctx["transformed"])
        return ctx

    sup = Supervisor()
    agent = PipelineAgent("etl", stages=[parse, transform, output])
    agent.register(sup)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisor.agent import Agent

if TYPE_CHECKING:
    from supervisor._core import Message

# Type alias for a pipeline stage function.
StageFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


class PipelineAgent(Agent):
    """Agent that processes messages through an ordered pipeline of stages.

    Each stage is a callable ``(ctx: dict) -> dict`` that transforms the
    context dictionary.  Stages are executed in the order they are added.

    Parameters:
        name: Unique name for the agent.
        stages: Optional initial list of stage callables.
    """

    def __init__(
        self,
        name: str,
        *,
        stages: Optional[List[StageFunc]] = None,
    ) -> None:
        super().__init__(name)
        self._stages: List[StageFunc] = list(stages) if stages else []

    # -- stage management ----------------------------------------------------

    def add_stage(self, stage: StageFunc) -> "PipelineAgent":
        """Append a *stage* to the pipeline.

        Returns *self* for chaining::

            agent.add_stage(parse).add_stage(transform).add_stage(output)
        """
        self._stages.append(stage)
        return self

    def stage(
        self, func: Optional[StageFunc] = None
    ) -> Any:
        """Decorator to register a function as a pipeline stage.

        Usage::

            @agent.stage
            def my_stage(ctx):
                ctx["key"] = "value"
                return ctx
        """
        if func is not None:
            self._stages.append(func)
            return func

        def decorator(fn: StageFunc) -> StageFunc:
            self._stages.append(fn)
            return fn

        return decorator

    @property
    def stages(self) -> List[StageFunc]:
        """Return a copy of the current stage list."""
        return list(self._stages)

    @property
    def stage_count(self) -> int:
        """Return the number of stages in the pipeline."""
        return len(self._stages)

    # -- pipeline hooks ------------------------------------------------------

    def on_pipeline_start(
        self, msg: "Message", ctx: Dict[str, Any]
    ) -> None:
        """Hook called before the first stage runs.

        Override to perform setup.  The default implementation is a no-op.
        """

    def on_stage_complete(
        self, stage_index: int, stage_name: str, ctx: Dict[str, Any]
    ) -> None:
        """Hook called after each stage completes.

        Override to add logging, metrics, or validation.

        Parameters:
            stage_index: Zero-based index of the completed stage.
            stage_name: Name of the completed stage function.
            ctx: The context dict after the stage ran.
        """

    def on_pipeline_end(
        self, msg: "Message", ctx: Dict[str, Any]
    ) -> None:
        """Hook called after the last stage completes.

        Override to perform cleanup or send results.
        """

    # -- core pipeline execution ---------------------------------------------

    def run_pipeline(self, msg: "Message") -> Dict[str, Any]:
        """Execute the full pipeline for the given *msg*.

        Builds an initial context from the message, runs each stage in
        order, and returns the final context.

        Returns:
            The final context dict after all stages have run.
        """
        ctx: Dict[str, Any] = {
            "input": msg.content,
            "sender": msg.sender,
            "recipient": msg.recipient,
        }

        self.on_pipeline_start(msg, ctx)

        for i, stage_func in enumerate(self._stages):
            ctx = stage_func(ctx)
            stage_name = getattr(stage_func, "__name__", f"stage_{i}")
            self.on_stage_complete(i, stage_name, ctx)

        self.on_pipeline_end(msg, ctx)
        return ctx

    def handle_message(self, msg: "Message") -> None:
        """Process an incoming message by running the pipeline.

        Subclasses may override this to customise post-pipeline behaviour.
        """
        self.run_pipeline(msg)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        return (
            f"PipelineAgent(name={self.name!r}, "
            f"stages={self.stage_count}, "
            f"extensions=[{ext_names}])"
        )


__all__ = ["PipelineAgent"]
