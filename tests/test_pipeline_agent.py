from supervisors import PipelineAgent, Message


def parse(ctx):
    ctx["tokens"] = ctx["input"].split()
    return ctx


def analyse(ctx):
    ctx["count"] = len(ctx["tokens"])
    return ctx


agent = PipelineAgent("analyser", stages=[parse, analyse])
result = agent.run_pipeline(Message("user", "analyser", "hello world"))
print(result["count"])  # 2
