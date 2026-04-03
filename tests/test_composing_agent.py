from supervisors import (
    Agent,
    MultiAgent,
    SupervisorAgent,
    Message,
    Supervisor,
)


class Specialist(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] working on: {msg.content}")


# Build a collaborative team.
team = MultiAgent(
    "dev_team",
    members=[
        Specialist("frontend"),
        Specialist("backend"),
    ],
)

# Wrap it in a hierarchical supervisor.
manager = SupervisorAgent("manager", router=lambda msg: "dev_team")
manager.add_sub_agent(team)

sup = Supervisor()
manager.register(sup)
sup.send(Message("cto", "manager", "Build feature X"))
sup.run_once()
