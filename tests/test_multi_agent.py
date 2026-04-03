from supervisors import Agent, MultiAgent, Message, Supervisor


class Researcher(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] researching: {msg.content}")


sup = Supervisor()
team = MultiAgent(
    "research_team",
    members=[Researcher("alice"), Researcher("bob")],
    max_rounds=5,
)
team.register(sup)

sup.send(Message("user", "research_team", "Investigate topic X"))
sup.run_once()
