from supervisors import Agent, SupervisorAgent, Message, Supervisor


class Worker(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] handling: {msg.content}")


sup = Supervisor()
manager = SupervisorAgent(
    "manager",
    router=lambda msg: "worker_a" if "urgent" in msg.content else "worker_b",
)
manager.add_sub_agent(Worker("worker_a"))
manager.add_sub_agent(Worker("worker_b"))
manager.register(sup)

sup.send(Message("user", "manager", "urgent: fix production"))
sup.run_once()
# Output: [worker_a] handling: urgent: fix production
