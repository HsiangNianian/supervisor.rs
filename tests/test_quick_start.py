from supervisors import Agent, Supervisor, Message


class GreeterAgent(Agent):
    def handle_message(self, msg: Message) -> None:
        print(f"Hello from '{self.name}'! You said: {msg.content}")


sup = Supervisor()
greeter = GreeterAgent("greeter")
greeter.register(sup)

sup.send(Message("user", "greeter", "Hi there!"))
sup.run_once()
# Output: Hello from 'greeter'! You said: Hi there!
