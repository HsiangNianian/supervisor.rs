from supervisor import Supervisor, Message


def main() -> None:
    """Run a short demonstration of the supervisor agent framework."""
    sup = Supervisor()

    log: list[str] = []

    def echo_handler(msg: Message) -> None:
        line = f"[{msg.recipient}] received from '{msg.sender}': {msg.content}"
        log.append(line)
        print(line)

    def reverse_handler(msg: Message) -> None:
        line = f"[{msg.recipient}] reversed: {msg.content[::-1]}"
        log.append(line)
        print(line)

    sup.register("echo", echo_handler)
    sup.register("reverse", reverse_handler)

    sup.send(Message("main", "echo", "Hello, Agent!"))
    sup.send(Message("main", "reverse", "supervisor.rs"))
    sup.send(Message("echo", "echo", "Self-message"))

    processed = sup.run_once()
    print(f"\nProcessed {processed} message(s) across {sup.agent_count()} agent(s).")


if __name__ == "__main__":
    main()
