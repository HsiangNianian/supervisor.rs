supervisor.rs
=============

An agent framework written in Rust and built for both Rust and Python.

``supervisor.rs`` lets you register named *agents* — lightweight message
handlers — and coordinate them through a central *Supervisor* that routes
messages and provides fault-tolerant delivery: if an agent's handler raises
an exception, the supervisor logs the error and continues processing the
remaining messages.

Installation
------------

Build and install with `maturin <https://github.com/PyO3/maturin>`_:

.. code-block:: bash

   pip install maturin
   maturin develop          # editable install for development

Quick start (Python)
--------------------

.. code-block:: python

   from supervisor import Supervisor, Message

   sup = Supervisor()

   def greeter(msg: Message) -> None:
       print(f"Hello from '{msg.recipient}'! You said: {msg.content}")

   sup.register("greeter", greeter)
   sup.send(Message("main", "greeter", "Hi there!"))
   processed = sup.run_once()
   print(f"Processed {processed} message(s).")

API reference
-------------

``Message(sender, recipient, content)``
    A message exchanged between agents.  All three fields are read/write
    strings exposed both to Rust and Python.

``Supervisor()``
    Manages a collection of named agents.

    ``register(name, handler)``
        Register an agent.  *handler* is a Python callable that accepts a
        single ``Message`` argument.

    ``unregister(name) -> bool``
        Remove an agent. Returns ``True`` if it existed.

    ``send(msg)``
        Enqueue *msg* for its recipient agent. Raises ``KeyError`` for unknown
        recipients.

    ``run_once() -> int``
        Deliver all pending messages. Returns the count of successfully
        processed messages. Faults in individual agents are logged to stderr
        and do not abort the dispatch loop.

    ``agent_names() -> list[str]``
        Names of all registered agents.

    ``agent_count() -> int``
        Number of registered agents.

    ``pending_count(name) -> int | None``
        Queued message count for *name*, or ``None`` if unknown.

License
-------

MIT

