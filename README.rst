supervisor.rs
=============

An agent framework written in Rust and built for both Rust and Python.

``supervisor.rs`` lets you register named *agents* — lightweight message
handlers — and coordinate them through a central *Supervisor* that routes
messages and provides fault-tolerant delivery: if an agent's handler raises
an exception, the supervisor logs the error and continues processing the
remaining messages.

Features
--------

* **Agent base class with inheritance** — subclass ``Agent`` and override
  ``handle_message`` to create custom agents.
* **Extension plugin system** — load "meta-level" extensions onto any agent
  to add capabilities such as RAG, Function Calling, MCP, Skills, and A2A.
* **Built-in A2A** — every ``Agent`` subclass automatically supports
  agent-to-agent messaging via ``agent.send(recipient, content)``.
* **Fault-tolerant supervision** — the ``Supervisor`` catches handler
  exceptions, logs them, and continues processing.
* **Cross-language** — core written in Rust, usable from both Rust and Python
  via PyO3.

Installation
------------

Build and install with `maturin <https://github.com/PyO3/maturin>`_:

.. code-block:: bash

   pip install maturin
   maturin develop          # editable install for development

Quick start (Python)
--------------------

Create an agent by subclassing ``Agent`` and overriding ``handle_message``:

.. code-block:: python

   from supervisor import Agent, Supervisor, Message

   class GreeterAgent(Agent):
       def handle_message(self, msg: Message) -> None:
           print(f"Hello from '{self.name}'! You said: {msg.content}")

   sup = Supervisor()
   greeter = GreeterAgent("greeter")
   greeter.register(sup)

   sup.send(Message("main", "greeter", "Hi there!"))
   processed = sup.run_once()
   print(f"Processed {processed} message(s).")

Agent-to-Agent (A2A) communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every agent automatically supports A2A messaging:

.. code-block:: python

   class Ping(Agent):
       def handle_message(self, msg):
           print(f"Ping got: {msg.content}")
           self.send("pong", "ping!")

   class Pong(Agent):
       def handle_message(self, msg):
           print(f"Pong got: {msg.content}")

   sup = Supervisor()
   Ping("ping").register(sup)
   Pong("pong").register(sup)

   sup.send(Message("main", "ping", "start"))
   sup.run_once()  # delivers to ping, which sends to pong
   sup.run_once()  # delivers to pong

Extension plugins
-----------------

Extensions are loaded onto agents via ``agent.use(extension)`` and provide
additional capabilities.  All extensions inherit from ``Extension``.

RAG (Retrieval-Augmented Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from supervisor.ext.rag import RAGExtension

   class MyRAG(RAGExtension):
       def __init__(self):
           super().__init__(auto_retrieve=True, top_k=5)
           self._store = []

       def retrieve(self, query, top_k=None):
           # Replace with ChromaDB, LightRAG, FAISS, etc.
           return [d for d in self._store if query.lower() in d.lower()]

       def add_documents(self, docs, **kwargs):
           self._store.extend(docs)

   agent = MyAgent("search")
   agent.use(MyRAG())

Function Calling
~~~~~~~~~~~~~~~~

.. code-block:: python

   from supervisor.ext.function_calling import FunctionCallingExtension

   fc = FunctionCallingExtension()

   @fc.tool(description="Add two numbers")
   def add(a: int, b: int) -> int:
       return a + b

   agent.use(fc)
   result = fc.call_tool("add", a=1, b=2)  # returns 3

MCP (Model Context Protocol)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from supervisor.ext.mcp import MCPExtension

   mcp = MCPExtension(server_url="http://localhost:8080")

   @mcp.mcp_tool(description="Reverse a string")
   def reverse(text: str) -> str:
       return text[::-1]

   agent.use(mcp)

Skills
~~~~~~

.. code-block:: python

   from supervisor.ext.skills import SkillsExtension

   skills = SkillsExtension()

   @skills.skill
   def summarise(agent, msg):
       return f"Summary: {msg.content[:50]}..."

   agent.use(skills)
   result = skills.invoke("summarise", agent, msg)

A2A Extension (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the built-in A2A, the ``A2AExtension`` adds broadcast and discovery:

.. code-block:: python

   from supervisor.ext.a2a import A2AExtension

   a2a = A2AExtension()
   agent.use(a2a)

   # Broadcast to all other agents
   a2a.broadcast(agent, "Hello everyone!")

   # Discover all agents on the supervisor
   names = a2a.discover_agents(agent)

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

``Agent(name)``
    Base class for agents.  Subclass and override ``handle_message(msg)``.

    ``use(extension) -> Agent``
        Load an extension plugin.  Returns ``self`` for chaining.

    ``remove_extension(name) -> bool``
        Remove a loaded extension by name.

    ``register(supervisor)``
        Register the agent with a supervisor.

    ``unregister() -> bool``
        Remove the agent from its supervisor.

    ``send(recipient, content)``
        Send a message to another agent (built-in A2A).

``Extension``
    Base class for extension plugins.

    ``on_load(agent)``
        Called when loaded onto an agent.

    ``on_unload(agent)``
        Called when removed.

    ``on_message(agent, msg) -> Message | None``
        Intercept a message before the agent handles it.

License
-------

MIT

