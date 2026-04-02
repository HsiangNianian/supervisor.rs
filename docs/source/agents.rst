Agents
======

The ``Agent`` base class
------------------------

All agents inherit from :class:`supervisor.agent.Agent`.  Override
``handle_message`` to define custom behaviour:

.. code-block:: python

   from supervisor import Agent, Message

   class MyAgent(Agent):
       def handle_message(self, msg: Message) -> None:
           print(f"[{self.name}] got: {msg.content}")

Registering with a Supervisor
-----------------------------

Agents are registered with a ``Supervisor`` for message routing:

.. code-block:: python

   from supervisor import Supervisor, Message

   sup = Supervisor()
   agent = MyAgent("myagent")
   agent.register(sup)

   sup.send(Message("system", "myagent", "hello"))
   sup.run_once()

Loading extensions
------------------

Extensions are loaded via ``agent.use(extension)``.  Calls can be chained:

.. code-block:: python

   from supervisor.ext.function_calling import FunctionCallingExtension
   from supervisor.ext.skills import SkillsExtension

   agent = MyAgent("multi")
   agent.use(FunctionCallingExtension()).use(SkillsExtension())

Built-in A2A
-------------

Every agent inherits ``send(recipient, content)`` for point-to-point
messaging between agents on the same supervisor:

.. code-block:: python

   class Forwarder(Agent):
       def handle_message(self, msg):
           self.send("next_agent", msg.content)

API reference
-------------

.. py:class:: Agent(name)

   Base class for all agents.

   .. py:method:: handle_message(msg)

      Override to process incoming messages.

   .. py:method:: use(extension) -> Agent

      Load an extension plugin.  Returns ``self`` for chaining.

   .. py:method:: remove_extension(name) -> bool

      Remove a loaded extension by name.

   .. py:method:: register(supervisor)

      Register the agent with a supervisor.

   .. py:method:: unregister() -> bool

      Remove the agent from its supervisor.

   .. py:method:: send(recipient, content)

      Send a message to another agent (built-in A2A).
