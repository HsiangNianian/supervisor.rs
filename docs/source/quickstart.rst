Quick start
===========

Installation
------------

Build and install with `maturin <https://github.com/PyO3/maturin>`_:

.. code-block:: bash

   pip install maturin
   maturin develop          # editable install for development

Creating an agent
-----------------

Create a custom agent by subclassing ``Agent`` and overriding
``handle_message``:

.. code-block:: python

   from supervisor import Agent, Supervisor, Message

   class GreeterAgent(Agent):
       def handle_message(self, msg: Message) -> None:
           print(f"Hello from '{self.name}'! You said: {msg.content}")

   sup = Supervisor()
   greeter = GreeterAgent("greeter")
   greeter.register(sup)

   sup.send(Message("main", "greeter", "Hi there!"))
   sup.run_once()

Agent-to-Agent messaging
------------------------

Every ``Agent`` subclass automatically supports A2A communication via
the built-in ``send`` method:

.. code-block:: python

   class Ping(Agent):
       def handle_message(self, msg):
           print(f"Ping received: {msg.content}")
           self.send("pong", "ping!")

   class Pong(Agent):
       def handle_message(self, msg):
           print(f"Pong received: {msg.content}")

   sup = Supervisor()
   Ping("ping").register(sup)
   Pong("pong").register(sup)

   sup.send(Message("main", "ping", "start"))
   sup.run_once()  # delivers to ping; ping sends to pong
   sup.run_once()  # delivers to pong

Running the demo
----------------

.. code-block:: bash

   python -m supervisor
