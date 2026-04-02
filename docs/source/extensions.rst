Extensions
==========

Extensions are "meta-level" plugins that add capabilities to agents.
All extensions inherit from :class:`supervisor.ext.Extension`.

Extension base class
--------------------

.. code-block:: python

   from supervisor.ext import Extension

   class MyExtension(Extension):
       name = "my_extension"

       def on_load(self, agent):
           print(f"Loaded onto {agent.name}")

       def on_unload(self, agent):
           print(f"Unloaded from {agent.name}")

       def on_message(self, agent, msg):
           # Return modified Message or None to pass through
           return None

RAG (Retrieval-Augmented Generation)
-------------------------------------

Subclass ``RAGExtension`` and implement ``retrieve`` and ``add_documents``
to connect your own retrieval backend (ChromaDB, LightRAG, FAISS, etc.):

.. code-block:: python

   from supervisor.ext.rag import RAGExtension

   class ChromaRAG(RAGExtension):
       def __init__(self, collection):
           super().__init__(auto_retrieve=True, top_k=5)
           self._col = collection

       def retrieve(self, query, top_k=None):
           results = self._col.query(query_texts=[query], n_results=top_k or self.top_k)
           return results["documents"][0]

       def add_documents(self, docs, **kwargs):
           ids = [str(i) for i in range(len(docs))]
           self._col.add(documents=docs, ids=ids)

When ``auto_retrieve=True``, incoming messages are automatically enriched
with retrieved context before reaching ``handle_message``.

Function Calling
----------------

Register callable tools with JSON-Schema-style specifications:

.. code-block:: python

   from supervisor.ext.function_calling import FunctionCallingExtension

   fc = FunctionCallingExtension()

   @fc.tool(description="Add two numbers")
   def add(a: int, b: int) -> int:
       return a + b

   agent.use(fc)

   # Invoke a tool
   result = fc.call_tool("add", a=1, b=2)

   # List all tool specs
   specs = fc.get_tools_spec()

MCP (Model Context Protocol)
-----------------------------

Use the built-in MCP client and server, or integrate the official MCP SDK:

.. code-block:: python

   from supervisor.ext.mcp import MCPExtension

   mcp = MCPExtension(server_url="http://localhost:8080")

   @mcp.mcp_tool(description="Reverse a string")
   def reverse(text: str) -> str:
       return text[::-1]

   agent.use(mcp)

   # Call remote tools
   mcp.call_remote("some_tool", param="value")

   # Handle requests on the local server
   response = mcp.server.handle_request({"tool": "reverse", "args": {"text": "hello"}})

Skills
------

Register reusable behaviours as skills:

.. code-block:: python

   from supervisor.ext.skills import SkillsExtension

   skills = SkillsExtension()

   @skills.skill
   def summarise(agent, msg):
       return f"Summary: {msg.content[:50]}..."

   agent.use(skills)
   result = skills.invoke("summarise", agent, msg)

A2A Extension
-------------

The ``A2AExtension`` adds advanced agent-to-agent patterns beyond the
built-in ``Agent.send``:

.. code-block:: python

   from supervisor.ext.a2a import A2AExtension

   a2a = A2AExtension()
   agent.use(a2a)

   # Broadcast to all other agents
   a2a.broadcast(agent, "Hello everyone!")

   # Discover all agents on the supervisor
   names = a2a.discover_agents(agent)

   # Request/reply pattern
   a2a.request(agent, "other_agent", "question", reply_handler)

Creating custom extensions
--------------------------

Extend the ``Extension`` base class:

.. code-block:: python

   from supervisor.ext import Extension

   class LoggingExtension(Extension):
       name = "logging"

       def on_message(self, agent, msg):
           print(f"[{agent.name}] received: {msg}")
           return None  # pass through unchanged
