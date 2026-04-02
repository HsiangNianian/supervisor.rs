use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A message exchanged between agents.
///
/// Backward compatible: `Message(sender, recipient, content)` still works.
/// New optional fields: `msg_type` and metadata via `set_meta`/`get_meta`.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    /// Name of the agent that sent this message.
    #[pyo3(get, set)]
    pub sender: String,
    /// Name of the agent this message is addressed to.
    #[pyo3(get, set)]
    pub recipient: String,
    /// Payload of the message.
    #[pyo3(get, set)]
    pub content: String,
    /// Type of the message (e.g. "text", "tool_call", "tool_result").
    #[pyo3(get, set)]
    pub msg_type: String,
    /// Key-value metadata attached to the message.
    metadata: HashMap<String, String>,
}

impl Message {
    /// Create a new message (Rust-native constructor).
    pub fn new(sender: String, recipient: String, content: String) -> Self {
        Message {
            sender,
            recipient,
            content,
            msg_type: "text".to_string(),
            metadata: HashMap::new(),
        }
    }
}

#[pymethods]
impl Message {
    /// Create a new `Message` (Python constructor).
    ///
    /// `msg_type` defaults to `"text"` when omitted.
    #[new]
    #[pyo3(signature = (sender, recipient, content, msg_type=None))]
    pub fn py_new(
        sender: String,
        recipient: String,
        content: String,
        msg_type: Option<String>,
    ) -> Self {
        Message {
            sender,
            recipient,
            content,
            msg_type: msg_type.unwrap_or_else(|| "text".to_string()),
            metadata: HashMap::new(),
        }
    }

    /// Set a metadata key-value pair on the message.
    pub fn set_meta(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get a metadata value by key, or `None` if not present.
    pub fn get_meta(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    /// Return all metadata as a dict.
    pub fn get_all_meta(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(sender={:?}, recipient={:?}, content={:?})",
            self.sender, self.recipient, self.content
        )
    }

    fn __str__(&self) -> String {
        format!("[{}→{}] {}", self.sender, self.recipient, self.content)
    }
}

// ---------------------------------------------------------------------------
// ToolSpec
// ---------------------------------------------------------------------------

/// Specification for a registered tool, stored in Rust for performance.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    #[pyo3(get)]
    pub name: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// JSON-encoded parameter schema.
    #[pyo3(get)]
    pub parameters_json: String,
}

#[pymethods]
impl ToolSpec {
    #[new]
    #[pyo3(signature = (name, description="", parameters_json="{}"))]
    pub fn py_new(name: String, description: &str, parameters_json: &str) -> Self {
        ToolSpec {
            name,
            description: description.to_string(),
            parameters_json: parameters_json.to_string(),
        }
    }

    /// Return a dict representation of the tool spec.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("name", &self.name)?;
        dict.set_item("description", &self.description)?;
        dict.set_item("parameters", &self.parameters_json)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolSpec(name={:?}, description={:?})",
            self.name, self.description
        )
    }
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// A Rust-backed registry that stores tool specifications and Python handlers.
///
/// The registry provides fast lookup and type-safe storage while letting
/// Python hold the actual callable implementations.
#[pyclass]
pub struct ToolRegistry {
    specs: HashMap<String, ToolSpec>,
    handlers: HashMap<String, PyObject>,
}

#[pymethods]
impl ToolRegistry {
    #[new]
    pub fn new() -> Self {
        ToolRegistry {
            specs: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    /// Register a tool with its spec and Python handler callable.
    pub fn register(&mut self, spec: ToolSpec, handler: PyObject) {
        let name = spec.name.clone();
        self.specs.insert(name.clone(), spec);
        self.handlers.insert(name, handler);
    }

    /// Unregister a tool by name. Returns `True` if it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        let s = self.specs.remove(name).is_some();
        let h = self.handlers.remove(name).is_some();
        s || h
    }

    /// Get the Python handler callable for a tool.
    ///
    /// Raises `KeyError` if no tool with the given name is registered.
    pub fn get_handler(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match self.handlers.get(name) {
            Some(handler) => Ok(handler.clone_ref(py)),
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "No tool registered with name '{}'",
                name
            ))),
        }
    }

    /// Return all registered tool specs.
    pub fn list_tools(&self) -> Vec<ToolSpec> {
        self.specs.values().cloned().collect()
    }

    /// Get the spec for a specific tool, or `None`.
    pub fn get_spec(&self, name: &str) -> Option<ToolSpec> {
        self.specs.get(name).cloned()
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.specs.keys().cloned().collect()
    }

    /// Return the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.specs.len()
    }

    /// Check whether a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.specs.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Supervisor (async via tokio)
// ---------------------------------------------------------------------------

/// Internal holder for an agent's Python handler and its pending message queue.
struct AgentEntry {
    handler: PyObject,
    queue: Vec<Message>,
}

/// The Supervisor manages a collection of named agents and routes [`Message`]s
/// between them.  Internally powered by a **tokio** async runtime.
///
/// When an agent's handler raises an exception the supervisor logs the error
/// and continues processing remaining messages, providing fault-tolerant
/// supervision in the style of Erlang/OTP supervisors.
#[pyclass]
pub struct Supervisor {
    runtime: Arc<Runtime>,
    agents: HashMap<String, AgentEntry>,
}

#[pymethods]
impl Supervisor {
    /// Create a new, empty `Supervisor` backed by a tokio runtime.
    #[new]
    pub fn new() -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create tokio runtime: {}",
                e
            ))
        })?;
        Ok(Supervisor {
            runtime: Arc::new(runtime),
            agents: HashMap::new(),
        })
    }

    /// Register a named agent with a Python callable as its message handler.
    pub fn register(&mut self, name: String, handler: PyObject) {
        self.agents.insert(
            name,
            AgentEntry {
                handler,
                queue: Vec::new(),
            },
        );
    }

    /// Remove a registered agent. Returns `True` if it existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.agents.remove(name).is_some()
    }

    /// Enqueue a message for delivery to its named recipient.
    ///
    /// Raises `KeyError` if no agent with `msg.recipient` is registered.
    pub fn send(&mut self, msg: Message) -> PyResult<()> {
        let recipient = msg.recipient.clone();
        match self.agents.get_mut(&recipient) {
            Some(entry) => {
                entry.queue.push(msg);
                Ok(())
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "No agent registered with name '{}'",
                recipient
            ))),
        }
    }

    /// Deliver all pending messages synchronously (backward compatible).
    ///
    /// Returns the total number of messages successfully processed.
    pub fn run_once(&mut self, py: Python<'_>) -> PyResult<usize> {
        let mut processed: usize = 0;
        for (name, entry) in self.agents.iter_mut() {
            let messages: Vec<Message> = entry.queue.drain(..).collect();
            for msg in messages {
                match entry.handler.call1(py, (msg,)) {
                    Ok(_) => processed += 1,
                    Err(e) => {
                        eprintln!("supervisor: agent '{}' raised an error: {}", name, e);
                    }
                }
            }
        }
        Ok(processed)
    }

    /// Deliver all pending messages using tokio for concurrent dispatch.
    ///
    /// Releases the Python GIL so that agents running on different tokio
    /// tasks can overlap when their handlers release the GIL (e.g. during
    /// network I/O).  Returns the total number of messages processed.
    pub fn dispatch_async(&mut self, py: Python<'_>) -> PyResult<usize> {
        let rt = self.runtime.clone();

        // Collect pending work while we hold the GIL.
        let mut work: Vec<(String, PyObject, Vec<Message>)> = Vec::new();
        for (name, entry) in self.agents.iter_mut() {
            let msgs: Vec<Message> = entry.queue.drain(..).collect();
            if !msgs.is_empty() {
                work.push((name.clone(), entry.handler.clone_ref(py), msgs));
            }
        }

        if work.is_empty() {
            return Ok(0);
        }

        // Release the GIL and dispatch concurrently on the tokio runtime.
        let processed = py.allow_threads(|| {
            rt.block_on(async {
                let mut handles = Vec::new();
                for (name, handler, messages) in work {
                    let handle = tokio::spawn(async move {
                        let mut count = 0usize;
                        for msg in messages {
                            let result =
                                Python::with_gil(|py| handler.call1(py, (msg,)));
                            match result {
                                Ok(_) => count += 1,
                                Err(e) => {
                                    eprintln!(
                                        "supervisor: agent '{}' raised an error: {}",
                                        name, e
                                    );
                                }
                            }
                        }
                        count
                    });
                    handles.push(handle);
                }

                let mut total = 0usize;
                for handle in handles {
                    match handle.await {
                        Ok(count) => total += count,
                        Err(e) => {
                            eprintln!("supervisor: task join error: {}", e);
                        }
                    }
                }
                total
            })
        });

        Ok(processed)
    }

    /// Return the names of all currently registered agents.
    pub fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Return the number of messages queued for an agent, or `None`.
    pub fn pending_count(&self, name: &str) -> Option<usize> {
        self.agents.get(name).map(|e| e.queue.len())
    }

    /// Return the total number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Message>()?;
    m.add_class::<ToolSpec>()?;
    m.add_class::<ToolRegistry>()?;
    m.add_class::<Supervisor>()?;
    Ok(())
}
