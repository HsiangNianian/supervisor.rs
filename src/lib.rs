use pyo3::prelude::*;
use std::collections::HashMap;

/// A message exchanged between agents.
///
/// # Examples
///
/// ```rust
/// use supervisor::Message;
///
/// let msg = Message::new("alice".into(), "bob".into(), "hello".into());
/// assert_eq!(msg.sender, "alice");
/// assert_eq!(msg.recipient, "bob");
/// assert_eq!(msg.content, "hello");
/// ```
#[pyclass]
#[derive(Clone, Debug)]
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
}

impl Message {
    /// Create a new message (Rust-native constructor).
    pub fn new(sender: String, recipient: String, content: String) -> Self {
        Message {
            sender,
            recipient,
            content,
        }
    }
}

#[pymethods]
impl Message {
    /// Create a new `Message` (Python constructor).
    #[new]
    #[pyo3(signature = (sender, recipient, content))]
    pub fn py_new(sender: String, recipient: String, content: String) -> Self {
        Message::new(sender, recipient, content)
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

/// Internal holder for an agent's Python handler and its pending message queue.
struct AgentEntry {
    handler: PyObject,
    queue: Vec<Message>,
}

/// The Supervisor manages a collection of named agents and routes [`Message`]s
/// between them.
///
/// When an agent's handler raises an exception the supervisor logs the error
/// and continues processing remaining messages, providing fault-tolerant
/// supervision in the style of Erlang/OTP supervisors.
///
/// # Python usage
///
/// ```python
/// from supervisor import Supervisor, Message
///
/// def greeter(msg):
///     print(f"Received: {msg.content}")
///
/// sup = Supervisor()
/// sup.register("greeter", greeter)
/// sup.send(Message("system", "greeter", "Hello!"))
/// sup.run_once()
/// ```
///
/// # Rust usage
///
/// Rust callers interact with the same `Supervisor` struct; agent handlers are
/// Python callables stored as `PyObject`.  Embed a Python interpreter (via
/// PyO3) or use the Python extension module built by maturin.
#[pyclass]
pub struct Supervisor {
    agents: HashMap<String, AgentEntry>,
}

#[pymethods]
impl Supervisor {
    /// Create a new, empty `Supervisor`.
    #[new]
    pub fn new() -> Self {
        Supervisor {
            agents: HashMap::new(),
        }
    }

    /// Register a named agent with a Python callable as its message handler.
    ///
    /// The callable receives a single positional argument: the [`Message`]
    /// being delivered.  Any return value is ignored.
    ///
    /// Registering a name that already exists replaces the previous handler
    /// (and discards any queued messages for that agent).
    pub fn register(&mut self, name: String, handler: PyObject) {
        self.agents.insert(
            name,
            AgentEntry {
                handler,
                queue: Vec::new(),
            },
        );
    }

    /// Remove a registered agent.
    ///
    /// Returns `True` if the agent existed and was removed, `False` otherwise.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.agents.remove(name).is_some()
    }

    /// Enqueue a message for delivery to its named recipient agent.
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

    /// Deliver all pending messages to their respective agents.
    ///
    /// Messages within each agent's queue are delivered in FIFO order.  The
    /// order in which different agents are visited is unspecified (HashMap
    /// iteration order).
    ///
    /// If an agent's handler raises an exception the error is printed to stderr
    /// and the supervisor continues with the next message (fault-tolerant).
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

    /// Return the names of all currently registered agents.
    pub fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    /// Return the number of messages queued for `name`, or `None` if no such
    /// agent is registered.
    pub fn pending_count(&self, name: &str) -> Option<usize> {
        self.agents.get(name).map(|e| e.queue.len())
    }

    /// Return the total number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

#[pymodule]
fn _core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Message>()?;
    m.add_class::<Supervisor>()?;
    Ok(())
}