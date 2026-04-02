# supervisor.rs 路线图

> 参考框架: LangChain, LangGraph, DeerFlow, AutoGen, CrewAI

---

## 现状分析

### 已完成
- ✅ Rust 核心: Supervisor + Message (消息路由)
- ✅ Python Agent 基类 + 继承体系
- ✅ Extension 插件系统 (on_load/on_unload/on_message 钩子)
- ✅ 扩展实现: RAG, Function Calling, MCP, Skills, A2A
- ✅ 容错机制 (异常隔离，继续处理)

### 缺失
- ❌ 异步支持 (async/await)
- ❌ 状态管理与持久化
- ❌ LLM 集成层
- ❌ 工作流/图编排 (类似 LangGraph)
- ❌ 流式处理支持
- ❌ 可观测性 (tracing, metrics)
- ❌ 类型化消息系统
- ❌ 工具生态 (预置工具库)

---

## Phase 1: 核心增强 (1-2 月)

### 1.1 异步运行时
```python
# 目标 API
sup = AsyncSupervisor()
await sup.send(msg)
await sup.run_once()
```

**任务:**
- [ ] `AsyncSupervisor` 实现 (Rust 端使用 tokio)
- [ ] `AsyncAgent` 基类，支持 `async handle_message`
- [ ] 异步扩展钩子 `async on_message`
- [ ] 同步/异步双 API 兼容

### 1.2 状态管理
```python
# 目标 API
class MyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.state = State(count=0)  # 自动持久化
    
    def handle_message(self, msg):
        self.state.count += 1  # 自动检查点
```

**任务:**
- [ ] `State` 基类 (支持 checkpoint/restore)
- [ ] 状态后端接口: `StateBackend` (ABC)
- [ ] 内置后端: `MemoryBackend`, `FileBackend`, `RedisBackend`
- [ ] 状态快照与恢复
- [ ] 状态版本控制

### 1.3 类型化消息
```python
# 目标 API
@dataclass
class ChatMessage(Message):
    content: str
    metadata: dict = field(default_factory=dict)

@dataclass  
class ToolCall(Message):
    tool_name: str
    args: dict
```

**任务:**
- [ ] `TypedMessage` 基类
- [ ] 消息序列化/反序列化 (JSON, MessagePack)
- [ ] 消息模式验证 (pydantic 集成)
- [ ] 消息路由规则 (基于类型)

---

## Phase 2: LLM 集成 (2-3 月)

### 2.1 LLM 抽象层
```python
# 目标 API - 类似 LangChain 的统一接口
from supervisor.llm import OpenAI, Anthropic, Ollama, get_model

llm = get_model("gpt-4")  # 自动从环境变量配置
response = await llm.ainvoke("Hello")
stream = await llm.astream("Tell me a story")
```

**任务:**
- [ ] `BaseLLM` 抽象类
- [ ] Provider 实现: OpenAI, Anthropic, Ollama, Azure, AWS Bedrock
- [ ] 统一的 `invoke`/`ainvoke`/`stream`/`astream` 接口
- [ ] Token 计数与成本追踪
- [ ] 重试与超时机制
- [ ] 模型配置管理 (YAML/TOML)

### 2.2 Prompt 模板
```python
# 目标 API
from supervisor.prompt import PromptTemplate

prompt = PromptTemplate.from_file("prompts/agent.md")
rendered = prompt.render(name="Alice", task="summarize")
```

**任务:**
- [ ] `PromptTemplate` 类
- [ ] 变量插值与验证
- [ ] 多文件模板管理
- [ ] 模板版本控制

### 2.3 LLM Agent 集成
```python
# 目标 API
from supervisor import Agent
from supervisor.llm import OpenAI

class ChatAgent(Agent):
    llm = OpenAI(model="gpt-4")
    
    def handle_message(self, msg):
        response = self.llm.invoke(msg.content)
        self.send(msg.sender, response.content)
```

**任务:**
- [ ] `LLMAgent` 基类 (内置对话历史管理)
- [ ] 对话上下文窗口管理
- [ ] 系统提示词注入
- [ ] 多模态消息支持 (文本、图像、音频)

---

## Phase 3: 工作流编排 (3-4 月)

### 3.1 图编排引擎 (对标 LangGraph)
```python
# 目标 API
from supervisor.graph import Graph, Node, Edge

graph = Graph()
graph.add_node("input", InputNode())
graph.add_node("process", ProcessAgent("proc"))
graph.add_node("output", OutputNode())

graph.add_edge("input", "process")
graph.add_edge("process", "output", condition=lambda state: state.done)

result = await graph.run(initial_state={"query": "hello"})
```

**任务:**
- [ ] `Graph` 核心引擎
- [ ] `Node` 抽象: AgentNode, ToolNode, RouterNode
- [ ] 条件边与循环
- [ ] 并行分支执行
- [ ] 子图嵌套
- [ ] 图可视化 (Mermaid/DOT 导出)
- [ ] 状态流控制 (类似 StateGraph)

### 3.2 工作流模式
```python
# 目标 API - 预置工作流模式
from supervisor.patterns import Sequential, Parallel, Router

# 顺序执行
workflow = Sequential([agent1, agent2, agent3])

# 并行执行
workflow = Parallel([agent1, agent2], merge=lambda results: sum(results))

# 路由分发
workflow = Router(routing_fn=lambda msg: msg.type)
```

**任务:**
- [ ] `Sequential` 模式
- [ ] `Parallel` 模式 (MapReduce 风格)
- [ ] `Router` 模式 (条件分发)
- [ ] `Loop` 模式 (迭代直到收敛)
- [ ] `HumanInLoop` 模式 (人工干预点)

### 3.3 记忆系统
```python
# 目标 API
from supervisor.memory import ConversationMemory, VectorMemory

# 对话记忆
agent.memory = ConversationMemory(window=10)

# 向量记忆 (语义检索)
agent.memory = VectorMemory(embedder="text-embedding-3-small")
```

**任务:**
- [ ] `BaseMemory` 抽象
- [ ] `ConversationMemory` (滑动窗口)
- [ ] `SummaryMemory` (对话摘要压缩)
- [ ] `VectorMemory` (语义检索)
- [ ] 记忆持久化

---

## Phase 4: 工具生态 (4-5 月)

### 4.1 预置工具库
```python
# 目标 API
from supervisor.tools import (
    WebSearch,      # 网页搜索
    CodeExecutor,   # 代码执行
    FileIO,         # 文件读写
    Database,       # 数据库查询
    HTTPClient,     # HTTP 请求
    Calculator,      # 计算器
)
```

**任务:**
- [ ] 工具接口标准化
- [ ] WebSearch (DuckDuckGo, Tavily, Serper)
- [ ] CodeExecutor (Docker 沙箱)
- [ ] FileIO (安全文件操作)
- [ ] Database (SQL 查询生成)
- [ ] HTTPClient (API 调用)
- [ ] 计算器与时区工具

### 4.2 工具发现与组合
```python
# 目标 API
agent.tools.discover()  # 自动发现可用工具
agent.tools.auto_select(query="search the web")  # 智能选择
```

**任务:**
- [ ] 工具注册中心
- [ ] 工具元数据描述 (OpenAPI 风格)
- [ ] 工具能力匹配算法
- [ ] 工具链自动编排

### 4.3 第三方集成
```python
# 目标 API
from supervisor.integrations import (
    LangChainTools,    # 兼容 LangChain 工具
    CrewAITools,       # 兼容 CrewAI 工具
    OpenAIAssistants,  # OpenAI Assistants API
)
```

**任务:**
- [ ] LangChain 工具适配器
- [ ] OpenAI Function Calling 兼容
- [ ] Anthropic Tools 兼容
- [ ] 外部 MCP Server 集成

---

## Phase 5: 可观测性 (5-6 月)

### 5.1 日志与追踪
```python
# 目标 API
from supervisor.tracing import trace, Span

with trace("agent_execution"):
    agent.handle(msg)  # 自动记录 span
```

**任务:**
- [ ] OpenTelemetry 集成
- [ ] Span 自动注入 (消息流转追踪)
- [ ] 结构化日志 (JSON 格式)
- [ ] 日志级别动态控制

### 5.2 Metrics 指标
```python
# 目标 API
from supervisor.metrics import Counter, Histogram

messages_processed = Counter("messages_processed")
latency = Histogram("message_latency")
```

**任务:**
- [ ] Prometheus 指标导出
- [ ] 内置指标: 消息吞吐量、延迟、错误率
- [ ] Agent 级别指标
- [ ] 自定义指标注册

### 5.3 调试与可视化
```python
# 目标 API
sup.visualize()  # 生成消息流转图
sup.debug()      # 进入交互式调试模式
```

**任务:**
- [ ] Web Dashboard (消息监控)
- [ ] 执行流程可视化
- [ ] 状态检查器 (Inspector)
- [ ] 回放功能 (消息历史回放)

---

## Phase 6: 生产就绪 (6-7 月)

### 6.1 部署支持
```yaml
# 目标 API - supervisor.yaml
agents:
  - name: chatbot
    class: MyChatAgent
    replicas: 3
    resources:
      cpu: "500m"
      memory: "512Mi"
```

**任务:**
- [ ] Docker 镜像
- [ ] Kubernetes Operator
- [ ] Docker Compose 模板
- [ ] 健康检查端点
- [ ] 优雅关闭

### 6.2 扩展性
**任务:**
- [ ] 水平扩展 (多 Supervisor 实例)
- [ ] 消息队列后端 (Redis Streams, Kafka)
- [ ] Agent 分片策略
- [ ] 负载均衡

### 6.3 安全性
**任务:**
- [ ] 消息加密 (TLS)
- [ ] 身份认证 (Agent 身份)
- [ ] 权限控制 (消息路由 ACL)
- [ ] 审计日志

---

## Phase 7: 高级特性 (7-8 月)

### 7.1 多模态支持
**任务:**
- [ ] 图像消息类型
- [ ] 音频消息类型
- [ ] 文件消息类型
- [ ] 多模态 LLM 集成

### 7.2 人机协作
**任务:**
- [ ] Human-in-the-loop 节点
- [ ] 审批工作流
- [ ] 人工干预 API
- [ ] 通知集成 (Slack, Email)

### 7.3 知识图谱
**任务:**
- [ ] 图数据库后端 (Neo4j)
- [ ] 知识抽取 Agent
- [ ] 图查询工具
- [ ] 推理引擎

---

## Rust vs Python 分层设计

### 设计原则

| 原则 | 说明 |
|------|------|
| **性能优先** | 消息路由、并发调度、序列化用 Rust |
| **易用优先** | 用户 API、业务逻辑用 Python |
| **生态优先** | LLM SDK、第三方库用 Python |
| **边界清晰** | 通过 PyO3 暴露最小接口集 |

### 架构分层图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户代码层 (Python)                        │
│  class MyAgent(Agent): handle_message() ...                  │
├─────────────────────────────────────────────────────────────┤
│                    业务逻辑层 (Python)                         │
│  Agent基类 │ Extension插件 │ LLM集成 │ 工具库 │ 工作流模式      │
├─────────────────────────────────────────────────────────────┤
│                     适配层 (Python + Rust)                    │
│  PyO3 bindings: supervisor._core                             │
├─────────────────────────────────────────────────────────────┤
│                     核心引擎层 (Rust)                          │
│  Supervisor │ Message │ 消息队列 │ 状态存储 │ 图执行引擎        │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层 (Rust)                           │
│  tokio │ serde │ tracing │ dashmap │ crossbeam               │
└─────────────────────────────────────────────────────────────┘
```

### 各模块语言分配

#### 🦀 Rust 实现 (核心性能层)

| 模块 | 职责 | 理由 |
|------|------|------|
| **Supervisor** | Agent 注册、消息路由、调度 | 高并发、零拷贝消息传递 |
| **Message** | 消息结构、序列化 | 频繁创建/传递，性能关键 |
| **MessageQueue** | 异步消息队列 | tokio mpsc/channel，高吞吐 |
| **GraphEngine** | DAG 执行引擎 | 并行调度、状态机 |
| **StateStore** | 状态持久化后端 | RocksDB/Redis 高性能 I/O |
| **MetricsCore** | 指标收集 | 原子计数器、无锁统计 |
| **TracingCore** | Span 管理 | tracing crate 原生支持 |
| **Serializer** | JSON/MessagePack 序列化 | serde 零成本抽象 |

#### 🐍 Python 实现 (业务生态层)

| 模块 | 职责 | 理由 |
|------|------|------|
| **Agent 基类** | 用户继承、生命周期钩子 | Python OO 更友好 |
| **Extension 系统** | 插件加载、钩子链 | 动态性、灵活性 |
| **LLM 集成** | OpenAI/Anthropic/Ollama | SDK 都是 Python |
| **工具库** | WebSearch/CodeExec/... | 调用外部服务，性能不敏感 |
| **工作流模式** | Sequential/Parallel/Router | 业务逻辑，易编写 |
| **Prompt 模板** | 模板渲染、变量插值 | Python 字符串处理方便 |
| **Memory 系统** | Conversation/Vector Memory | 依赖向量库 (Python 生态) |
| **配置管理** | YAML/TOML 解析 | Python 库丰富 |

#### 🔀 Rust + Python 混合 (边界层)

| 模块 | Rust 部分 | Python 部分 |
|------|----------|------------|
| **TypedMessage** | 底层序列化 | pydantic 验证 |
| **AsyncSupervisor** | tokio 运行时 | asyncio 接口适配 |
| **工具执行** | 沙箱隔离 (Docker) | 工具定义 (Python) |
| **持久化** | RocksDB 后端 | 状态 API (Python) |

### 详细模块规划

#### Phase 1: 核心增强

**1.1 异步运行时**

```
Rust:
├── AsyncSupervisor (tokio::runtime::Runtime)
│   ├── agents: DashMap<String, AgentEntry>
│   ├── queue: mpsc::UnboundedReceiver<Message>
│   └── run_once() -> usize
│
├── AsyncMessage (Arc<MessageInner>)
│   └── 零拷贝消息传递
│
Python:
├── AsyncSupervisor (PyO3 wrapper)
│   ├── async send(msg)
│   ├── async run_once()
│   └── await 转换为 tokio Future
│
├── AsyncAgent
│   ├── async handle_message(msg)
│   └── asyncio -> tokio bridge
```

**1.2 状态管理**

```
Rust:
├── StateBackend trait
│   ├── MemoryBackend (DashMap)
│   ├── FileBackend (RocksDB)
│   ├── RedisBackend (redis-rs)
│
├── StateSnapshot
│   ├── checkpoint() -> Vec<u8>
│   ├── restore(data: &[u8])
│
Python:
├── State (PyO3 wrapper)
│   ├── __getattr__/__setattr__ 自动追踪
│   ├── transaction() 上下文管理器
│
├── StateBackend ABC (用户可扩展)
│   ├── CustomBackend 实现示例
```

**1.3 类型化消息**

```
Rust:
├── MessageCore
│   ├── payload: Vec<u8> (原始字节)
│   ├── schema: Option<SchemaRef>
│   ├── serialize/deserialize (serde)
│
Python:
├── TypedMessage (pydantic BaseModel)
│   ├── 字段验证
│   ├── schema_to_json()
│   ├── from_rust(payload) 转换
```

#### Phase 2: LLM 集成

```
全部 Python 实现:

├── supervisor.llm
│   ├── BaseLLM (ABC)
│   │   ├── invoke() / ainvoke()
│   │   ├── stream() / astream()
│   │   ├── batch()
│   │
│   ├── providers/
│   │   ├── OpenAI (openai SDK)
│   │   ├── Anthropic (anthropic SDK)
│   │   ├── Ollama (httpx)
│   │   ├── Azure (azure SDK)
│   │   ├── Bedrock (boto3)
│   │
│   ├── config/
│   │   ├── ModelConfig (YAML)
│   │   ├── RateLimiter
│   │   ├── RetryPolicy
│   │
├── supervisor.prompt
│   ├── PromptTemplate
│   ├── TemplateManager
│   ├── VariableResolver
│
├── supervisor.agents
│   ├── LLMAgent
│   │   ├── memory: BaseMemory
│   │   ├── system_prompt: str
│   │   ├── tools: List[Tool]
│   │
│   ├── ChatAgent
│   ├── RAGAgent
```

理由: LLM SDK (openai, anthropic) 都是 Python，调用外部 API 性能不敏感。

#### Phase 3: 工作流编排

```
Rust (核心引擎):
├── GraphEngine
│   ├── nodes: HashMap<NodeId, NodeSpec>
│   ├── edges: Vec<EdgeSpec>
│   ├── execute(state: State) -> Result
│   ├── parallel_execute() -> Vec<Result>
│   ├── condition_evaluator: Fn(State) -> bool
│
├── NodeExecutor
│   ├── run_node(id, state) -> State
│   ├── error_handling (容错)
│
├── Scheduler
│   ├── dag_executor (topological sort)
│   ├── parallel_scheduler (rayon/tokio)
│
Python (用户接口):
├── Graph
│   ├── add_node(name, node)
│   ├── add_edge(from, to, condition)
│   ├── run(initial_state)
│   ├── visualize() -> Mermaid
│
├── Node (ABC)
│   ├── AgentNode
│   ├── ToolNode
│   ├── RouterNode
│   ├── HumanNode
│
├── patterns/
│   ├── Sequential
│   ├── Parallel
│   ├── Router
│   ├── Loop
│   ├── HumanInLoop
│
├── memory/
│   ├── BaseMemory (ABC)
│   ├── ConversationMemory
│   ├── VectorMemory (调用 chromadb)
│   ├── SummaryMemory
```

#### Phase 4: 工具生态

```
全部 Python:

├── supervisor.tools
│   ├── BaseTool (ABC)
│   │   ├── name, description
│   │   ├── run(**kwargs) -> Any
│   │   ├── schema() -> dict
│   │
│   ├── builtin/
│   │   ├── WebSearch (DuckDuckGo/Tavily API)
│   │   ├── HTTPClient (httpx)
│   │   ├── FileIO (pathlib)
│   │   ├── Database (SQLAlchemy)
│   │   ├── Calculator
│   │   ├── Timezone
│   │
│   ├── sandbox/
│   │   ├── CodeExecutor (Docker SDK)
│   │   ├── SafeEval (RestrictedPython)
│   │
│   ├── registry/
│   │   ├── ToolRegistry
│   │   ├── discover()
│   │   ├── auto_select()
│   │
├── supervisor.integrations
│   ├── LangChainAdapter
│   ├── CrewAIAdapter
│   ├── OpenAIAdapter
```

例外: CodeExecutor 的 Docker 调用可以考虑 Rust 实现安全沙箱。

#### Phase 5: 可观测性

```
Rust (核心):
├── MetricsCore
│   ├── Counter (AtomicU64)
│   ├── Gauge (AtomicF64)
│   ├── Histogram (lock-free)
│   ├── export_prometheus()
│
├── TracingCore
│   ├── Span (tracing crate)
│   ├── Context propagation
│   ├── OpenTelemetry exporter
│
Python (接口):
├── supervisor.tracing
│   ├── trace(name) decorator
│   ├── Span wrapper
│   ├── configure(otlp_endpoint)
│
├── supervisor.metrics
│   ├── Counter / Histogram wrappers
│   ├── MetricsServer (Prometheus HTTP)
│
├── supervisor.debug
│   ├── Inspector
│   ├── visualize()
│   ├── replay()
```

#### Phase 6: 生产就绪

```
Rust:
├── NetworkLayer
│   ├── TLS (rustls)
│   ├── Auth middleware
│   ├── Rate limiting
│
├── Persistence
│   ├── Redis Streams backend
│   ├── Kafka backend (rdkafka)
│   ├── WAL (write-ahead log)
│
├── Cluster
│   ├── Node coordination
│   ├── Sharding strategy
│   ├── Health check
│
Python:
├── supervisor.server
│   ├── HTTPServer (FastAPI)
│   ├── WebSocket (实时消息)
│   ├── Admin API
│
├── supervisor.config
│   ├── YAML 配置解析
│   ├── 部署模板生成
│
├── supervisor.cli
│   ├── 命令行工具
│   ├── 部署脚本
```

### PyO3 接口边界

Rust 暴露给 Python 的最小接口集:

```rust
// src/lib.rs
#[pymodule]
fn _core(py: Python, m: &PyModule) -> PyResult<()> {
    // Phase 0 (已实现)
    m.add_class::<Message>()?;
    m.add_class::<Supervisor>()?;
    
    // Phase 1
    m.add_class::<AsyncSupervisor>()?;
    m.add_class::<State>()?;
    m.add_class::<StateBackend>()?;
    m.add_class::<TypedMessage>()?;
    
    // Phase 3
    m.add_class::<GraphEngine>()?;
    m.add_class::<NodeExecutor>()?;
    
    // Phase 5
    m.add_class::<MetricsCore>()?;
    m.add_class::<TracingCore>()?;
    
    // Phase 6
    m.add_class::<ClusterManager>()?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    
    Ok(())
}
```

### 性能对比预期

| 操作 | 纯 Python | Rust + Python | 提升 |
|------|----------|---------------|------|
| 消息路由 (100万条) | ~500ms | ~50ms | 10x |
| 序列化 (100万次) | ~800ms | ~100ms | 8x |
| 并发调度 (1000 agents) | ~200ms | ~20ms | 10x |
| 状态快照 | ~100ms | ~10ms | 10x |
| LLM 调用 | 2000ms | 2000ms | 1x (网络主导) |

### 开发优先级

**第一阶段: Rust 核心完善**
1. AsyncSupervisor (tokio)
2. 高性能序列化 (serde)
3. 状态存储后端

**第二阶段: Python 业务层**
1. LLM 集成 (复用生态)
2. 工具库 (Python SDK)
3. 工作流模式

**第三阶段: 混合边界**
1. GraphEngine + Python Graph API
2. Metrics Rust core + Python wrapper
3. Cluster Rust + Python admin

---

## 技术栈规划

### Rust 核心
- tokio: 异步运行时
- serde: 序列化
- serde_json / rmp-serde: JSON/MessagePack
- tracing: 日志追踪
- tracing-opentelemetry: OTLP 导出
- dashmap: 并发 HashMap
- crossbeam: 无锁队列
- rocksdb: 嵌入式存储
- redis-rs: Redis 客户端
- rdkafka: Kafka 客户端
- rustls: TLS
- pyo3: Python 绑定

### Python 层
- pydantic: 数据验证
- pydantic-settings: 配置管理
- asyncio: 异步支持
- httpx: HTTP 客户端
- openai / anthropic: LLM SDK
- chromadb / faiss: 向量存储
- docker: 容器管理
- fastapi: HTTP 服务
- pyyaml / tomli: 配置解析
- rich / textual: CLI UI

### 可选依赖
- prometheus-client: 指标导出
- opentelemetry-sdk: 可观测性
- sqlalchemy: 数据库
- redis: Redis 客户端
- kafka-python: Kafka 客户端

---

## 版本规划

| 版本 | 内容 | 时间 |
|------|------|------|
| 0.2.0 | 异步支持、状态管理、类型化消息 | Phase 1 |
| 0.3.0 | LLM 集成层、Prompt 模板 | Phase 2 |
| 0.4.0 | 图编排引擎、工作流模式 | Phase 3 |
| 0.5.0 | 工具生态、第三方集成 | Phase 4 |
| 0.6.0 | 可观测性、调试工具 | Phase 5 |
| 1.0.0 | 生产就绪、稳定 API | Phase 6 |
| 1.1.0+ | 高级特性 | Phase 7 |

---

## 与其他框架对比

| 特性 | supervisor.rs | LangChain | LangGraph | CrewAI | AutoGen |
|------|--------------|-----------|-----------|--------|---------|
| 核心语言 | Rust+Python | Python | Python | Python | Python |
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 异步原生 | 规划中 | ✅ | ✅ | ❌ | ✅ |
| 图编排 | 规划中 | ❌ | ✅ | ❌ | ❌ |
| 多 Agent | ✅ | ❌ | ✅ | ✅ | ✅ |
| 容错机制 | ✅ | ❌ | 部分 | ❌ | ❌ |
| 状态管理 | 规划中 | ✅ | ✅ | ❌ | ✅ |
| 工具生态 | 基础 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 可观测性 | 规划中 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 生产就绪 | 规划中 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## 差异化定位

### 核心优势
1. **性能**: Rust 核心，消息路由零成本抽象
2. **容错**: Erlang/OTP 风格的监督树
3. **简洁**: 最小化 API，概念清晰
4. **扩展**: 灵活的插件系统

### 目标用户
- 需要高性能 Agent 系统的开发者
- 希望从简单开始、逐步扩展的团队
- 对 Rust 技术栈感兴趣的用户
- 需要自定义扩展的场景

---

## 贡献指南

优先级高的任务欢迎认领:
1. 异步运行时实现
2. 状态持久化后端
3. LLM Provider 适配器
4. 图编排引擎核心
5. 工具库实现
6. 文档与示例

详见 [CONTRIBUTING.md](CONTRIBUTING.md)