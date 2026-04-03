# ReAct Loop with Bash Tool — DeepSeek Reasoner

A single `LoopAgent` that uses the DeepSeek `deepseek-reasoner` model (OpenAI-compatible API) with a `bash` tool to execute shell commands. The agent runs a ReAct-style reasoning loop: **think → act → observe → repeat**.

## Setup

```bash
cd examples/react_bash
cp .env.example .env   # fill in OPENAI_API_KEY
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your DeepSeek API key |
| `OPENAI_BASE_URL` | No | `https://api.deepseek.com` | API endpoint |
| `MODEL` | No | `deepseek-reasoner` | Model name |

## Usage

```bash
uv run python main.py
```

## How It Works

1. **`ReActLoopAgent`** extends `LoopAgent` from supervisors.rs
2. Each `step()` calls the DeepSeek API with function-calling support
3. If the model invokes the `bash` tool, the result is appended and the loop continues
4. When the model responds with text only (no tool calls), the loop terminates
5. The final answer is printed and returned to the caller
