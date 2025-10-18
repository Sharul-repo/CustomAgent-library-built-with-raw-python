# CustomAgent-Python

**CustomAgent** is a lightweight Python library for building LLM agents **from scratch**, without relying on any third-party SDKs like OpenAI Agent SDK, Google's ADK,PydanticAI etc... It’s built entirely with raw Python and `requests`, giving you full control over LLM requests, streaming, and structured responses.  

This library is designed to work with the [OpenRouter](https://openrouter.ai/) API endpoint and is perfect for developers who want to understand the mechanics of LLM agents without heavy abstractions.

---

## 🚀 Features

- **Streamed Conversations** – `stream_chat_conversation` streams responses in real-time while keeping conversation history.  
- **Single-Shot Queries** – `chat_single_ask` for one-off prompts.  
- **Structured Outputs** – Pass a Pydantic model via `output_type` to get validated and structured LLM responses.  
- **Environment-Based Configuration** – API keys and endpoints can be managed via a `.env` file.  
- **Lightweight and Transparent** – No hidden SDKs, full control over HTTP requests, retries, and payloads.  
- **Logging** – Rotating file logs plus console output for easy debugging.

---

## 💡 Future Add-Ons

- **Tool Calling** – almost complete, coming soon.  
- **Web Searching** – under development for advanced LLM interactions.  

---

## 📦 Requirements

- Python 3.9+  
- `requests`  
- `pydantic`  
- `python-dotenv`  
- `urllib3`  

The project is managed with **uv**, so dependency installation is simple:  

---

### 1️⃣ Clone the repo  

```bash
git clone https://github.com/your-username/langgraph-multiagent.git
cd Multi-Agent

# Install dependencies using uv (reads pyproject.toml + uv.lock)
uv sync

# Run your app
uv run main.py
