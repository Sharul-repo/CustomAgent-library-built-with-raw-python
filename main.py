import os
import json
import logging
import requests
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from logging.handlers import RotatingFileHandler

LOG_FILENAME = "app.log"

file_handler = RotatingFileHandler(
    LOG_FILENAME,
    maxBytes=5_000_000,   # 5 MB
    backupCount=3,
    encoding="utf-8"
)

file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        file_handler,                    
        logging.StreamHandler()           
    ]
)

log = logging.getLogger(__name__)

class ConfigError(RuntimeError):
    pass

class CustomLLMCall:
    def __init__(
        self,
        model: str,
        temp: float = 0.7,
        reasoning: bool = False,
        max_tokens: int = 512,
        output_type: Optional[Type[BaseModel]] = None,
        timeout: float = 30.0,
    ) -> None:
        load_dotenv()
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = os.environ.get("CHAT_COMPLETION_URL")

        if not self.api_key:
            log.warning("OPENROUTER_API_KEY missing from environment")
            raise ConfigError("OPENROUTER_API_KEY missing from environment")
        if not self.base_url:
            log.warning("CHAT_COMPLETION_URL missing from environment")
            raise ConfigError("CHAT_COMPLETION_URL missing from environment")

        self.model = model
        self.temp = temp
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.output_type = output_type
        self.timeout = timeout

        self.session = requests.Session()
        retries = Retry(total=1, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_payload(self, messages: List[Dict[str, Any]], stream: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temp,
            "max_tokens": self.max_tokens,
            "reasoning": {"enabled": self.reasoning},
        }
        if stream:
            payload["stream"] = True

        if self.output_type is not None:
            if not issubclass(self.output_type, BaseModel):
                raise TypeError("output_type must be a pydantic BaseModel subclass")
            json_schema = self.output_type.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.output_type.__name__,
                    "strict": True,
                    "schema": json_schema,
                },
            }
        return payload

    def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = self.build_payload(messages, stream=False)
        try:
            resp = self.session.post(url=self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            log.error("Network error while calling LLM: %s", e)
            raise

        if not resp.ok:
            log.error("LLM returned status %s: %s", resp.status_code, resp.text)
            raise RuntimeError(f"LLM request failed: {resp.status_code}")

        try:
            return resp.json()
        except ValueError:
            raise RuntimeError("Invalid JSON response from LLM")

    def stream_invoke(self, messages: List[Dict[str, Any]]):
        """
        Robust SSE-style streaming parser.
        Yields parsed data objects (dict) as they arrive.
        """
        payload = self.build_payload(messages, stream=True)
        try:
            with self.session.post(self.base_url, headers=self.headers, json=payload, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                buffer = ""
                for chunk in r.iter_content(chunk_size=4096, decode_unicode=True):
                    if chunk is None:
                        continue
                    buffer += chunk
                    # SSE sends lines starting with "data: "
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            # blank line means dispatch event in SSE;
                            continue
                        if line.startswith("data:"):
                            data = line[len("data:"):].strip()
                            if not data:
                                continue
                            if data == "[DONE]":
                                return
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                log.debug("Failed to decode SSE chunk as JSON: %s", data)
                                yield {"raw": data}
        except requests.RequestException as e:
            log.error("Streaming request failed: %s", e)
            raise


class CustomAgent(CustomLLMCall):
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        output_type: Optional[Type[BaseModel]] = None,
        temp: float = 0.7,
        reasoning: bool = False,
        max_tokens: int = 150,
    ) -> None:
        super().__init__(model=model, temp=temp, reasoning=reasoning, max_tokens=max_tokens, output_type=output_type)
        self.name = name
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def chat_single_ask(self, user_input: str) -> str:
        msgs = self.messages + [{"role": "user", "content": user_input}]
        res = self.invoke(msgs)
        try:
            return res["choices"][-1]["message"]["content"]
        except Exception:
            log.error("Unexpected response structure: %s", res)
            raise

    def chat_conversation(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response = self.invoke(self.messages)
        try:
            content = response["choices"][-1]["message"]["content"]
            self.messages.append({"role": "assistant", "content": content})
            return content
        except Exception:
            log.error("Bad response while continuing conversation: %s", response)
            raise

    def stream_chat_conversation(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        assistant_response: List[str] = []
        for data in self.stream_invoke(self.messages):
            delta = None
            try:
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
            except Exception:
                log.error("unable to stream content")
            if delta:
                assistant_response.append(delta)
                print(delta, end="", flush=True)
        assistant_message = "".join(assistant_response)
        self.messages.append({"role": "assistant", "content": assistant_message})
        print()
        return assistant_message

if __name__=="__main__":
    agent=CustomAgent(
        name="MyCustomAgent",
        model="meta-llama/llama-3.3-8b-instruct:free",
        system_prompt="You are an assistant who replies with very little words in english",
        max_tokens=150
        )
    print("Hi , welcome to the CustomAgent chat ")
    print("send 'exit' or 'end' to close the chat ")
    while True:
        user_input=input("User Input : ")
        if(user_input in("end","End","END","exit","Exit","EXIT","stop","Stop","STOP")):
            print("Chat session closed , thankss....")
            break
        print("CustomAgent: ",end="")
        agent.stream_chat_conversation(user_input=user_input)