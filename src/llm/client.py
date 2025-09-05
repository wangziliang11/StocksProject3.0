from dataclasses import dataclass
from typing import Optional, Dict
import requests

@dataclass
class OpenAICompatConfig:
    base_url: str
    api_key: str
    model: str
    timeout: int = 30

class LLMClient:
    """
    兼容 OpenAI 接口的任意大模型供应商（如：通义千问、DeepSeek、豆包、腾讯混元、硅基流动等）
    通过自定义 base_url / model / api_key 即可调用。
    """
    def __init__(self, config: OpenAICompatConfig):
        self.cfg = config
        self.headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: list, extra: Optional[Dict] = None) -> str:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.3,
            "stream": False,
        }
        if extra:
            payload.update(extra)
        resp = requests.post(url, headers=self.headers, json=payload, timeout=self.cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI 兼容返回格式
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)