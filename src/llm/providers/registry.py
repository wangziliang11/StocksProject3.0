from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import requests
import yaml
import os


@dataclass
class ProviderConfig:
    name: str
    type: str  # 'openai' (OpenAI-compatible)
    base_url: str
    api_key: Optional[str] = None


@dataclass
class Route:
    name: str
    provider: str
    model: str


class ProviderRegistry:
    """
    解析 models.yaml 与 models.local.yaml，生成 provider 映射与路由表。
    - 公共配置(models.yaml)可提交；
    - 本地敏感信息(models.local.yaml)仅包含 api_key/覆盖 base_url（已 gitignore）。
    - 额外支持独立 routing.yaml 与 routing.local.yaml（如存在），用于声明与覆盖路由。
    """
    def __init__(self, public_cfg_path: str = "models.yaml", local_cfg_path: str = "models.local.yaml", routing_cfg_path: str = "routing.yaml", routing_local_cfg_path: str = "routing.local.yaml"):
        self.public_cfg_path = public_cfg_path
        self.local_cfg_path = local_cfg_path
        self.routing_cfg_path = routing_cfg_path
        self.routing_local_cfg_path = routing_local_cfg_path
        self.providers: Dict[str, ProviderConfig] = {}
        self.routes: Dict[str, Route] = {}
        self._load()

    def _load_yaml(self, path: str) -> Dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load(self):
        pub = self._load_yaml(self.public_cfg_path)
        loc = self._load_yaml(self.local_cfg_path)
        route_pub_file = self._load_yaml(self.routing_cfg_path)
        route_loc_file = self._load_yaml(self.routing_local_cfg_path)

        # providers
        prov_pub = pub.get("providers", {})
        prov_loc = loc.get("providers", {})
        for pname, pconf in prov_pub.items():
            merged = dict(pconf)
            if pname in prov_loc:
                merged.update({k: v for k, v in prov_loc[pname].items() if v is not None})
            self.providers[pname] = ProviderConfig(
                name=pname,
                type=merged.get("type", "openai"),
                base_url=merged.get("base_url", ""),
                api_key=merged.get("api_key"),
            )

        # routes：合并顺序 => models.yaml.routing -> routing.yaml -> models.local.yaml.routing -> routing.local.yaml
        routing_pub = pub.get("routing", {})
        routing_file_pub = route_pub_file.get("routing", route_pub_file) or {}
        routing_loc = loc.get("routing", {})
        routing_file_loc = route_loc_file.get("routing", route_loc_file) or {}
        merged_routes: Dict[str, Dict[str, str]] = {}
        for src in (routing_pub, routing_file_pub, routing_loc, routing_file_loc):
            for rname, rconf in src.items():
                merged_routes[rname] = {"provider": rconf["provider"], "model": rconf["model"]}
        for rname, rconf in merged_routes.items():
            self.routes[rname] = Route(name=rname, provider=rconf["provider"], model=rconf["model"])

    def get_route(self, name: str = "default") -> Route:
        if name in self.routes:
            return self.routes[name]
        # fallback to any route if default not found
        return next(iter(self.routes.values()))

    def get_provider(self, name: str) -> ProviderConfig:
        return self.providers[name]


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str] = None, extra: Optional[Dict] = None) -> Dict:
        # 仅追加统一的 "/chat/completions"，版本路径由 base_url 指定（例如：/v1 或 /api/v3）
        url = self.base_url + "/chat/completions"
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.3, "stream": False}
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if extra:
            payload.update(extra)
        resp = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class LLMRouter:
    """
    通过 routing.yaml（合并到 models.yaml 的 routing 字段中）选择 provider 与 model，并提供统一 chat 接口。
    支持 function calling（tools）。
    """
    def __init__(self, registry: ProviderRegistry, route_name: str = "default"):
        self.registry = registry
        self.route = registry.get_route(route_name)
        self.provider = registry.get_provider(self.route.provider)
        assert self.provider.api_key, f"Provider {self.provider.name} api_key 未配置，请在 models.local.yaml 设置"
        self.client = OpenAICompatClient(base_url=self.provider.base_url, api_key=self.provider.api_key, model=self.route.model)

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str] = None, extra: Optional[Dict] = None) -> Dict:
        return self.client.chat(messages=messages, tools=tools, tool_choice=tool_choice, extra=extra)