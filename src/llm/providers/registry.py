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
    # 新增：可选代理配置（requests 兼容的 proxies 字典），例如 {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    proxies: Optional[Dict[str, str]] = None


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
            # 解析代理配置：优先读取 providers.<name>.proxies，其次读取 http_proxy/https_proxy/no_proxy
            proxies_cfg: Optional[Dict[str, str]] = None
            loc_section = prov_loc.get(pname, {}) if isinstance(prov_loc, dict) else {}
            if isinstance(loc_section, dict):
                if isinstance(loc_section.get("proxies"), dict):
                    # 直接使用 proxies 字段
                    proxies_cfg = {k: v for k, v in loc_section.get("proxies", {}).items() if isinstance(v, str) and v}
                else:
                    http_p = loc_section.get("http_proxy")
                    https_p = loc_section.get("https_proxy")
                    no_p = loc_section.get("no_proxy")
                    tmp: Dict[str, str] = {}
                    if isinstance(http_p, str) and http_p:
                        tmp["http"] = http_p
                    if isinstance(https_p, str) and https_p:
                        tmp["https"] = https_p
                    if isinstance(no_p, str) and no_p:
                        # 非标准键，用于在运行时写入环境变量 NO_PROXY
                        tmp["no_proxy"] = no_p
                    if tmp:
                        proxies_cfg = tmp

            self.providers[pname] = ProviderConfig(
                name=pname,
                type=merged.get("type", "openai"),
                base_url=merged.get("base_url", ""),
                api_key=merged.get("api_key"),
                proxies=proxies_cfg,
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
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30, proxies: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.proxies = proxies
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
        resp = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout, proxies=self.proxies)
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
        # 处理 no_proxy：如果在本地配置中提供，则写入环境变量供 requests/urllib3 使用
        proxies = None
        if self.provider.proxies:
            proxies = {k: v for k, v in self.provider.proxies.items() if k in ("http", "https", "all")}
            no_proxy_val = self.provider.proxies.get("no_proxy") if isinstance(self.provider.proxies, dict) else None
            if isinstance(no_proxy_val, str) and no_proxy_val:
                os.environ["NO_PROXY"] = no_proxy_val
        self.client = OpenAICompatClient(base_url=self.provider.base_url, api_key=self.provider.api_key, model=self.route.model, proxies=proxies)

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str] = None, extra: Optional[Dict] = None) -> Dict:
        return self.client.chat(messages=messages, tools=tools, tool_choice=tool_choice, extra=extra)