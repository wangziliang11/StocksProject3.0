
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import requests
import yaml
import asyncio

# MCP Python SDK（若未安装，requirements.txt 已包含 mcp>=1.2.0）
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except Exception:  # 运行环境中缺失时延迟报错
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class MCPRouter:
    """
    A minimal MCP adapter that routes to OpenAI-compatible HTTP endpoints
    (SiliconFlow, Doubao/Ark, Qwen/DashScope compatible mode, OpenAI, DeepSeek, etc.).

    It also bridges to real MCP servers for tool calls via the Python SDK (stdio transport),
    if configured in YAML or environment variables.

    It reads merged configs from:
    - models.yaml (required)
    - routing.yaml (optional)
    - models.local.yaml (optional, overrides providers/api_key/base_url and routing, and mcp servers)
    - routing.local.yaml (optional, overrides routing)
    """

    def __init__(self, route_name: str = "default"):
        self.route_name = route_name
        # cache merged config
        self._providers: Dict[str, Any] = {}
        self._routing: Dict[str, Any] = {}
        self._mcp: Dict[str, Any] = {}
        self._load_and_merge_configs()

    def _load_and_merge_configs(self) -> None:
        models_cfg = _load_yaml_file(ROOT / "models.yaml")
        routing_cfg = _load_yaml_file(ROOT / "routing.yaml")
        models_local = _load_yaml_file(ROOT / "models.local.yaml")
        routing_local = _load_yaml_file(ROOT / "routing.local.yaml")

        # providers: models.yaml + models.local.yaml.providers override
        providers = (models_cfg.get("providers") or {}).copy()
        local_providers = (models_local.get("providers") or {})
        for name, lp in local_providers.items():
            if name in providers and isinstance(providers[name], dict):
                providers[name] = _deep_merge(providers[name], lp or {})
            else:
                providers[name] = lp or {}
        self._providers = providers

        # routing precedence: models.yaml.routing < routing.yaml < models.local.yaml.routing < routing.local.yaml
        r = (models_cfg.get("routing") or {}).copy()
        r = _deep_merge(r, routing_cfg.get("routing") or {})
        r = _deep_merge(r, models_local.get("routing") or {})
        r = _deep_merge(r, routing_local.get("routing") or {})
        self._routing = r

        # mcp servers config (optional)
        mcp_cfg = (models_cfg.get("mcp") or {})
        mcp_cfg = _deep_merge(mcp_cfg, models_local.get("mcp") or {})
        self._mcp = mcp_cfg

    # ---------- OpenAI-compatible chat routing ----------
    def _resolve_endpoint(self) -> Dict[str, str]:
        route = self._routing.get(self.route_name) or {}
        provider_name = route.get("provider")
        model_name = route.get("model")
        if not provider_name or not model_name:
            raise ValueError(f"未找到路由或模型: route={self.route_name}")
        p = self._providers.get(provider_name) or {}
        base_url = (p.get("base_url") or "").rstrip("/")
        api_key = p.get("api_key") or os.getenv("API_KEY")
        if not base_url:
            raise ValueError(f"提供商 {provider_name} 未配置 base_url")
        # OpenAI-compatible chat completions endpoint
        endpoint = f"{base_url}/chat/completions"
        return {
            "endpoint": endpoint,
            "api_key": api_key or "",
            "model": model_name,
        }

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            resolved = self._resolve_endpoint()
            headers = {
                "Content-Type": "application/json",
            }
            if resolved["api_key"]:
                headers["Authorization"] = f"Bearer {resolved['api_key']}"

            payload: Dict[str, Any] = {
                "model": resolved["model"],
                "messages": messages,
            }
            if tools:
                payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
            if extra and isinstance(extra, dict):
                payload.update(extra)

            resp = requests.post(resolved["endpoint"], headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # 直接返回 OpenAI 兼容响应，供上层解析（content/tool_calls 等）
            return data
        except Exception as e:
            # 失败时返回一个兼容结构，便于 UI 直接展示错误
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"MCPRouter 调用失败: {type(e).__name__}: {e}",
                        }
                    }
                ]
            }

    # ---------- Real MCP client bridge (tool discovery & execution) ----------
    def _resolve_mcp_server(self) -> Optional[Dict[str, Any]]:
        """Resolve MCP server launch config from YAML or environment variables.
        Priority: routing[route_name].mcp_server -> mcp.default_server -> env.
        Structure expected:
        mcp:
          default_server: stocks_tools
          servers:
            stocks_tools:
              command: "uv"
              args: ["run", "mcp-simple-tool"]
              env: { }
        """
        # route-scoped server name
        route = self._routing.get(self.route_name) or {}
        server_name = route.get("mcp_server")

        mcp_cfg = self._mcp or {}
        if not server_name:
            server_name = mcp_cfg.get("default_server")

        servers = (mcp_cfg.get("servers") or {}) if isinstance(mcp_cfg, dict) else {}
        srv = (servers.get(server_name) or {}) if server_name else {}

        # env overrides
        cmd = srv.get("command") or os.getenv("MCP_SERVER_COMMAND")
        args = srv.get("args") or json.loads(os.getenv("MCP_SERVER_ARGS", "[]"))
        env = srv.get("env") or {}

        if not cmd:
            # not configured
            return None
        if not isinstance(args, list):
            try:
                args = list(args)
            except Exception:
                args = []
        return {"command": cmd, "args": args, "env": env}

    async def _with_session(self, coro_fn):
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            raise RuntimeError("MCP Python SDK 未安装或不可用")
        cfg = self._resolve_mcp_server()
        if not cfg:
            raise RuntimeError("未配置 MCP 服务端（缺少 mcp.servers 或 MCP_SERVER_COMMAND）")
        params = StdioServerParameters(command=cfg["command"], args=cfg.get("args") or [], env=cfg.get("env") or {})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await coro_fn(session)

    def mcp_list_tools_openai_schema(self) -> List[Dict[str, Any]]:
        """List tools from MCP server and convert to OpenAI tools schema.
        Returns empty list if not configured.
        """
        async def _list(session: Any) -> List[Dict[str, Any]]:  # type: ignore
            resp = await session.list_tools()
            tools = []
            for t in getattr(resp, "tools", []) or []:
                name = getattr(t, "name", None)
                desc = getattr(t, "description", None)
                # Attempt to get JSON schema for parameters if available
                params = getattr(t, "inputSchema", None)
                if params is None:
                    # fallback to empty schema
                    params = {"type": "object", "properties": {}}
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc or "",
                        "parameters": params,
                    }
                })
            return tools

        try:
            return asyncio.run(self._with_session(_list))
        except Exception:
            return []

    def mcp_call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async def _call(session: Any) -> Dict[str, Any]:  # type: ignore
            result = await session.call_tool(name, arguments or {})
            # Try structured content first
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                return {"ok": True, "data": structured}
            # Fallback to unstructured text content
            content = getattr(result, "content", None) or []
            if content:
                # try first content block's text
                blk = content[0]
                text = getattr(blk, "text", None) if hasattr(blk, "text") else None
                if text:
                    return {"ok": True, "data": {"text": text}}
            return {"ok": True, "data": {"raw": str(result)}}

        try:
            return asyncio.run(self._with_session(_call))
        except Exception as e:
            return {"ok": False, "error": f"mcp_call_tool failed: {type(e).__name__}: {e}"}