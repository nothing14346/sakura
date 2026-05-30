from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MCP_CALL_TIMEOUT_SECONDS = 30.0
SUPPORTED_MCP_TRANSPORTS = {"stdio", "sse"}
SUPPORTED_MCP_RISKS = {"low", "medium", "high"}


@dataclass(frozen=True)
class MCPServerConfig:
    """单个 MCP Server 的连接和工具暴露配置。"""

    name: str
    transport: str
    enabled: bool = True
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    name_prefix: str | None = None
    call_timeout: float | None = None
    risk: str = "medium"
    requires_confirmation: bool | None = None

    def effective_name_prefix(self) -> str:
        if self.name_prefix is not None:
            return self.name_prefix
        normalized = "".join(char if char.isalnum() or char == "_" else "_" for char in self.name)
        return f"{normalized}__"

    def effective_call_timeout(self, default_timeout: float) -> float:
        return self.call_timeout if self.call_timeout is not None else default_timeout

    def effective_requires_confirmation(self) -> bool:
        if self.requires_confirmation is not None:
            return self.requires_confirmation
        return self.risk != "low"


@dataclass(frozen=True)
class MCPConfig:
    """MCP 总配置；文件不存在时保持禁用，避免影响主流程。"""

    enabled: bool = False
    default_call_timeout: float = DEFAULT_MCP_CALL_TIMEOUT_SECONDS
    servers: list[MCPServerConfig] = field(default_factory=list)


def load_mcp_config(path: Path) -> MCPConfig:
    """读取 data/config/mcp.yaml；不存在时静默禁用 MCP。"""

    if not path.exists():
        return MCPConfig()

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，无法读取 MCP 配置。请安装 requirements.txt。") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return MCPConfig()
    if not isinstance(raw, dict):
        raise ValueError("MCP 配置顶层必须是 YAML object。")

    enabled = _optional_bool(raw.get("enabled"), default=True)
    default_timeout = _optional_positive_float(
        raw.get("default_call_timeout"),
        DEFAULT_MCP_CALL_TIMEOUT_SECONDS,
        "default_call_timeout",
    )
    servers = _parse_servers(raw.get("servers"), default_timeout)
    return MCPConfig(
        enabled=enabled,
        default_call_timeout=default_timeout,
        servers=servers,
    )


def _parse_servers(raw_servers: Any, default_timeout: float) -> list[MCPServerConfig]:
    if raw_servers is None:
        return []

    items: list[tuple[str, Any]] = []
    if isinstance(raw_servers, dict):
        items = [(str(name), value) for name, value in raw_servers.items()]
    elif isinstance(raw_servers, list):
        for index, item in enumerate(raw_servers):
            if not isinstance(item, dict):
                raise ValueError(f"servers[{index}] 必须是 YAML object。")
            name = item.get("name") or item.get("id") or f"server_{index + 1}"
            items.append((str(name), item))
    else:
        raise ValueError("servers 必须是 object 或 list。")

    return [_parse_server(name, data, default_timeout) for name, data in items]


def _parse_server(name: str, data: Any, default_timeout: float) -> MCPServerConfig:
    if not isinstance(data, dict):
        raise ValueError(f"MCP Server {name} 配置必须是 YAML object。")

    transport = str(data.get("transport") or data.get("type") or "").strip().lower()
    if not transport:
        transport = "sse" if data.get("url") else "stdio"
    if transport not in SUPPORTED_MCP_TRANSPORTS:
        raise ValueError(f"MCP Server {name} transport 只支持 stdio 或 sse。")

    risk = str(data.get("risk") or "medium").strip().lower()
    if risk not in SUPPORTED_MCP_RISKS:
        raise ValueError(f"MCP Server {name} risk 只支持 low、medium 或 high。")

    call_timeout = data.get("call_timeout")
    parsed_call_timeout = (
        None
        if call_timeout is None
        else _optional_positive_float(call_timeout, default_timeout, f"{name}.call_timeout")
    )

    return MCPServerConfig(
        name=name.strip(),
        transport=transport,
        enabled=_optional_bool(data.get("enabled"), default=True),
        command=str(data.get("command") or "").strip(),
        args=_string_list(data.get("args"), f"{name}.args"),
        env=_string_dict(data.get("env"), f"{name}.env"),
        url=str(data.get("url") or "").strip(),
        headers=_string_dict(data.get("headers"), f"{name}.headers"),
        name_prefix=_optional_string(data.get("name_prefix")),
        call_timeout=parsed_call_timeout,
        risk=risk,
        requires_confirmation=_optional_bool_or_none(data.get("requires_confirmation")),
    )


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("布尔配置项必须是 true 或 false。")


def _optional_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return _optional_bool(value, default=False)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError("字符串配置项必须是 string。")


def _optional_positive_float(value: Any, default: float, field_name: str) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是正数。")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{field_name} 必须大于 0。")
    return result


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是 list。")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} 只能包含字符串。")
        result.append(item)
    return result


def _string_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 object。")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{field_name} 的键和值都必须是字符串。")
        result[key] = item
    return result
