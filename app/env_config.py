from __future__ import annotations

from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    """读取简单的 KEY=VALUE 格式配置，忽略注释和空行。"""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def save_env_values(path: Path, updates: dict[str, str]) -> None:
    """更新指定配置项，并保留 .env 中的其他内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    saved_keys: set[str] = set()
    output_lines: list[str] = []
    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            output_lines.append(raw_line)
            continue

        key = raw_line.split("=", 1)[0].strip()
        if key not in updates:
            output_lines.append(raw_line)
            continue

        if key in saved_keys:
            continue
        output_lines.append(f"{key}={format_env_value(updates[key])}")
        saved_keys.add(key)

    for key, value in updates.items():
        if key not in saved_keys:
            output_lines.append(f"{key}={format_env_value(value)}")

    path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")


def format_env_value(value: str) -> str:
    if not value or any(char.isspace() for char in value) or "#" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value
