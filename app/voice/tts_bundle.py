from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


ProgressCallback = Callable[[int], None]
StatusCallback = Callable[[str], None]
MigrationProgressCallback = Callable[["TTSBundleMigrationProgress"], None]

class DownloadCancelledError(Exception):
    """用户主动取消下载时抛出此异常，调用方据此判断是用户取消而非真正的错误。"""

_DOWNLOAD_CHUNK_SIZE = 512 * 1024
_HASH_CHUNK_SIZE = 4 * 1024 * 1024
_VERIFY_PROGRESS_END = 10
_DOWNLOAD_PROGRESS_END = 70
_SEVEN_ZIP_COMMANDS = ("7zz.exe", "7za.exe", "7z.exe", "7zz", "7za", "7z")
_WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class UrlOpenCallable(Protocol):
    def __call__(self, request: urllib.request.Request, timeout: int) -> object:
        ...


@dataclass(frozen=True)
class TTSBundleEntry:
    key: str
    label: str
    filename: str
    download_url: str
    size: int
    sha256: str
    provider: str = "gpt-sovits"


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vram_gb: float


@dataclass(frozen=True)
class TTSBundleMigration:
    """内置 TTS 整合包从旧深目录迁移到短目录所需的信息。"""

    entry: TTSBundleEntry
    source_dir: Path
    target_dir: Path


@dataclass(frozen=True)
class TTSBundleMigrationProgress:
    """TTS 整合包迁移进度。"""

    entry: TTSBundleEntry
    current_file: str
    completed_files: int
    total_files: int
    copied_bytes: int
    total_bytes: int


GENIE_TTS = TTSBundleEntry(
    key="genie_tts_server",
    label="Genie TTS CPU 整合包",
    filename="Genie-TTS Server.7z",
    download_url=(
        "https://www.modelscope.cn/models/twillzxy/genie-tts-server/"
        "resolve/master/Genie-TTS%20Server.7z"
    ),
    size=1041915345,
    sha256="8f06077b6102aa29f1c9473926db9a74890d627f077393aa8ebb928b52f15de1",
    provider="genie-tts",
)
GPT_SOVITS_STANDARD = TTSBundleEntry(
    key="gpt_sovits_v2pro",
    label="GPT-SoVITS v2pro 通用整合包",
    filename="GPT-SoVITS-v2pro-20250604.7z",
    download_url=(
        "https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/"
        "resolve/master/GPT-SoVITS-v2pro-20250604.7z"
    ),
    size=8185086602,
    sha256="bd60d0796553ff05d8568136e199c13e0dc22ebe2ed24273134e34ed6f215cd6",
)
GPT_SOVITS_NVIDIA50 = TTSBundleEntry(
    key="gpt_sovits_nvidia50",
    label="GPT-SoVITS v2pro NVIDIA 50 系整合包",
    filename="GPT-SoVITS-v2pro-20250604-nvidia50.7z",
    download_url=(
        "https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/"
        "resolve/master/GPT-SoVITS-v2pro-20250604-nvidia50.7z"
    ),
    size=8835144925,
    sha256="97b4edcd451c42357db7e26e6c1c877ca5d85144fe97beaff6d7005d35bee008",
)
GPT_SOVITS_BUNDLES = (GPT_SOVITS_STANDARD, GPT_SOVITS_NVIDIA50)
TTS_BUNDLES = (GENIE_TTS, GPT_SOVITS_STANDARD, GPT_SOVITS_NVIDIA50)
MIN_GPT_SOVITS_VRAM_GB = 6.0
_GPT_SOVITS_VRAM_TOLERANCE_GB = 0.25
_SHORT_BUNDLE_DIRS = {
    GENIE_TTS.key: "cpu",
    GPT_SOVITS_STANDARD.key: "gpt",
    GPT_SOVITS_NVIDIA50.key: "g50",
}
_MIGRATING_DIR_NAME = ".migrating"
_MIGRATION_STATE_FILE = ".sakura_migration.json"
_MIGRATION_TEMP_SUFFIX = ".__sakura_tmp__"
_MIGRATION_COPY_CHUNK_SIZE = 16 * 1024 * 1024


def format_platform_summary() -> str:
    try:
        return platform.platform(aliased=True, terse=True)
    except Exception:
        return f"{platform.system()} {platform.release()}"


def list_nvidia_gpus() -> list[GPUInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    cmd = [
        exe,
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    kwargs: dict[str, object] = {
        "args": cmd,
        "capture_output": True,
        "text": True,
        "timeout": 8,
    }
    if sys.platform == "win32" and _WIN_NO_WINDOW:
        kwargs["creationflags"] = _WIN_NO_WINDOW
    try:
        result = subprocess.run(**kwargs)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    gpus: list[GPUInfo] = []
    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        name, _, memory = line.partition(",")
        try:
            memory_mib = float(re.sub(r"[^\d.]", "", memory) or "0")
        except ValueError:
            memory_mib = 0.0
        gpus.append(GPUInfo(name=name.strip() or "NVIDIA GPU", vram_gb=round(memory_mib / 1024, 2)))
    return gpus


def format_gpu_summary(gpus: list[GPUInfo]) -> str:
    if not gpus:
        return "未检测到 NVIDIA GPU，将推荐 Genie TTS CPU 整合包。"
    return "\n".join(f"#{i} NVIDIA | {gpu.name} | {gpu.vram_gb} GB" for i, gpu in enumerate(gpus, start=1))


def format_bundle_size(entry: TTSBundleEntry) -> str:
    gb = entry.size / 1_000_000_000
    if gb >= 1:
        return f"约 {gb:.1f} GB"
    return f"约 {entry.size / 1_000_000:.0f} MB"


def format_bundle_label(entry: TTSBundleEntry) -> str:
    return f"{entry.label}（{format_bundle_size(entry)}）"


def recommend_gpt_sovits_bundle(gpus: list[GPUInfo] | None = None) -> TTSBundleEntry:
    gpus = list_nvidia_gpus() if gpus is None else gpus
    if any(_is_rtx_50_series(gpu.name) for gpu in gpus):
        return GPT_SOVITS_NVIDIA50
    return GPT_SOVITS_STANDARD


def recommend_tts_bundle(gpus: list[GPUInfo] | None = None) -> TTSBundleEntry:
    gpus = list_nvidia_gpus() if gpus is None else gpus
    capable_nvidia = [gpu for gpu in gpus if _has_gpt_sovits_vram(gpu)]
    if not capable_nvidia:
        return GENIE_TTS
    if any(_is_rtx_50_series(gpu.name) for gpu in capable_nvidia):
        return GPT_SOVITS_NVIDIA50
    return GPT_SOVITS_STANDARD


def default_bundle_work_dir(entry: TTSBundleEntry, base_dir: Path) -> Path:
    """返回整合包对应的工作目录；已安装时解析真实解压根目录，未安装时返回预期目录。"""

    short_dir = _short_bundle_install_dir(entry, base_dir)
    if _is_installed_bundle_ready(short_dir):
        try:
            return _resolve_extracted_root(short_dir)
        except OSError:
            pass

    legacy_dir = _legacy_bundle_install_dir(entry, base_dir)
    if _is_installed_bundle_ready(legacy_dir):
        try:
            return _resolve_extracted_root(legacy_dir)
        except OSError:
            pass
    return short_dir


def default_provider_bundle_work_dir(provider: str, base_dir: Path) -> Path | None:
    """按 provider 选择整合包默认工作目录，自定义外部 provider 不返回目录。"""

    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"genie", "genie-tts", "genietts"}:
        return default_bundle_work_dir(GENIE_TTS, base_dir)
    if normalized not in {"gpt-sovits", "gpt-so-vits", "gptsovits"}:
        return None

    for entry in GPT_SOVITS_BUNDLES:
        if _is_bundle_installed(entry, base_dir):
            return default_bundle_work_dir(entry, base_dir)
    recommended = recommend_gpt_sovits_bundle()
    return default_bundle_work_dir(recommended, base_dir)


def is_provider_bundle_work_dir(path: Path, base_dir: Path) -> bool:
    """判断路径是否位于内置 TTS 整合包安装目录下。"""

    resolved = path.resolve()
    for root in (
        (base_dir / "tts").resolve(),
        (base_dir / "data" / "tts_bundles" / "installed").resolve(),
    ):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def find_pending_bundle_migrations(base_dir: Path, provider: str | None = None) -> list[TTSBundleMigration]:
    """查找需要从旧深目录迁移到短目录的内置 TTS 整合包。"""

    migrations: list[TTSBundleMigration] = []
    for entry in _entries_for_provider(provider):
        target_dir = _short_bundle_install_dir(entry, base_dir)
        if target_dir.exists():
            if _is_installed_bundle_ready(target_dir):
                _cleanup_migration_work_dir(TTSBundleMigration(entry=entry, source_dir=target_dir, target_dir=target_dir))
            continue
        legacy_dir = _legacy_bundle_install_dir(entry, base_dir)
        if not _is_installed_bundle_ready(legacy_dir):
            continue
        try:
            source_dir = _resolve_extracted_root(legacy_dir)
        except OSError:
            continue
        migrations.append(TTSBundleMigration(entry=entry, source_dir=source_dir, target_dir=target_dir))
    return migrations


def migrate_bundle_to_short_path(
    migration: TTSBundleMigration,
    *,
    on_progress: MigrationProgressCallback | None = None,
) -> Path:
    """执行单个整合包迁移；失败时保留旧目录和中间目录供下次续迁。"""

    if migration.target_dir.exists():
        _cleanup_migration_work_dir(migration)
        return _resolve_extracted_root(migration.target_dir)
    migration.target_dir.parent.mkdir(parents=True, exist_ok=True)

    if _try_fast_migration_rename(migration):
        source_parent = migration.source_dir.parent
        _remove_empty_legacy_parents(source_parent)
        return migration.target_dir.resolve()

    work_dir = _migration_work_dir(migration)
    work_dir.mkdir(parents=True, exist_ok=True)
    plan = _migration_file_plan(migration.source_dir)
    total_files = len(plan)
    total_bytes = sum(size for _relative, size in plan)
    copied_files = 0
    copied_bytes = 0
    _write_migration_state(
        migration,
        total_files=total_files,
        total_bytes=total_bytes,
        completed_files=0,
        copied_bytes=0,
        current_file="",
    )

    for relative_path, size in plan:
        source = migration.source_dir / relative_path
        target = work_dir / relative_path
        current_file = relative_path.as_posix()
        if _migration_target_file_ready(source, target, size):
            copied_files += 1
            copied_bytes += size
            _emit_migration_progress(
                on_progress,
                migration,
                current_file=current_file,
                completed_files=copied_files,
                total_files=total_files,
                copied_bytes=copied_bytes,
                total_bytes=total_bytes,
            )
            _write_migration_state(
                migration,
                total_files=total_files,
                total_bytes=total_bytes,
                completed_files=copied_files,
                copied_bytes=copied_bytes,
                current_file=current_file,
            )
            continue

        _copy_file_resumable(source, target)
        copied_files += 1
        copied_bytes += size
        _emit_migration_progress(
            on_progress,
            migration,
            current_file=current_file,
            completed_files=copied_files,
            total_files=total_files,
            copied_bytes=copied_bytes,
            total_bytes=total_bytes,
        )
        _write_migration_state(
            migration,
            total_files=total_files,
            total_bytes=total_bytes,
            completed_files=copied_files,
            copied_bytes=copied_bytes,
            current_file=current_file,
        )

    _migration_state_path(work_dir).unlink(missing_ok=True)
    work_dir.replace(migration.target_dir)
    source_parent = migration.source_dir.parent
    shutil.rmtree(migration.source_dir, ignore_errors=True)
    _remove_empty_legacy_parents(source_parent)
    return migration.target_dir.resolve()


def normalize_bundle_work_dir(path: Path | None, base_dir: Path) -> Path | None:
    """把旧内置整合包路径归一到已迁移后的短路径。"""

    if path is None:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for entry in TTS_BUNDLES:
        target_dir = _short_bundle_install_dir(entry, base_dir)
        if not _is_installed_bundle_ready(target_dir):
            continue
        legacy_dir = _legacy_bundle_install_dir(entry, base_dir)
        try:
            resolved.relative_to(legacy_dir.resolve())
        except ValueError:
            continue
        return _resolve_extracted_root(target_dir)
    return path


def download_and_extract_bundle(
    entry: TTSBundleEntry,
    base_dir: Path,
    *,
    check_cancel: Callable[[], None] | None = None,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
    urlopen: UrlOpenCallable = urllib.request.urlopen,
    extractor: Callable[[Path, Path], str | None] | None = None,
) -> Path:
    bundle_base = base_dir / "tts"
    downloads_dir = bundle_base / "_dl"
    tmp_dir = bundle_base / "_tmp" / entry.key
    installed_dir = _short_bundle_install_dir(entry, base_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive = downloads_dir / entry.filename

    _emit_status(on_status, "verify")
    _emit_progress(on_progress, 0)
    if _archive_verification_error(archive, entry, on_progress=on_progress) is not None:
        _emit_status(on_status, "download")
        _download_archive(entry, archive, on_progress=on_progress, urlopen=urlopen, check_cancel=check_cancel)
    _emit_progress(on_progress, _DOWNLOAD_PROGRESS_END)

    _emit_status(on_status, "extract")
    _reset_extract_dir(tmp_dir)
    extract = extractor or _extract_archive
    error = extract(archive, tmp_dir)
    if error is not None:
        raise RuntimeError(f"解压 TTS 整合包失败：{error}")
    work_dir = _replace_installed_bundle_from_extract(tmp_dir, installed_dir)
    _emit_status(on_status, "cleanup")
    _cleanup_archive(archive)
    _emit_progress(on_progress, 100)
    return work_dir


def cleanup_stale_download_archives(base_dir: Path) -> list[Path]:
    """清理旧版本解压成功后遗留在下载目录里的整合包压缩包。"""

    cleaned: list[Path] = []
    for entry in TTS_BUNDLES:
        if not _is_bundle_installed(entry, base_dir):
            continue
        for downloads_dir in (base_dir / "tts" / "_dl", base_dir / "data" / "tts_bundles" / "downloads"):
            archive = downloads_dir / entry.filename
            if not archive.is_file():
                continue
            _cleanup_archive(archive)
            cleaned.append(archive)
    return cleaned


def _entries_for_provider(provider: str | None) -> tuple[TTSBundleEntry, ...]:
    if provider is None:
        return TTS_BUNDLES
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"genie", "genie-tts", "genietts"}:
        return (GENIE_TTS,)
    if normalized in {"gpt-sovits", "gpt-so-vits", "gptsovits"}:
        return GPT_SOVITS_BUNDLES
    return ()


def _short_bundle_install_dir(entry: TTSBundleEntry, base_dir: Path) -> Path:
    short_name = _SHORT_BUNDLE_DIRS.get(entry.key, entry.key)
    return base_dir / "tts" / short_name


def _legacy_bundle_install_dir(entry: TTSBundleEntry, base_dir: Path) -> Path:
    return base_dir / "data" / "tts_bundles" / "installed" / entry.key


def _is_bundle_installed(entry: TTSBundleEntry, base_dir: Path) -> bool:
    return (
        _is_installed_bundle_ready(_short_bundle_install_dir(entry, base_dir))
        or _is_installed_bundle_ready(_legacy_bundle_install_dir(entry, base_dir))
    )


def _replace_installed_bundle_from_extract(tmp_dir: Path, installed_dir: Path) -> Path:
    root = _resolve_extracted_root(tmp_dir)
    if installed_dir.exists():
        shutil.rmtree(installed_dir, ignore_errors=True)
    installed_dir.parent.mkdir(parents=True, exist_ok=True)

    if root == tmp_dir.resolve():
        installed_dir.mkdir(parents=True, exist_ok=True)
        for child in list(tmp_dir.iterdir()):
            shutil.move(str(child), str(installed_dir / child.name))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return installed_dir.resolve()

    shutil.move(str(root), str(installed_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return installed_dir.resolve()


def _migration_work_dir(migration: TTSBundleMigration) -> Path:
    return migration.target_dir.parent / _MIGRATING_DIR_NAME / migration.entry.key


def _migration_state_path(work_dir: Path) -> Path:
    return work_dir / _MIGRATION_STATE_FILE


def _cleanup_migration_work_dir(migration: TTSBundleMigration) -> None:
    work_dir = _migration_work_dir(migration)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


def _try_fast_migration_rename(migration: TTSBundleMigration) -> bool:
    """同盘且没有续迁目录时优先尝试目录改名，失败则交给可恢复复制。"""

    if _migration_work_dir(migration).exists():
        return False
    try:
        if migration.source_dir.resolve().anchor != migration.target_dir.resolve().anchor:
            return False
        migration.source_dir.rename(migration.target_dir)
    except OSError:
        return False
    _cleanup_migration_work_dir(migration)
    return True


def _migration_file_plan(source_dir: Path) -> list[tuple[Path, int]]:
    plan: list[tuple[Path, int]] = []
    for path in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
        if not path.is_file():
            continue
        if path.name == _MIGRATION_STATE_FILE or path.name.endswith(_MIGRATION_TEMP_SUFFIX):
            continue
        relative_path = path.relative_to(source_dir)
        plan.append((relative_path, path.stat().st_size))
    return plan


def _write_migration_state(
    migration: TTSBundleMigration,
    *,
    total_files: int,
    total_bytes: int,
    completed_files: int,
    copied_bytes: int,
    current_file: str,
) -> None:
    work_dir = _migration_work_dir(migration)
    work_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "entry_key": migration.entry.key,
        "source_dir": str(migration.source_dir),
        "target_dir": str(migration.target_dir),
        "work_dir": str(work_dir),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "completed_files": completed_files,
        "copied_bytes": copied_bytes,
        "current_file": current_file,
        "updated_at": time.time(),
    }
    _migration_state_path(work_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _migration_target_file_ready(source: Path, target: Path, expected_size: int) -> bool:
    if not target.is_file():
        return False
    try:
        source_stat = source.stat()
        target_stat = target.stat()
    except OSError:
        return False
    if target_stat.st_size != expected_size or target_stat.st_size != source_stat.st_size:
        return False
    # Windows/FAT/压缩包解压后的 mtime 精度可能有误差，大小一致即可续迁跳过。
    return True


def _copy_file_resumable(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}{_MIGRATION_TEMP_SUFFIX}")
    tmp_target.unlink(missing_ok=True)
    try:
        with source.open("rb") as src, tmp_target.open("wb") as dst:
            while True:
                chunk = src.read(_MIGRATION_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                dst.write(chunk)
                time.sleep(0)
        shutil.copystat(source, tmp_target)
        os.replace(tmp_target, target)
    except Exception:
        tmp_target.unlink(missing_ok=True)
        raise


def _emit_migration_progress(
    callback: MigrationProgressCallback | None,
    migration: TTSBundleMigration,
    *,
    current_file: str,
    completed_files: int,
    total_files: int,
    copied_bytes: int,
    total_bytes: int,
) -> None:
    if callback is None:
        return
    callback(
        TTSBundleMigrationProgress(
            entry=migration.entry,
            current_file=current_file,
            completed_files=completed_files,
            total_files=total_files,
            copied_bytes=copied_bytes,
            total_bytes=total_bytes,
        )
    )


def _remove_empty_legacy_parents(path: Path) -> None:
    tts_bundles_dir: Path | None = None
    for parent in path.parents:
        if parent.name == "tts_bundles":
            tts_bundles_dir = parent
            break
    if tts_bundles_dir is None:
        return

    current = path
    while True:
        try:
            current.rmdir()
        except OSError:
            break
        if current == tts_bundles_dir:
            return
        current = current.parent

    # 迁移后只清理旧整合包运行时相关的空目录；ONNX 属于用户生成数据，保留在 data 下。
    for candidate in (
        tts_bundles_dir / "downloads",
        tts_bundles_dir / "installed",
        tts_bundles_dir,
    ):
        try:
            candidate.rmdir()
        except OSError:
            continue


def _is_rtx_50_series(name: str) -> bool:
    return bool(re.search(r"\bRTX\s*50[0-9]{2}\b", name, re.IGNORECASE))


def _has_gpt_sovits_vram(gpu: GPUInfo) -> bool:
    # nvidia-smi 常把 6GB / 8GB 显卡报成 5.9x / 7.9x GB，这里保留误差余量避免误判成 CPU 包。
    return gpu.vram_gb + _GPT_SOVITS_VRAM_TOLERANCE_GB >= MIN_GPT_SOVITS_VRAM_GB


def _cleanup_archive(archive: Path) -> None:
    try:
        archive.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"TTS 整合包已解压，但清理下载压缩包失败：{exc}") from exc


def _is_installed_bundle_ready(installed_dir: Path) -> bool:
    if not installed_dir.is_dir():
        return False
    try:
        root = _resolve_extracted_root(installed_dir)
    except OSError:
        return False
    return (root / "runtime" / "python.exe").is_file()


def _emit_progress(callback: ProgressCallback | None, value: int) -> None:
    if callback is not None:
        callback(max(0, min(100, int(value))))


def _emit_status(callback: StatusCallback | None, value: str) -> None:
    if callback is not None:
        callback(value)


def _archive_verification_error(
    archive: Path,
    entry: TTSBundleEntry,
    *,
    on_progress: ProgressCallback | None = None,
) -> str | None:
    if not archive.is_file():
        return "archive is missing"
    if archive.stat().st_size != entry.size:
        return "size mismatch"
    if _sha256_file(
        archive,
        expected_size=entry.size,
        on_progress=on_progress,
        progress_start=0,
        progress_end=_VERIFY_PROGRESS_END,
    ).lower() != entry.sha256.lower():
        return "sha256 mismatch"
    return None


def _download_archive(
    entry: TTSBundleEntry,
    archive: Path,
    *,
    on_progress: ProgressCallback | None,
    urlopen: UrlOpenCallable,
    check_cancel: Callable[[], None] | None = None,
) -> None:
    part = archive.with_name(f"{archive.name}.part")
    if part.exists():
        part.unlink()
    request = urllib.request.Request(
        entry.download_url,
        headers={"User-Agent": "Sakura-Desktop-Pet/1.0"},
    )
    hasher = hashlib.sha256()
    downloaded = 0
    try:
        with urlopen(request, timeout=600) as response:  # type: ignore[attr-defined]
            with part.open("wb") as file:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    file.write(chunk)
                    hasher.update(chunk)
                    if check_cancel is not None:
                        check_cancel()
                    downloaded += len(chunk)
                    _emit_progress(
                        on_progress,
                        _VERIFY_PROGRESS_END
                        + int((_DOWNLOAD_PROGRESS_END - _VERIFY_PROGRESS_END) * downloaded / entry.size),
                    )
        if downloaded != entry.size:
            raise RuntimeError(f"文件大小不匹配：期望 {entry.size}，实际 {downloaded}")
        actual_sha256 = hasher.hexdigest()
        if actual_sha256.lower() != entry.sha256.lower():
            raise RuntimeError(f"SHA256 不匹配：期望 {entry.sha256}，实际 {actual_sha256}")
        part.replace(archive)
    except Exception:
        if part.exists():
            part.unlink()
        raise


def _sha256_file(
    path: Path,
    *,
    expected_size: int | None = None,
    on_progress: ProgressCallback | None = None,
    progress_start: int = 0,
    progress_end: int = 100,
) -> str:
    hasher = hashlib.sha256()
    total = expected_size if expected_size and expected_size > 0 else path.stat().st_size
    read_bytes = 0
    last_progress: int | None = None
    with path.open("rb") as file:
        while True:
            chunk = file.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            read_bytes += len(chunk)
            if total > 0:
                progress = progress_start + int((progress_end - progress_start) * read_bytes / total)
                if progress != last_progress:
                    _emit_progress(on_progress, progress)
                    last_progress = progress
            # 大文件哈希会持续数秒，主动让出执行权避免 Qt 前台窗口假死。
            time.sleep(0)
    _emit_progress(on_progress, progress_end)
    return hasher.hexdigest()


def _extract_archive(archive: Path, out_dir: Path) -> str | None:
    py7zz_error = _extract_with_py7zz(archive, out_dir)
    if py7zz_error is None:
        return None

    seven_zip = _seven_zip_exe()
    cli_error: str | None = None
    if seven_zip is not None:
        if py7zz_error != "missing":
            _reset_extract_dir(out_dir)
        cli_error = _extract_with_7zip(seven_zip, archive, out_dir)
        if cli_error is None:
            return None

    py7zr = _load_py7zr()
    if py7zr is None:
        return _format_extractor_missing_error(py7zz_error, cli_error)

    if cli_error is not None or py7zz_error != "missing":
        _reset_extract_dir(out_dir)
    try:
        _extract_with_py7zr(py7zr, archive, out_dir)
    except Exception as exc:
        return _format_py7zr_failure_error(py7zz_error, cli_error, exc)
    return None


def _extract_with_py7zz(archive: Path, out_dir: Path) -> str | None:
    try:
        py7zz = importlib.import_module("py7zz")
    except ImportError:
        return "missing"
    try:
        py7zz.extract_archive(str(archive), str(out_dir))
    except Exception as exc:
        return str(exc)[:2000]
    return None


def _load_py7zr() -> Any | None:
    """py7zr 是最后兜底，部分 BCJ2 压缩包仍需要 7-Zip CLI。"""
    try:
        return importlib.import_module("py7zr")
    except ImportError:
        return None


def _seven_zip_exe() -> Path | None:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            for name in _SEVEN_ZIP_COMMANDS:
                bundled = Path(meipass) / "7za" / name
                if bundled.is_file():
                    return bundled

    project_root = _project_root()
    for name in _SEVEN_ZIP_COMMANDS:
        bundled = project_root / "build_exe" / name
        if bundled.is_file():
            return bundled

    for name in _SEVEN_ZIP_COMMANDS:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _extract_with_7zip(exe: Path, archive: Path, out_dir: Path) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(out_dir.resolve())
    if not output_dir.endswith(("/", "\\")):
        output_dir += "\\" if sys.platform == "win32" else "/"
    cmd = [str(exe), "x", "-y", f"-o{output_dir}", str(archive)]
    kwargs: dict[str, object] = {
        "args": cmd,
        "capture_output": True,
        "text": True,
        "timeout": 3600,
    }
    if sys.platform == "win32" and _WIN_NO_WINDOW:
        kwargs["creationflags"] = _WIN_NO_WINDOW
    try:
        result = subprocess.run(**kwargs)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)[:2000]
    if result.returncode != 0:
        return (result.stderr or result.stdout or f"exit {result.returncode}")[:2000]
    return None


def _extract_with_py7zr(py7zr: Any, archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "r") as seven_zip:
        seven_zip.extractall(path=out_dir)


def _reset_extract_dir(out_dir: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def _format_extractor_missing_error(py7zz_error: str, cli_error: str | None) -> str:
    if cli_error is None:
        return (
            "未找到 py7zz、7-Zip CLI 或 py7zr，请先安装 py7zz、7-Zip 或 py7zr 后重试。"
            f"py7zz: {py7zz_error}"
        )[:2000]
    return f"py7zz: {py7zz_error}; 7-Zip CLI: {cli_error}; py7zr: missing"[:2000]


def _format_py7zr_failure_error(py7zz_error: str, cli_error: str | None, exc: Exception) -> str:
    cli_part = "missing" if cli_error is None else cli_error
    return (
        "需要 py7zz 或 7-Zip CLI 才能解压此压缩包；"
        f"py7zr 兜底解压失败：{exc}; py7zz: {py7zz_error}; 7-Zip CLI: {cli_part}"
    )[:2000]


def _resolve_extracted_root(extract_to: Path) -> Path:
    if (extract_to / "runtime" / "python.exe").is_file():
        return extract_to.resolve()
    children = [path for path in extract_to.iterdir() if not path.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0].resolve()
    return extract_to.resolve()
