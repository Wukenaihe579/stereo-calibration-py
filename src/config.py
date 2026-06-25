"""配置文件加载模块。

从 YAML 配置文件读取标定参数，校验合法性后返回给 main.py 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from .calibration import CalibrationBoard


@dataclass
class CalibrationConfig:
    """从 YAML 解析后的完整标定配置。"""

    board: CalibrationBoard
    use_sb: bool = False
    min_pairs: int = 6
    max_pairs: int | None = None
    subpix_window: int = 11
    fix_intrinsic: bool = True
    max_iter: int = 30
    epsilon: float = 1.0e-6
    image_extensions: list[str] = field(default_factory=lambda: [
        ".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ])
    left_dir: Path = Path("data/left")
    right_dir: Path = Path("data/right")
    output: Path = Path("data/stereo_calib_data.yaml")
    save_rectification_maps: bool = False


def load_config(config_path: Path) -> CalibrationConfig:
    """加载并校验 YAML 配置文件。

    参数:
        config_path: YAML 配置文件路径。

    返回:
        解析后的 CalibrationConfig。
    """
    raw = _read_yaml(config_path)
    base_dir = config_path.resolve().parent

    # ---- 解析 board ----
    board_raw = _get_map(raw, "board")
    board = CalibrationBoard(
        cols=int(_required(board_raw, "cols")),
        rows=int(_required(board_raw, "rows")),
        square_size=_optional_float(board_raw, "square_size"),
        col_spacing=_optional_float(board_raw, "col_spacing"),
        row_spacing=_optional_float(board_raw, "row_spacing"),
        pattern=str(board_raw.get("pattern", "checkerboard")),
    )

    # ---- 解析 detection ----
    det_raw = _get_map(raw, "detection")
    use_sb = bool(det_raw.get("use_sb", False))
    min_pairs = int(det_raw.get("min_pairs", 6))
    max_pairs = _optional_int(det_raw, "max_pairs")
    subpix_window = int(det_raw.get("subpix_window", 11))

    # ---- 解析 stereo ----
    st_raw = _get_map(raw, "stereo")
    fix_intrinsic = bool(st_raw.get("fix_intrinsic", True))
    max_iter = int(st_raw.get("max_iter", 30))
    epsilon = float(st_raw.get("epsilon", 1.0e-6))

    # ---- 解析 image ----
    img_raw = _get_map(raw, "image")
    extensions = list(img_raw.get("extensions", [
        ".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ]))

    # ---- 解析 paths ----
    paths_raw = _get_map(raw, "paths")
    left_dir = _resolve_path(paths_raw, "left_dir", base_dir)
    right_dir = _resolve_path(paths_raw, "right_dir", base_dir)
    output = _resolve_output(paths_raw, "output", base_dir)

    # ---- 解析 output ----
    out_raw = _get_map(raw, "output")
    save_rect_maps = bool(out_raw.get("save_rectification_maps", False))

    # 把图片扩展名注入到 calibration 模块的全局列表中
    import sys
    calib_module = sys.modules.get("src.calibration")
    if calib_module is not None:
        calib_module._IMAGE_EXTENSIONS = set(extensions)

    return CalibrationConfig(
        board=board,
        use_sb=use_sb,
        min_pairs=min_pairs,
        max_pairs=max_pairs,
        subpix_window=subpix_window,
        fix_intrinsic=fix_intrinsic,
        max_iter=max_iter,
        epsilon=epsilon,
        image_extensions=extensions,
        left_dir=left_dir,
        right_dir=right_dir,
        output=output,
        save_rectification_maps=save_rect_maps,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    """用 PyYAML 读取配置文件。"""
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "读取 YAML 配置需要 PyYAML。安装: pip install pyyaml"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"配置文件必须包含 YAML mapping: {path}")
    return dict(data)


def _get_map(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """安全获取嵌套 mapping，缺失时返回空字典。"""
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"配置项 {key} 应当是一个 mapping")
    return value


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    """校验必填字段。"""
    value = mapping.get(name)
    if value is None:
        raise ValueError(f"缺少必填配置项: {name}")
    return value


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    """读取可选浮点字段。"""
    v = mapping.get(key)
    return float(v) if v is not None else None


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    """读取可选整数字段。"""
    v = mapping.get(key)
    return int(v) if v is not None else None


def _resolve_path(
    mapping: Mapping[str, Any], key: str, base_dir: Path,
) -> Path:
    """解析路径字段，相对路径基于 base_dir。"""
    raw = mapping.get(key)
    if raw is None:
        raise ValueError(f"缺少必填路径配置: paths.{key}")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_output(
    mapping: Mapping[str, Any], key: str, base_dir: Path,
) -> Path:
    """解析输出路径，自动创建父目录。"""
    raw = mapping.get(key)
    if raw is None:
        raise ValueError(f"缺少必填路径配置: paths.{key}")
    path = Path(str(raw))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
