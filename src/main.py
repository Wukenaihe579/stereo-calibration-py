#!/usr/bin/env python
"""双目相机标定 — 命令行入口。

只通过 YAML 配置文件传入参数，不接收其他 CLI 参数。
用法:
    python -m src.main                 # 使用 configs/default.yaml
    python -m src.main -c configs/xxx.yaml  # 指定配置
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(
        description="双目相机标定 — 仅通过 YAML 配置文件传入参数",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="YAML 配置文件路径，默认 configs/default.yaml",
    )
    parsed = parser.parse_args(argv)

    config_path = _resolve_config_path(parsed.config)
    if config_path is None:
        parser.print_help()
        return 1

    # 延迟导入，让 --help 和路径检查更快
    from .calibration import calibrate_stereo, print_calibration_result
    from .config import load_config

    try:
        cfg = load_config(config_path)
    except (ValueError, RuntimeError) as e:
        print(f"配置加载失败: {e}")
        return 1

    print(f"配置文件: {config_path.resolve()}")
    print(f"左图目录: {cfg.left_dir}")
    print(f"右图目录: {cfg.right_dir}")
    print(f"棋盘格: {cfg.board.cols}×{cfg.board.rows}, "
          f"方格={cfg.board.col_spacing}×{cfg.board.row_spacing} mm")
    print(f"标定输出: {cfg.output}")
    print()

    try:
        result = calibrate_stereo(
            left_dir=cfg.left_dir,
            right_dir=cfg.right_dir,
            board=cfg.board,
            output_path=cfg.output,
            use_sb=cfg.use_sb,
            min_pairs=cfg.min_pairs,
            max_pairs=cfg.max_pairs,
            subpix_window=cfg.subpix_window,
            fix_intrinsic=cfg.fix_intrinsic,
            stereo_max_iter=cfg.max_iter,
            stereo_epsilon=cfg.epsilon,
        )
    except (ValueError, cv2.error) as e:
        print(f"标定失败: {e}")
        return 1

    print_calibration_result(result)
    return 0


def _resolve_config_path(given: str | None) -> Path | None:
    if given:
        path = Path(given)
        if not path.exists():
            print(f"错误: 配置文件不存在: {path.resolve()}")
            return None
        return path

    # 未指定时使用默认配置
    default = _resolve_project_root() / "configs" / "default.yaml"
    if not default.exists():
        print(f"错误: 默认配置文件不存在: {default}")
        return None
    print(f"使用默认配置: {default}\n")
    return default


if __name__ == "__main__":
    raise SystemExit(main())
