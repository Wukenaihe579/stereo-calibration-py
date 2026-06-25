"""通用图片读取工具函数。

提供从文件夹批量读取灰度/彩色图片的便捷函数。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

_DEFAULT_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def _normalize_extensions(
    extensions: Sequence[str] | None,
) -> list[str]:
    """确保扩展名为小写。"""
    if extensions is None:
        return list(_DEFAULT_EXTENSIONS)
    return [ext.lower() for ext in extensions]


def read_images_from_folder(
    folder_path: str | Path,
    extensions: Sequence[str] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """读取文件夹内的所有灰度图片。

    Args:
        folder_path: 文件夹路径。
        extensions: 图片格式列表，如 ['.jpg', '.png', '.bmp']。
                    为 None 时默认读取常见格式。

    Returns:
        (图片数组列表, 图片文件名列表) 的元组。
        图片数组为灰度图像。
    """
    exts = _normalize_extensions(extensions)
    folder = Path(folder_path)

    if not folder.exists():
        print(f"警告: 文件夹 '{folder}' 不存在")
        return [], []

    images: list[np.ndarray] = []
    filenames: list[str] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(file_path.name)
            print(f"已加载: {file_path.name}, 尺寸: {img.shape}")
        else:
            print(f"警告: 无法读取 '{file_path.name}'")

    print(f"\n总共加载了 {len(images)} 张图片")
    return images, filenames


def read_images_color_from_folder(
    folder_path: str | Path,
    extensions: Sequence[str] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """读取文件夹内的所有彩色图片。

    Args:
        folder_path: 文件夹路径。
        extensions: 图片格式列表，为 None 时默认读取常见格式。

    Returns:
        (彩色图片数组列表, 图片文件名列表) 的元组。
        图片数组为 BGR 格式（OpenCV 标准格式）。
    """
    exts = _normalize_extensions(extensions)
    folder = Path(folder_path)

    if not folder.exists():
        print(f"警告: 文件夹 '{folder}' 不存在")
        return [], []

    images: list[np.ndarray] = []
    filenames: list[str] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(file_path))
        if img is not None:
            images.append(img)
            filenames.append(file_path.name)
            print(f"已加载: {file_path.name}, 尺寸: {img.shape}")
        else:
            print(f"警告: 无法读取 '{file_path.name}'")

    print(f"\n总共加载了 {len(images)} 张图片")
    return images, filenames
