"""双目相机标定模块。

对标 C++ 参考项目 D:\\Qt_project\\vision_experiment 的标定流程：
  逐对棋盘格角点检测 → 左右分别单目标定 → 固定内参双目标定 → 输出 YAML

流程细节（参考 calib3d_stereo_camera.cpp::calibrateStereoCamera）：
  1. 左右图片按文件名自然排序配对
  2. 每对图片检测棋盘格内角点，仅当两侧都成功时加入样本
  3. 先用 cv2.calibrateCamera 分别标定左右单目相机
  4. 再用 cv2.stereoCalibrate(flags=CALIB_FIX_INTRINSIC) 求解 R/T
  5. 调用 cv2.stereoRectify 计算 Q 矩阵
  6. 输出 OpenCV FileStorage 兼容的 YAML 文件
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# 数据结构
# ============================================================================

@dataclass(frozen=True)
class CalibrationBoard:
    """棋盘格标定板参数。

    属性:
        cols, rows: 内角点数量（不是格子数），对应 OpenCV patternSize。
        square_size: 方格物理尺寸（毫米），正方形方格专用。
        col_spacing, row_spacing: 非正方形方格时分别设置。
        pattern: 标定板类型（仅支持 checkerboard）。
    """

    cols: int
    rows: int
    square_size: Optional[float] = None
    col_spacing: Optional[float] = None
    row_spacing: Optional[float] = None
    pattern: str = "checkerboard"

    def __post_init__(self) -> None:
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError(f"棋盘格行列数必须为正: {self.cols}×{self.rows}")

        spacing_candidates = [
            self.col_spacing,
            self.row_spacing,
            self.square_size,
        ]
        if all(v is None for v in spacing_candidates):
            raise ValueError("必须设置 square_size 或 col_spacing/row_spacing")

        # 如果设置了 square_size 但未分别设置 col/row_spacing，则自动填充
        if self.square_size is not None:
            object.__setattr__(self, "col_spacing", self.col_spacing or self.square_size)
            object.__setattr__(self, "row_spacing", self.row_spacing or self.square_size)

        if self.col_spacing is not None and self.col_spacing <= 0:
            raise ValueError("col_spacing 必须为正数")
        if self.row_spacing is not None and self.row_spacing <= 0:
            raise ValueError("row_spacing 必须为正数")

        if self.pattern.lower() not in {"checkerboard", "chessboard", "checker"}:
            raise NotImplementedError("目前仅支持棋盘格标定板")

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """OpenCV 角点检测用的棋盘格尺寸 ``(cols, rows)``。"""
        return self.cols, self.rows

    @property
    def board_size(self) -> Tuple[int, int]:
        """返回 ``(rows, cols)`` 兼容 C++ BoardDescribe 的习惯。"""
        return self.rows, self.cols


@dataclass(frozen=True)
class StereoCalibration:
    """双目相机标定结果。

    属性:
        k1, k2: 左、右相机内参矩阵 (3×3)。
        d1, d2: 左、右相机畸变系数 (5×1)。
        r: 左相机到右相机的旋转矩阵 (3×3)。
        t: 左相机到右相机的平移向量 (3×1)。
        q: 视差到三维坐标的重投影矩阵 (4×4)。
        image_size: 图像尺寸 ``(width, height)``。
    """

    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray
    q: np.ndarray
    image_size: Tuple[int, int]

    def rectification_maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """生成双目校正映射表。

        返回:
            (left_x, left_y, right_x, right_y, q):
            左右相机的重映射坐标矩阵和 Q 矩阵。
        """
        width, height = self.image_size
        r1, r2, p1, p2, q_rect, _, _ = cv2.stereoRectify(
            self.k1, self.d1, self.k2, self.d2,
            (width, height), self.r, self.t,
            flags=0, alpha=-1, newImageSize=(width, height),
        )
        left_x, left_y = cv2.initUndistortRectifyMap(
            self.k1, self.d1, r1, p1, (width, height), cv2.CV_32FC1,
        )
        right_x, right_y = cv2.initUndistortRectifyMap(
            self.k2, self.d2, r2, p2, (width, height), cv2.CV_32FC1,
        )
        q = self.q if self.q.size == 16 else q_rect
        return left_x, left_y, right_x, right_y, q


@dataclass(frozen=True)
class CalibrationResult:
    """标定结果及统计信息。"""

    calibration: StereoCalibration
    stereo_rms: float
    left_rms: float
    right_rms: float
    accepted_pairs: int
    rejected_pairs: int
    rejected_reasons: Tuple[str, ...] = field(default_factory=tuple)
    output_path: Optional[Path] = None
    stage_times: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "accepted_pairs": self.accepted_pairs,
            "rejected_pairs": self.rejected_pairs,
            "left_rms": self.left_rms,
            "right_rms": self.right_rms,
            "stereo_rms": self.stereo_rms,
            "image_size": f"{self.calibration.image_size[0]}×{self.calibration.image_size[1]}",
        }


# ============================================================================
# 角点检测
# ============================================================================

def _natural_sort_key(path: Path) -> List:
    """按文件名中的数字自然排序，使 10.bmp 排在 9.bmp 之后而非之前。"""
    parts = re.split(r"(\d+)", path.stem)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part.lower())
    return result


_IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def _list_images(folder: Path) -> List[Path]:
    """列出目录中所有标定图片，按自然排序。"""
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS],
        key=_natural_sort_key,
    )


def _read_image(path: Path) -> np.ndarray:
    """读取标定图片为灰度 uint8。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return _as_gray_uint8(img)


def _as_gray_uint8(img: np.ndarray) -> np.ndarray:
    """转灰度 uint8。"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype == np.uint8:
        return gray
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros(gray.shape, dtype=np.uint8)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray.astype(np.uint8)


def _find4_quad_corner_subpix(gray: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """调用 OpenCV find4QuadCornerSubpix 做亚像素优化。

    find4QuadCornerSubpix 在部分 OpenCV 版本中不可用，
    不可用时返回 None 让调用方回退到 cornerSubPix。
    """
    if not hasattr(cv2, "find4QuadCornerSubpix"):
        return None
    try:
        result = cv2.find4QuadCornerSubpix(gray, corners, (11, 11))
    except cv2.error:
        return None
    if result is None:
        return None
    if isinstance(result, tuple):
        ok, refined = result
        return np.asarray(refined, dtype=np.float32) if ok else None
    return np.asarray(result, dtype=np.float32)


def find_checkerboard_corners(
    gray: np.ndarray,
    board: CalibrationBoard,
    use_sb: bool = False,
    subpix_window: int = 11,
) -> Optional[np.ndarray]:
    """检测棋盘格内角点。

    流程（参考 calib3d_camera.cpp 的 findCheckerboardPixelCorners 系列函数）：
      1. 先用 cv2.checkChessboard 做快速验证
      2. 标准检测 findChessboardCorners
      3. 亚像素优化：优先 find4QuadCornerSubpix，失败回退 cornerSubPix

    参数:
        gray: 灰度 uint8 图像。
        board: 棋盘格参数。
        use_sb: 优先使用 SB 高精度检测。
        subpix_window: 亚像素优化窗口半径。

    返回:
        (N, 1, 2) float32 角点数组；检测失败时返回 None。
    """
    image = _as_gray_uint8(gray)
    pattern_size = board.pattern_size
    subpix_win = (subpix_window, subpix_window)

    # SB 检测（高精度，失败时回退到标准检测）
    if use_sb:
        flags = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_EXHAUSTIVE
        found_sb, corners_sb = cv2.findChessboardCornersSB(image, pattern_size, flags=flags)
        if found_sb:
            return np.asarray(corners_sb, dtype=np.float32)

    # 标准检测：先用 checkChessboard 做快速验证
    if not cv2.checkChessboard(image, pattern_size):
        return None

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, pattern_size, flags=flags)
    if not found:
        return None

    corners = np.asarray(corners, dtype=np.float32)

    # 亚像素优化 — 参考 C++ 项目：优先 find4QuadCornerSubpix
    refined = _find4_quad_corner_subpix(image, corners)
    if refined is not None:
        return refined

    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    return cv2.cornerSubPix(image, corners, subpix_win, (-1, -1), criteria)


def create_checkerboard_object_points(board: CalibrationBoard) -> np.ndarray:
    """生成棋盘格角点在标定板坐标系下的三维物理坐标 (Z=0)。

    参考 C++ createCheckerboardPhysicalPos：按行优先顺序排列。

    返回:
        (rows*cols, 3) float32 数组。
    """
    points = np.zeros((board.rows * board.cols, 3), dtype=np.float32)
    idx = 0
    for r in range(board.rows):
        for c in range(board.cols):
            points[idx, 0] = c * board.col_spacing
            points[idx, 1] = r * board.row_spacing
            idx += 1
    return points


# ============================================================================
# 图片配对
# ============================================================================

def pair_stereo_images(left_dir: Path, right_dir: Path) -> List[Tuple[Path, Path]]:
    """按自然排序配对左右标定图片。

    参考 C++ StereoCameraCalibratorImpl::appendBoardImage 的逐对处理逻辑。

    返回:
        [(left_path, right_path), ...] 一一对应的图片对列表。

    异常:
        ValueError: 目录为空或左右图片数量不一致。
    """
    left_files = _list_images(left_dir)
    right_files = _list_images(right_dir)

    if not left_files:
        raise ValueError(f"左图目录为空: {left_dir}")
    if not right_files:
        raise ValueError(f"右图目录为空: {right_dir}")
    if len(left_files) != len(right_files):
        raise ValueError(
            f"左右图片数量不一致: left={len(left_files)}, right={len(right_files)}"
        )

    return list(zip(left_files, right_files))


# ============================================================================
# 标定主流程
# ============================================================================

def calibrate_stereo(
    left_dir: Path,
    right_dir: Path,
    board: CalibrationBoard,
    output_path: Optional[Path] = None,
    use_sb: bool = False,
    min_pairs: int = 6,
    max_pairs: Optional[int] = None,
    subpix_window: int = 11,
    fix_intrinsic: bool = True,
    stereo_max_iter: int = 30,
    stereo_epsilon: float = 1.0e-6,
) -> CalibrationResult:
    """执行双目相机标定。
    流程对齐 C++ 项目 calib3d_stereo_camera.cpp::calibrateStereoCamera：
    1. 左右图片成对检测棋盘格角点，仅保留两侧都成功的样本
    2. 分别对左右相机做单目标定 → 获取初始内参和畸变系数
    3. 固定内参做双目标定 → 求解 R/T/E/F
    4. 计算 Q 矩阵
    5. 输出 YAML
    参数:
        left_dir: 左相机标定图片目录。
        right_dir: 右相机标定图片目录。
        board: 棋盘格参数。
        output_path: 可选输出 YAML 路径。
        use_sb: 是否优先使用 SB 角点检测。
        min_pairs: 最少有效图片对数量。
        max_pairs: 可选最大图片对数量（调试用）。
        subpix_window: 亚像素优化窗口。
        fix_intrinsic: 双目标定时是否固定内参。
        stereo_max_iter: 双目标定最大迭代次数。
        stereo_epsilon: 双目标定收敛精度。

    返回:
        CalibrationResult。
    """
    # ---- 计时开始 ----
    _t = time.perf_counter
    t0 = _t()

    # ---- 1. 图片配对 ----
    pairs = pair_stereo_images(left_dir, right_dir)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    t_pair = _t()

    object_template = create_checkerboard_object_points(board)
    object_points: List[np.ndarray] = []
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    rejected_reasons: List[str] = []
    image_size: Optional[Tuple[int, int]] = None

    for left_path, right_path in pairs:
        left_img = _read_image(left_path)
        right_img = _read_image(right_path)

        if left_img.shape != right_img.shape:
            rejected_reasons.append(
                f"尺寸不一致: {left_path.name}({left_img.shape}) vs "
                f"{right_path.name}({right_img.shape})"
            )
            continue

        current_size = (left_img.shape[1], left_img.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected_reasons.append(
                f"尺寸与首对不一致: {left_path.name} ({current_size})"
            )
            continue

        # 左图检测
        left_corners = find_checkerboard_corners(
            left_img, board, use_sb=use_sb, subpix_window=subpix_window,
        )
        if left_corners is None:
            rejected_reasons.append(f"左图棋盘格检测失败: {left_path.name}")
            continue

        # 右图检测
        right_corners = find_checkerboard_corners(
            right_img, board, use_sb=use_sb, subpix_window=subpix_window,
        )
        if right_corners is None:
            rejected_reasons.append(f"右图棋盘格检测失败: {right_path.name}")
            continue

        object_points.append(object_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)
    t_corner = _t()

    if image_size is None:
        raise ValueError("未找到可读的标定图片对")

    accepted = len(object_points)
    if accepted < min_pairs:
        raise ValueError(
            f"有效图片对不足: {accepted} < {min_pairs}，请补充更多标定图片或降低 min_pairs"
        )

    # ---- 2. 左右单目标定 ----
    # 参考 C++ calibrateCamera()：先获取独立内参，再用于双目标定
    left_rms, k1, d1, _, _ = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None,
    )
    t_left = _t()
    right_rms, k2, d2, _, _ = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None,
    )
    t_right = _t()

    # ---- 3. 双目标定 ----
    # 参考 C++ calibrateStereoCamera()：先 mono 再 stereo with FIX_INTRINSIC
    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                stereo_max_iter, stereo_epsilon)
    stereo_rms, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        object_points, left_points, right_points,
        k1, d1, k2, d2, image_size,
        criteria=criteria,
        flags=flags,
    )
    t_stereo = _t()

    # ---- 4. 计算 Q 矩阵 ----
    _, _, _, _, q, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, image_size, r, t,
        flags=0, alpha=-1, newImageSize=image_size,
    )

    calibration = StereoCalibration(
        k1=np.asarray(k1, dtype=np.float64),
        d1=np.asarray(d1, dtype=np.float64).reshape(-1, 1),
        k2=np.asarray(k2, dtype=np.float64),
        d2=np.asarray(d2, dtype=np.float64).reshape(-1, 1),
        r=np.asarray(r, dtype=np.float64),
        t=np.asarray(t, dtype=np.float64).reshape(3, 1),
        q=np.asarray(q, dtype=np.float64),
        image_size=image_size,
    )

    saved = None
    if output_path is not None:
        saved = save_stereo_calibration(output_path, calibration)
    t_end = _t()

    stage_times = {
        "图片配对":        round(t_pair - t0, 3),
        "角点检测":        round(t_corner - t_pair, 3),
        "左单目标定":      round(t_left - t_corner, 3),
        "右单目标定":      round(t_right - t_left, 3),
        "双目标定":        round(t_stereo - t_right, 3),
        "Q矩阵&输出":      round(t_end - t_stereo, 3),
        "总用时":          round(t_end - t0, 3),
    }

    return CalibrationResult(
        calibration=calibration,
        stereo_rms=float(stereo_rms),
        left_rms=float(left_rms),
        right_rms=float(right_rms),
        accepted_pairs=accepted,
        rejected_pairs=len(rejected_reasons),
        rejected_reasons=tuple(rejected_reasons),
        output_path=saved,
        stage_times=stage_times,
    )


# ============================================================================
# YAML 序列化 / 反序列化
# ============================================================================

def _format_yaml_number(value: float) -> str:
    """格式化浮点数为 OpenCV FileStorage 兼容格式。

    - 0 → "0."
    - 科学计数法或含小数点的保留原样
    - 整数追加 "." 以保持浮点类型
    """
    if value == 0:
        return "0."
    text = f"{value:.17g}"
    if "e" not in text.lower() and "." not in text:
        text += "."
    return text


def save_stereo_calibration(path: Path, calib: StereoCalibration) -> Path:
    """保存双目标定参数为 OpenCV FileStorage 兼容的 YAML 文件。

    YAML 格式与 D:\\3d_project\\python_stereo_structured_light 项目兼容。
    每个矩阵展开为一维列表逐元素写入，便于跨语言读取。

    参数:
        path: 输出 YAML 路径。
        calib: 标定参数。

    返回:
        实际写出的路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    values = {
        "K1": calib.k1.reshape(-1),
        "D1": calib.d1.reshape(-1),
        "K2": calib.k2.reshape(-1),
        "D2": calib.d2.reshape(-1),
        "R": calib.r.reshape(-1),
        "T": calib.t.reshape(-1),
        "Q": calib.q.reshape(-1),
    }

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("%YAML:1.0\n---\n")
        for key, array in values.items():
            f.write(f"{key}:\n")
            for v in array:
                f.write(f"   - {_format_yaml_number(float(v))}\n")
        f.write(f"width: {calib.image_size[0]}\n")
        f.write(f"height: {calib.image_size[1]}\n")

    return path.resolve()


def load_stereo_calibration(path: Path) -> StereoCalibration:
    """从 YAML 文件加载双目标定参数。

    使用轻量级解析器（不依赖 PyYAML），兼容 OpenCV FileStorage 格式。

    参数:
        path: YAML 文件路径。

    返回:
        StereoCalibration。
    """
    data = _parse_simple_yaml(Path(path))
    w = int(data["width"][0])
    h = int(data["height"][0])
    return StereoCalibration(
        k1=np.asarray(data["K1"], dtype=np.float64).reshape(3, 3),
        d1=np.asarray(data["D1"], dtype=np.float64).reshape(-1, 1),
        k2=np.asarray(data["K2"], dtype=np.float64).reshape(3, 3),
        d2=np.asarray(data["D2"], dtype=np.float64).reshape(-1, 1),
        r=np.asarray(data["R"], dtype=np.float64).reshape(3, 3),
        t=np.asarray(data["T"], dtype=np.float64).reshape(3, 1),
        q=np.asarray(data["Q"], dtype=np.float64).reshape(4, 4),
        image_size=(w, h),
    )


def _parse_simple_yaml(path: Path) -> Dict[str, List[float]]:
    """解析标定 YAML 文件的数值子集。

    支持的格式：
      - %YAML:1.0 头部
      - --- 文档分隔符
      - key: value 标量
      - key: 后跟 - value 列表

    异常:
        ValueError: 缺少必需字段。
    """
    data: Dict[str, List[float]] = {}
    current_key: Optional[str] = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("%") or text == "---":
            continue
        if text.startswith("-"):
            if current_key is not None:
                data[current_key].append(float(text[1:].strip()))
            continue
        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            data[key] = [float(value)] if value else []

    required = {"K1", "D1", "K2", "D2", "R", "T", "Q", "width", "height"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"标定文件缺少必需字段: {', '.join(missing)}")
    return data


# ============================================================================
# 便捷显示
# ============================================================================

def print_calibration_result(result: CalibrationResult, align: int = 18) -> None:
    """打印标定结果摘要到控制台。

    参数:
        result: 标定结果。
        align: 耗时打印时标签右对齐宽度。
    """
    calib = result.calibration
    print("=" * 54)
    print("  双目标定完成")
    print("=" * 54)
    print(f"  图像尺寸:           {calib.image_size[0]} × {calib.image_size[1]}")
    print(f"  有效图片对:         {result.accepted_pairs}")
    print(f"  跳过图片对:         {result.rejected_pairs}")
    print(f"  左目重投影误差:     {result.left_rms:.4f} px")
    print(f"  右目重投影误差:     {result.right_rms:.4f} px")
    print(f"  双目标定误差:       {result.stereo_rms:.4f} px")
    print(f"  基线 Tx:            {float(calib.t[0]):.2f} mm")
    print(f"  光轴夹角:           {_calc_toe_in_angle(calib.r):.2f}°")
    if result.output_path:
        print(f"  输出文件:           {result.output_path}")

    # 各阶段耗时
    if result.stage_times:
        total = result.stage_times.get("总用时", 0)
        print("  " + "-" * 50)
        print(f"  {'阶段耗时':>{align}}")
        print("  " + "-" * 50)
        for stage in ("图片配对", "角点检测", "左单目标定", "右单目标定",
                      "双目标定", "Q矩阵&输出"):
            sec = result.stage_times.get(stage, 0)
            pct = f"({sec / total * 100:.0f}%)" if total > 0 else ""
            print(f"  {stage:>{align}}  {sec:.3f} s  {pct}")
        print("  " + "-" * 50)
        print(f"  {'总用时':>{align}}  {total:.3f} s")

    if result.rejected_reasons:
        print(f"\n  跳过原因（前 10 条）:")
        for reason in result.rejected_reasons[:10]:
            print(f"    - {reason}")


def _calc_toe_in_angle(r: np.ndarray) -> float:
    """计算双目光轴夹角（toe-in angle），用旋转矩阵的 Rodrigues 角表示。"""
    angle_rad = cv2.Rodrigues(r)[0]
    norm = np.linalg.norm(angle_rad)
    return float(np.degrees(norm))
