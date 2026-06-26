"""标定项目冒烟测试。

覆盖：
- 模块导入
- 数据结构创建
- 配置文件加载
- 角点检测基本流程
- 完整标定流程（使用项目已有数据）
- YAML 序列化/反序列化
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class TestImports(unittest.TestCase):
    """模块导入测试。"""

    def test_import_calibration(self) -> None:
        from src.calibration import (  # noqa: F811
            CalibrationBoard,
            CalibrationResult,
            StereoCalibration,
            calibrate_stereo,
            create_checkerboard_object_points,
            find_checkerboard_corners,
            load_stereo_calibration,
            pair_stereo_images,
            print_calibration_result,
            save_stereo_calibration,
        )

    def test_import_config(self) -> None:
        from src.config import CalibrationConfig, load_config  # noqa: F811

    def test_import_main(self) -> None:
        from src.main import main  # noqa: F811

    def test_import_utils(self) -> None:
        from src.utils import (  # noqa: F811
            read_images_color_from_folder,
            read_images_from_folder,
        )


class TestDataStructures(unittest.TestCase):
    """数据结构创建与验证测试。"""

    def test_calibration_board_valid(self) -> None:
        from src.calibration import CalibrationBoard

        board = CalibrationBoard(cols=11, rows=8, square_size=6.0)
        self.assertEqual(board.pattern_size, (11, 8))
        self.assertEqual(board.board_size, (8, 11))
        self.assertEqual(board.col_spacing, 6.0)
        self.assertEqual(board.row_spacing, 6.0)

    def test_calibration_board_invalid_cols(self) -> None:
        from src.calibration import CalibrationBoard

        with self.assertRaises(ValueError):
            CalibrationBoard(cols=0, rows=8, square_size=6.0)

    def test_calibration_board_no_spacing(self) -> None:
        from src.calibration import CalibrationBoard

        with self.assertRaises(ValueError):
            CalibrationBoard(cols=11, rows=8)

    def test_stereo_calibration_result_summary(self) -> None:
        from src.calibration import CalibrationResult, StereoCalibration

        k = np.eye(3, dtype=np.float64)
        d = np.zeros((5, 1), dtype=np.float64)
        r = np.eye(3, dtype=np.float64)
        t = np.zeros((3, 1), dtype=np.float64)
        q = np.eye(4, dtype=np.float64)

        calib = StereoCalibration(
            k1=k, d1=d, k2=k, d2=d, r=r, t=t, q=q, image_size=(720, 540),
        )
        result = CalibrationResult(
            calibration=calib,
            stereo_rms=0.05,
            left_rms=0.04,
            right_rms=0.03,
            accepted_pairs=40,
            rejected_pairs=0,
        )
        summary = result.summary()
        self.assertEqual(summary["accepted_pairs"], 40)
        self.assertEqual(summary["image_size"], "720×540")

    def test_create_object_points(self) -> None:
        from src.calibration import CalibrationBoard, create_checkerboard_object_points

        board = CalibrationBoard(cols=11, rows=8, square_size=6.0)
        obj_pts = create_checkerboard_object_points(board)
        self.assertEqual(obj_pts.shape, (88, 3))
        # 最后一个点 (row=7, col=10) = (10*6, 7*6, 0) = (60, 42, 0)
        np.testing.assert_allclose(obj_pts[-1], [60.0, 42.0, 0.0])


class TestConfig(unittest.TestCase):
    """配置文件加载测试。"""

    def setUp(self) -> None:
        self.config_path = CONFIG_PATH

    def test_load_config(self) -> None:
        from src.calibration import CalibrationBoard
        from src.config import load_config

        cfg = load_config(self.config_path)
        self.assertIsInstance(cfg.board, CalibrationBoard)
        self.assertEqual(cfg.board.cols, 11)
        self.assertEqual(cfg.board.rows, 8)
        self.assertEqual(cfg.board.square_size, 6.0)
        self.assertTrue(cfg.use_sb)
        self.assertEqual(cfg.min_pairs, 15)

    def test_config_paths_resolved(self) -> None:
        from src.config import load_config

        cfg = load_config(self.config_path)
        self.assertTrue(cfg.left_dir.exists(), f"left_dir not found: {cfg.left_dir}")
        self.assertTrue(cfg.right_dir.exists(), f"right_dir not found: {cfg.right_dir}")
        # Verify they point to actual image directories
        left_images = list(cfg.left_dir.glob("*"))
        right_images = list(cfg.right_dir.glob("*"))
        self.assertGreater(len(left_images), 0, "No images in left_dir")
        self.assertGreater(len(right_images), 0, "No images in right_dir")


class TestImagePairing(unittest.TestCase):
    """图片配对测试。"""

    def setUp(self) -> None:
        from src.config import load_config

        self.cfg = load_config(CONFIG_PATH)

    def test_pair_stereo_images(self) -> None:
        from src.calibration import pair_stereo_images

        pairs = pair_stereo_images(self.cfg.left_dir, self.cfg.right_dir)
        self.assertGreater(len(pairs), 0)
        left, right = pairs[0]
        self.assertIsInstance(left, Path)
        self.assertIsInstance(right, Path)
        # 左右文件名应匹配
        self.assertEqual(left.name, right.name)

    def test_pair_count_matches_files(self) -> None:
        from src.calibration import pair_stereo_images

        pairs = pair_stereo_images(self.cfg.left_dir, self.cfg.right_dir)
        left_count = len(list(self.cfg.left_dir.glob("*")))
        right_count = len(list(self.cfg.right_dir.glob("*")))
        self.assertEqual(len(pairs), min(left_count, right_count))


class TestCornerDetection(unittest.TestCase):
    """角点检测测试。"""

    def setUp(self) -> None:
        from src.calibration import _list_images
        from src.config import load_config

        cfg = load_config(CONFIG_PATH)
        self.left_images = _list_images(cfg.left_dir)
        self.board = cfg.board

    def test_find_checkerboard_corners_first_image(self) -> None:
        from src.calibration import _read_image, find_checkerboard_corners

        img = _read_image(self.left_images[0])
        corners = find_checkerboard_corners(img, self.board, use_sb=True)
        self.assertIsNotNone(corners)
        self.assertEqual(corners.shape, (88, 1, 2))  # 11*8=88 corners


class TestYamlIO(unittest.TestCase):
    """YAML 序列化/反序列化测试。"""

    def setUp(self) -> None:
        from src.calibration import StereoCalibration

        k = np.eye(3, dtype=np.float64)
        d = np.zeros((5, 1), dtype=np.float64)
        r = np.eye(3, dtype=np.float64)
        t = np.zeros((3, 1), dtype=np.float64)
        q = np.eye(4, dtype=np.float64)
        self.calib = StereoCalibration(
            k1=k, d1=d, k2=k, d2=d, r=r, t=t, q=q, image_size=(720, 540),
        )

    def test_save_and_load_roundtrip(self) -> None:
        import tempfile

        from src.calibration import load_stereo_calibration, save_stereo_calibration

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "calib.yaml"
            saved = save_stereo_calibration(out_path, self.calib)
            loaded = load_stereo_calibration(saved)

            np.testing.assert_allclose(loaded.k1, self.calib.k1)
            np.testing.assert_allclose(loaded.d1, self.calib.d1)
            np.testing.assert_allclose(loaded.k2, self.calib.k2)
            np.testing.assert_allclose(loaded.d2, self.calib.d2)
            np.testing.assert_allclose(loaded.r, self.calib.r)
            np.testing.assert_allclose(loaded.t, self.calib.t)
            np.testing.assert_allclose(loaded.q, self.calib.q)
            self.assertEqual(loaded.image_size, self.calib.image_size)


class TestFullCalibration(unittest.TestCase):
    """完整标定流程测试（慢测试，使用项目已有数据）。"""

    def test_calibrate_stereo_full(self) -> None:
        from src.calibration import calibrate_stereo
        from src.config import load_config

        cfg = load_config(CONFIG_PATH)
        result = calibrate_stereo(
            left_dir=cfg.left_dir,
            right_dir=cfg.right_dir,
            board=cfg.board,
            output_path=None,  # 不写文件，只验证流程
            use_sb=cfg.use_sb,
            min_pairs=cfg.min_pairs,
            subpix_window=cfg.subpix_window,
            fix_intrinsic=cfg.fix_intrinsic,
            stereo_max_iter=cfg.max_iter,
            stereo_epsilon=cfg.epsilon,
        )
        self.assertGreater(result.accepted_pairs, 0)
        self.assertLess(result.stereo_rms, 1.0)  # 标定误差 < 1px
        # 基线 Tx 大约在 100-110mm 之间
        tx = float(result.calibration.t[0, 0])
        self.assertGreater(tx, 50)
        self.assertLess(tx, 200)


if __name__ == "__main__":
    unittest.main()
