基于 Python 的双目相机标定工具 —— 通过 YAML 配置文件驱动，完成从角点检测到双目立体标定的完整流程，输出可直接被 OpenCV 及下游三维视觉项目加载的标定结果。

YAML 配置驱动：所有标定参数集中在 configs/default.yaml 中，无需修改代码，改配置即可重跑标定。

分阶段标定策略：流程为“左右单目标定 → 固定内参 → 双目联合标定”，避免直接联合优化时内参抖动，标定结果更稳定。

支持多种标定板：checkerboard（棋盘格）、circle_symmetric（对称圆点）、circle_asymmetric（非对称圆点）三种图案均可使用。

SB 高精度检测（可选） ：优先使用 OpenCV 的 Sharpness & Brightness 检测器，精度更高；遇到模糊或低对比度图像失败时，自动回退到标准检测，兼顾精度与鲁棒性。

灵活的内参控制：可通过 fix_intrinsic 开关选择“固定单目标定内参、仅求解 R/T”，或让双目优化同时微调内参。

输出格式友好：标定结果保存为 YAML，兼容 OpenCV FileStorage，可被 python_stereo_structured_light 等下游项目直接通过 load_stereo_calibration() 读取。

附带冒烟测试：tests/test_smoke.py 提供基础的冒烟测试，快速验证环境与流程是否就绪。
