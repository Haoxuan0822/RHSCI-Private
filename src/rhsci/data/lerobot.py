"""缓存构建用原始 LeRobot 读取，不依赖模型、训练参数或归一化。"""

from pathlib import Path

RGB_KEY = "observation.images.egocentric"


def load_raw_lerobot_dataset(root):
    """读取本地完整帧表及视频元数据；不取训练窗口、不解码视频、不应用增强。"""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).resolve(strict=True)
    dataset = LeRobotDataset(root.name, root=root, delta_timestamps=None, image_transforms=None)
    if dataset.meta.fps != 50:
        raise ValueError("Raw input cache requires 50Hz LeRobot data")
    return dataset
