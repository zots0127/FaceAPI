"""
人脸识别模块 - 基于FaceNet的人脸特征提取和比对
支持1:1人脸验证和基础的特征向量提取
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
    from torchvision import transforms
except ImportError:
    logger.error("请安装facenet-pytorch: pip install facenet-pytorch")
    raise


class FaceRecognition:
    """人脸识别核心类"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化人脸识别模型

        Args:
            model_path: FaceNet模型路径，如果为None则使用预训练模型
            device: 计算设备，'cuda'或'cpu'，如果为None则自动选择
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        logger.info(f"使用设备: {self.device}")

        # 加载FaceNet模型
        self.model = self._load_model(model_path)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        logger.info("FaceNet模型加载完成")

    def _load_model(self, model_path: Optional[str] = None) -> InceptionResnetV1:
        """加载FaceNet模型"""
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"从本地加载模型: {model_path}")
                model = InceptionResnetV1(pretrained=None).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                logger.info("使用预训练的vggface2模型")
                model = InceptionResnetV1(pretrained='vggface2').to(self.device)

            model.eval()
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> torch.Tensor:
        """
        预处理人脸图像

        Args:
            face_image: 人脸图像 (H, W, C)
            target_size: 目标尺寸 (width, height)

        Returns:
            预处理后的tensor
        """
        # 转换为RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        face_image = cv2.resize(face_image, target_size)

        # 转换为tensor并标准化
        face_tensor = self.transform(face_image)
        return face_tensor

    def extract_feature(self, face_image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        提取人脸特征向量

        Args:
            face_image: 人脸图像，可以是numpy数组或tensor

        Returns:
            128维特征向量
        """
        try:
            # 预处理
            if isinstance(face_image, np.ndarray):
                face_tensor = self.preprocess_face(face_image)
            else:
                face_tensor = face_image

            # 添加batch维度
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                feature = self.model(face_tensor)

            # 转换为numpy数组并归一化
            feature_vector = feature.cpu().numpy()[0]
            feature_vector = feature_vector / np.linalg.norm(feature_vector)

            return feature_vector

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise

    def extract_features_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量提取人脸特征

        Args:
            face_images: 人脸图像列表

        Returns:
            特征向量列表
        """
        features = []

        # 批量预处理
        face_tensors = []
        for face_img in face_images:
            if isinstance(face_img, np.ndarray):
                face_tensor = self.preprocess_face(face_img)
            else:
                face_tensor = face_img
            face_tensors.append(face_tensor)

        # 批量推理
        if face_tensors:
            batch_tensor = torch.stack(face_tensors).to(self.device)

            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            # 处理结果
            for feature in batch_features:
                feature_vector = feature.cpu().numpy()
                feature_vector = feature_vector / np.linalg.norm(feature_vector)
                features.append(feature_vector)

        return features

    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度

        Args:
            feature1: 特征向量1
            feature2: 特征向量2

        Returns:
            余弦相似度 (0-1)
        """
        try:
            # 确保向量已归一化
            feature1 = feature1 / np.linalg.norm(feature1)
            feature2 = feature2 / np.linalg.norm(feature2)

            # 计算余弦相似度
            similarity = np.dot(feature1, feature2)
            return float(similarity)
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0

    @staticmethod
    def euclidean_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算两个特征向量的欧氏距离

        Args:
            feature1: 特征向量1
            feature2: 特征向量2

        Returns:
            欧氏距离
        """
        try:
            return np.linalg.norm(feature1 - feature2)
        except Exception as e:
            logger.error(f"距离计算失败: {e}")
            return float('inf')

    def compare_faces(self, feature1: np.ndarray, feature2: np.ndarray,
                     method: str = 'cosine', threshold: float = 0.6) -> dict:
        """
        比较两个人脸特征

        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 比较方法，'cosine'或'euclidean'
            threshold: 相似度阈值

        Returns:
            比较结果字典
        """
        try:
            if method == 'cosine':
                similarity = self.cosine_similarity(feature1, feature2)
                is_same = similarity >= threshold
                distance = 1 - similarity
            elif method == 'euclidean':
                distance = self.euclidean_distance(feature1, feature2)
                # 对于欧氏距离，通常阈值设为1.0左右
                is_same = distance <= threshold
                similarity = max(0, 1 - distance / 2.0)  # 转换为相似度
            else:
                raise ValueError(f"不支持的比较方法: {method}")

            return {
                'is_same_person': is_same,
                'similarity': similarity,
                'distance': distance,
                'method': method,
                'threshold': threshold
            }

        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            return {
                'is_same_person': False,
                'similarity': 0.0,
                'distance': float('inf'),
                'method': method,
                'threshold': threshold,
                'error': str(e)
            }

    def is_same_person(self, feature1: np.ndarray, feature2: np.ndarray,
                      threshold: float = 0.6) -> bool:
        """
        判断是否为同一个人

        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            threshold: 相似度阈值

        Returns:
            是否为同一个人
        """
        result = self.compare_faces(feature1, feature2, threshold=threshold)
        return result['is_same_person']


def extract_face_feature(face_image: np.ndarray, model_path: Optional[str] = None) -> np.ndarray:
    """
    便捷函数：提取单张人脸特征

    Args:
        face_image: 人脸图像
        model_path: 模型路径

    Returns:
        特征向量
    """
    recognizer = FaceRecognition(model_path)
    return recognizer.extract_feature(face_image)


def compare_faces_1v1(face1_image: np.ndarray, face2_image: np.ndarray,
                     threshold: float = 0.6) -> dict:
    """
    便捷函数：1:1人脸验证

    Args:
        face1_image: 第一个人脸图像
        face2_image: 第二个人脸图像
        threshold: 相似度阈值

    Returns:
        比较结果
    """
    recognizer = FaceRecognition()

    # 提取特征
    feature1 = recognizer.extract_feature(face1_image)
    feature2 = recognizer.extract_feature(face2_image)

    # 比较人脸
    return recognizer.compare_faces(feature1, feature2, threshold=threshold)


if __name__ == "__main__":
    # 简单测试
    print("人脸识别模块测试")

    # 创建测试图像 (160x160x3 的随机图像)
    test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

    try:
        # 初始化识别器
        recognizer = FaceRecognition()

        # 提取特征
        feature = recognizer.extract_feature(test_face)
        print(f"特征向量形状: {feature.shape}")
        print(f"特征向量范数: {np.linalg.norm(feature):.6f}")

        # 自我比较测试
        similarity = recognizer.cosine_similarity(feature, feature)
        print(f"自相似度: {similarity:.6f}")

        print("✅ 模块测试通过")

    except Exception as e:
        print(f"❌ 模块测试失败: {e}")