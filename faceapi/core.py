"""
FaceAPI 核心模块
包含人脸检测的核心类和功能
"""

import cv2
import numpy as np
import logging
import os
import time
import math
from typing import List, Dict, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

# MediaPipe导入
try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe未安装，相关功能将不可用")

# Ultralytics YOLO导入
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO未安装，相关功能将不可用")


class SmartCropper:
    """智能重叠裁剪器"""

    def __init__(self, crop_size: int = 640, overlap_ratio: float = 0.2):
        """
        初始化智能裁剪器

        Args:
            crop_size: 裁剪尺寸 (正方形)
            overlap_ratio: 重叠比例
        """
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.overlap_pixels = int(crop_size * overlap_ratio)

    def should_crop(self, image_shape: tuple, threshold: int = 800) -> bool:
        """判断是否需要裁剪"""
        h, w = image_shape[:2]
        return max(h, w) > threshold

    def generate_crops(self, image: np.ndarray) -> List[tuple]:
        """
        生成重叠裁剪区域

        Args:
            image: 输入图像

        Returns:
            裁剪区域列表 [(x1, y1, x2, y2), ...]
        """
        h, w = image.shape[:2]
        crops = []

        # 计算步长
        step = self.crop_size - self.overlap_pixels

        # 生成裁剪坐标
        for y in range(0, h, step):
            for x in range(0, w, step):
                x1 = x
                y1 = y
                x2 = min(x + self.crop_size, w)
                y2 = min(y + self.crop_size, h)

                # 确保裁剪区域有效
                if x2 - x1 > 100 and y2 - y1 > 100:
                    crops.append((x1, y1, x2, y2))

        return crops

    def crop_image(self, image: np.ndarray, crop_coords: tuple) -> np.ndarray:
        """裁剪图像"""
        x1, y1, x2, y2 = crop_coords
        return image[y1:y2, x1:x2]

    def map_coordinates_to_original(self, coords: List[float], crop_coords: tuple) -> List[float]:
        """将裁剪区域坐标映射回原始图像坐标"""
        x1, y1, _, _ = crop_coords
        return [coord + x1 if i % 2 == 0 else coord + y1 for i, coord in enumerate(coords)]

    def remove_overlaps(self, faces: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """使用IoU去除重叠的检测框"""
        if len(faces) <= 1:
            return faces

        # 按置信度排序
        faces.sort(key=lambda x: x['confidence'], reverse=True)

        keep = []
        for i, face in enumerate(faces):
            keep_face = True

            for j in range(i):
                if self.calculate_iou(face['bbox'], faces[j]['bbox']) > iou_threshold:
                    keep_face = False
                    break

            if keep_face:
                keep.append(face)

        return keep

    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class MediaPipeFaceDetector:
    """MediaPipe人脸检测器"""

    def __init__(self):
        """初始化MediaPipe检测器"""
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe不可用")
            self.face_detection = None
            self.face_mesh = None
            return

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # 初始化人脸检测
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # 初始化人脸网格（用于关键点检测）
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=10,
            refine_landmarks=True, min_detection_confidence=0.5
        )

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测人脸"""
        if self.face_detection is None:
            return []

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                face_info = {
                    "id": i,
                    "bbox": [x, y, width, height],
                    "confidence": detection.score[0],
                    "keypoints": []
                }
                faces.append(face_info)

        return faces

    def get_landmarks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """获取人脸关键点"""
        if self.face_mesh is None:
            return []

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        landmarks_list = []
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                h, w = image.shape[:2]
                landmarks = []

                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])

                landmarks_list.append({
                    "id": i,
                    "landmarks": landmarks,
                    "count": len(landmarks)
                })

        return landmarks_list

    def draw_faces(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """在图像上绘制人脸检测结果"""
        result_image = image.copy()

        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']

            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制置信度
            cv2.putText(result_image, f"{confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result_image


class MultiYOLODetector:
    """多YOLO模型检测器 - 支持智能重叠裁剪"""

    def __init__(self, models_dir: str = "models"):
        """
        初始化多YOLO检测器

        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = models_dir
        self.models = {}
        self.available_models = []
        self.smart_cropper = SmartCropper()
        self._load_models()

    def _load_models(self):
        """加载所有可用的YOLO模型"""
        if not YOLO_AVAILABLE:
            logger.warning("Ultralytics YOLO不可用，跳过模型加载")
            return

        model_files = {
            "face11sss": "face11sss.pt",
            "face11n": "face11n.pt",
            "face11s": "face11s.pt",
            "face11m": "face11m.pt",
            "face11l": "face11l.pt",
            "face11x": "face11x.pt"
        }

        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                try:
                    logger.info(f"加载YOLO模型: {model_name} from {model_path}")
                    self.models[model_name] = YOLO(model_path)
                    self.available_models.append(model_name)
                    logger.info(f"✅ 成功加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")

        logger.info(f"总共加载了 {len(self.available_models)} 个YOLO模型")

    def detect_faces(self, image: np.ndarray, model_name: str = "face11n",
                    conf_threshold: float = 0.5, enable_smart_crop: bool = True) -> List[Dict[str, Any]]:
        """
        使用指定YOLO模型检测人脸

        Args:
            image: 输入图像
            model_name: 模型名称
            conf_threshold: 置信度阈值
            enable_smart_crop: 是否启用智能裁剪

        Returns:
            检测到的人脸列表
        """
        if model_name not in self.models:
            logger.error(f"模型 {model_name} 不可用")
            return []

        model = self.models[model_name]
        faces = []

        try:
            if enable_smart_crop and self.smart_cropper.should_crop(image.shape):
                # 使用智能裁剪
                logger.info("启用智能重叠裁剪")
                crop_coords = self.smart_cropper.generate_crops(image)
                logger.info(f"生成了 {len(crop_coords)} 个裁剪区域")

                for i, crop_coord in enumerate(crop_coords):
                    cropped_image = self.smart_cropper.crop_image(image, crop_coord)
                    results = model(cropped_image, verbose=False, conf=conf_threshold)

                    if results and len(results) > 0 and results[0].boxes is not None:
                        for result in results[0].boxes:
                            # 获取边界框坐标
                            xyxy = result.xyxy[0].cpu().numpy()
                            conf = float(result.conf[0].cpu().numpy())

                            # 映射回原始图像坐标
                            original_coords = self.smart_cropper.map_coordinates_to_original(
                                xyxy.tolist(), crop_coord
                            )

                            face_info = {
                                "id": len(faces),
                                "bbox": [
                                    int(original_coords[0]), int(original_coords[1]),
                                    int(original_coords[2] - original_coords[0]),
                                    int(original_coords[3] - original_coords[1])
                                ],
                                "confidence": conf,
                                "keypoints": [],  # YOLO人脸检测通常不包含关键点
                                "model_used": model.ckpt.get("name", "yolo")
                            }
                            faces.append(face_info)

                # 去除重叠检测框
                faces = self.smart_cropper.remove_overlaps(faces)
            else:
                # 直接检测
                results = model(image, verbose=False, conf=conf_threshold)
                if results and len(results) > 0 and results[0].boxes is not None:
                    h, w = image.shape[:2]
                    for result in results[0].boxes:
                        xyxy = result.xyxy[0].cpu().numpy()
                        conf = float(result.conf[0].cpu().numpy())

                        face_info = {
                            "id": len(faces),
                            "bbox": [
                                int(xyxy[0]), int(xyxy[1]),
                                int(xyxy[2] - xyxy[0]),
                                int(xyxy[3] - xyxy[1])
                            ],
                            "confidence": conf,
                            "keypoints": []
                        }
                        faces.append(face_info)

            return faces

        except Exception as e:
            logger.error(f"YOLO检测失败: {str(e)}")
            return []

    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型信息"""
        return {
            "available_models": self.available_models,
            "total_models": len(self.available_models),
            "smart_crop_enabled": True
        }


# 便捷函数
def detect_faces(image: np.ndarray, model: str = "mediapipe", conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """便捷的人脸检测函数"""
    if model.lower() == "mediapipe":
        detector = MediaPipeFaceDetector()
        return detector.detect_faces(image)
    else:
        detector = MultiYOLODetector()
        return detector.detect_faces(image, model_name=model, conf_threshold=conf_threshold)


def get_landmarks(image: np.ndarray) -> List[Dict[str, Any]]:
    """便捷的关键点检测函数"""
    detector = MediaPipeFaceDetector()
    return detector.get_landmarks(image)


def extract_face(image: np.ndarray, bbox: List[int], margin: int = 20) -> np.ndarray:
    """便捷的人脸提取函数"""
    x, y, w, h = bbox

    # 添加边距
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = w + 2 * margin
    h = h + 2 * margin

    # 确保不超出图像边界
    img_h, img_w = image.shape[:2]
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return image[y:y+h, x:x+w]