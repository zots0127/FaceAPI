from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import base64
import logging
import os
from typing import List, Optional, Dict, Any
import uvicorn
import time
import math

# MediaPipe imports
try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Ultralytics YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartCropper:
    """智能重叠裁剪器 - 用于大图像的人脸检测优化"""

    def __init__(self, crop_size: int = 640, overlap_ratio: float = 0.2):
        """
        初始化智能裁剪器

        Args:
            crop_size: 裁剪尺寸 (默认640x640)
            overlap_ratio: 重叠比例 (默认20%)
        """
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(crop_size * (1 - overlap_ratio))

    def get_crops(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        获取智能裁剪区域

        Args:
            image: 输入图像

        Returns:
            裁剪区域列表
        """
        height, width = image.shape[:2]

        # 如果图像小于等于裁剪尺寸，直接返回原图
        if width <= self.crop_size and height <= self.crop_size:
            return [{"x": 0, "y": 0, "w": width, "h": height, "scale_x": 1.0, "scale_y": 1.0}]

        crops = []

        # 计算裁剪区域
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # 计算实际裁剪区域
                crop_x = max(0, x - int(self.stride * self.overlap_ratio))
                crop_y = max(0, y - int(self.stride * self.overlap_ratio))

                crop_w = min(self.crop_size, width - crop_x)
                crop_h = min(self.crop_size, height - crop_y)

                # 如果是边缘区域，调整到完整裁剪尺寸
                if crop_x + crop_w < self.crop_size and crop_x > 0:
                    crop_x = max(0, width - self.crop_size)
                    crop_w = min(self.crop_size, width - crop_x)

                if crop_y + crop_h < self.crop_size and crop_y > 0:
                    crop_y = max(0, height - self.crop_size)
                    crop_h = min(self.crop_size, height - crop_y)

                crops.append({
                    "x": crop_x,
                    "y": crop_y,
                    "w": crop_w,
                    "h": crop_h,
                    "scale_x": width / crop_w,
                    "scale_y": height / crop_h
                })

        # 去重相邻的相同区域
        unique_crops = []
        for crop in crops:
            is_duplicate = False
            for existing in unique_crops:
                if (abs(crop["x"] - existing["x"]) < 10 and
                    abs(crop["y"] - existing["y"]) < 10):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_crops.append(crop)

        return unique_crops


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
            logger.error(f"模型不可用: {model_name}")
            return []

        model = self.models[model_name]
        height, width = image.shape[:2]

        # 判断是否需要智能裁剪
        use_smart_crop = enable_smart_crop and (width > 800 or height > 800)

        if use_smart_crop:
            return self._detect_with_smart_crop(image, model, conf_threshold)
        else:
            return self._detect_single_image(image, model, conf_threshold, scale_x=1.0, scale_y=1.0)

    def _detect_with_smart_crop(self, image: np.ndarray, model, conf_threshold: float) -> List[Dict[str, Any]]:
        """使用智能重叠裁剪进行检测"""
        crops = self.smart_cropper.get_crops(image)
        all_faces = []

        logger.info(f"使用智能裁剪，原图尺寸: {image.shape[:2]}, 裁剪区域数: {len(crops)}")

        for i, crop_info in enumerate(crops):
            # 提取裁剪区域
            crop_img = image[
                crop_info["y"]:crop_info["y"] + crop_info["h"],
                crop_info["x"]:crop_info["x"] + crop_info["w"]
            ]

            # 检测人脸
            faces = self._detect_single_image(
                crop_img, model, conf_threshold,
                crop_info["scale_x"], crop_info["scale_y"],
                offset_x=crop_info["x"], offset_y=crop_info["y"]
            )

            all_faces.extend(faces)

        # 使用IoU去重
        return self._remove_overlaps(all_faces)

    def _detect_single_image(self, image: np.ndarray, model, conf_threshold: float,
                           scale_x: float = 1.0, scale_y: float = 1.0,
                           offset_x: int = 0, offset_y: int = 0) -> List[Dict[str, Any]]:
        """单张图像检测"""
        try:
            results = model(image, conf=conf_threshold, verbose=False)
            faces = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())

                        # 转换为原始图像坐标
                        x1 = int((xyxy[0] * scale_x) + offset_x)
                        y1 = int((xyxy[1] * scale_y) + offset_y)
                        x2 = int((xyxy[2] * scale_x) + offset_x)
                        y2 = int((xyxy[3] * scale_y) + offset_y)

                        # 确保坐标在图像范围内
                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        face_info = {
                            "id": len(faces),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "confidence": conf,
                            "keypoints": [],  # YOLO人脸检测通常不包含关键点
                            "model_used": model.ckpt.get("name", "yolo")
                        }
                        faces.append(face_info)

            return faces

        except Exception as e:
            logger.error(f"YOLO检测失败: {str(e)}")
            return []

    def _remove_overlaps(self, faces: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """使用IoU去除重叠的检测框"""
        if len(faces) <= 1:
            return faces

        # 按置信度排序
        faces.sort(key=lambda x: x["confidence"], reverse=True)

        keep_faces = []
        for face in faces:
            is_duplicate = False
            for kept_face in keep_faces:
                iou = self._calculate_iou(face["bbox"], kept_face["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_faces.append(face)

        return keep_faces

    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2

        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_model_info(self, model_name: str = "face11n") -> Dict[str, Any]:
        """获取模型信息"""
        if model_name not in self.models:
            return {"error": f"模型不可用: {model_name}"}

        return {
            "name": model_name,
            "available_models": self.available_models,
            "smart_crop_enabled": True,
            "models_loaded": len(self.available_models)
        }


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
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box

                h, w = image.shape[:2]
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


# 创建检测器实例
yolo_detector = MultiYOLODetector()
mediapipe_detector = MediaPipeFaceDetector()
face_detector = mediapipe_detector  # 使用MediaPipe检测器作为通用人脸检测器

# 创建 FastAPI 应用
app = FastAPI(
    title="人脸识别 API",
    description="基于 MediaPipe + YOLO 的智能人脸检测API，支持多模型和智能重叠裁剪",
    version="2.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 检测器已在上面初始化

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "人脸识别 API",
        "version": "1.0.0",
        "endpoints": {
            "detect_faces": "/detect_faces",
            "detect_faces_yolo": "/detect_faces_yolo",
            "detect_faces_multi_yolo": "/detect_faces_multi_yolo",
            "benchmark_yolo_models": "/benchmark_yolo_models",
            "face_landmarks": "/face_landmarks",
            "extract_face": "/extract_face",
            "detect_and_draw": "/detect_and_draw",
            "models": "/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "face-api",
        "version": "1.0.0"
    }

@app.get("/models")
async def get_models():
    """获取可用模型信息"""
    models_info = {
        "mediapipe": {
            "status": "available" if MEDIAPIPE_AVAILABLE else "unavailable",
            "type": "MediaPipe Face Detection"
        }
    }

    # 添加 YOLO 多模型信息
    if yolo_detector is not None and len(yolo_detector.available_models) > 0:
        models_info["yolo"] = {
            "status": "available",
            "type": "Multi YOLO Face Detection",
            "available_models": yolo_detector.available_models,
            "smart_crop_enabled": True,
            "default_model": "face11n"
        }
    else:
        models_info["yolo"] = {
            "status": "unavailable",
            "error": "YOLO models not loaded"
        }

    return {
        "available_models": models_info,
        "default_model": "mediapipe"
    }

@app.post("/detect_faces")
async def detect_faces(
    file: UploadFile = File(...),
    model: str = Query("mediapipe", description="检测模型: mediapipe 或 yolo"),
    conf_threshold: float = Query(0.5, description="置信度阈值")
):
    """
    检测图像中的人脸

    Args:
        file: 上传的图像文件

    Returns:
        检测到的人脸信息
    """
    try:
        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")

        # 支持的图像格式和文件扩展名
        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        # 检查 MIME 类型或文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 选择检测模型
        logger.info(f"使用模型: {model}, 置信度阈值: {conf_threshold}")

        if model.lower() == "yolo":
            if yolo_detector is None:
                raise HTTPException(status_code=503, detail="YOLO 模型不可用")
            faces = yolo_detector.detect_faces(image, conf_threshold)
        elif model.lower() == "mediapipe":
            faces = face_detector.detect_faces(image)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {model}。支持的模型: mediapipe, yolo"
            )

        # 转换结果为可序列化格式
        result = {
            "success": True,
            "face_count": len(faces),
            "model_used": model,
            "confidence_threshold": conf_threshold,
            "faces": []
        }

        for i, face in enumerate(faces):
            face_info = {
                "id": i,
                "bbox": face["bbox"],
                "confidence": round(face["confidence"], 3),
                "keypoints": [[round(x, 3), round(y, 3)] for x, y in face["keypoints"]]
            }
            result["faces"].append(face_info)

        logger.info(f"检测到 {len(faces)} 个人脸")
        return result

    except Exception as e:
        logger.error(f"人脸检测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"人脸检测失败: {str(e)}")

@app.post("/face_landmarks")
async def get_face_landmarks(file: UploadFile = File(...)):
    """
    获取人脸关键点

    Args:
        file: 上传的图像文件

    Returns:
        人脸关键点坐标
    """
    try:
        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")

        # 支持的图像格式和文件扩展名
        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        # 检查 MIME 类型或文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 获取人脸关键点
        landmarks = face_detector.get_face_landmarks(image)

        if landmarks is None:
            return {
                "success": True,
                "message": "未检测到人脸",
                "landmarks": []
            }

        result = {
            "success": True,
            "landmarks": landmarks,
            "count": len(landmarks)
        }

        logger.info(f"获取到 {len(landmarks)} 个人脸关键点")
        return result

    except Exception as e:
        logger.error(f"获取人脸关键点错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取人脸关键点失败: {str(e)}")

@app.post("/extract_face")
async def extract_face(file: UploadFile = File(...), face_id: int = 0, margin: int = 20):
    """
    提取指定的人脸区域

    Args:
        file: 上传的图像文件
        face_id: 要提取的人脸ID
        margin: 边界框扩展边距

    Returns:
        提取的人脸图像
    """
    try:
        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")

        # 支持的图像格式和文件扩展名
        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        # 检查 MIME 类型或文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 检测人脸
        faces = face_detector.detect_faces(image)

        if not faces:
            raise HTTPException(status_code=404, detail="未检测到人脸")

        if face_id >= len(faces):
            raise HTTPException(status_code=400, detail=f"人脸ID {face_id} 超出范围")

        # 提取人脸区域
        face_image = face_detector.extract_face(image, faces[face_id]["bbox"], margin)

        if face_image is None:
            raise HTTPException(status_code=500, detail="提取人脸失败")

        # 将图像编码为 JPEG 格式
        _, buffer = cv2.imencode('.jpg', face_image)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(
            io.BytesIO(buffer),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=face_{face_id}.jpg"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提取人脸错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提取人脸失败: {str(e)}")

@app.post("/detect_and_draw")
async def detect_and_draw(file: UploadFile = File(...)):
    """
    检测人脸并在图像上绘制结果

    Args:
        file: 上传的图像文件

    Returns:
        绘制检测结果后的图像
    """
    try:
        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")

        # 支持的图像格式和文件扩展名
        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        # 检查 MIME 类型或文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 检测人脸
        faces = face_detector.detect_faces(image)

        # 在图像上绘制检测结果
        result_image = face_detector.draw_faces(image, faces)

        # 将图像编码为 JPEG 格式
        _, buffer = cv2.imencode('.jpg', result_image)

        return StreamingResponse(
            io.BytesIO(buffer),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=detection_result.jpg"}
        )

    except Exception as e:
        logger.error(f"检测并绘制人脸错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/detect_faces_yolo")
async def detect_faces_yolo(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, description="置信度阈值"),
    detailed: bool = Query(False, description="是否返回详细信息")
):
    """
    使用 YOLO 模型检测人脸

    Args:
        file: 上传的图像文件
        conf_threshold: 置信度阈值
        detailed: 是否返回详细信息

    Returns:
        YOLO 检测到的人脸信息
    """
    try:
        # 验证 YOLO 模型是否可用
        if yolo_detector is None:
            raise HTTPException(status_code=503, detail="YOLO 模型不可用")

        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")

        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 使用 YOLO 检测人脸（默认使用face11n模型，启用智能裁剪）
        logger.info(f"使用 YOLO 模型检测人脸，置信度阈值: {conf_threshold}")

        faces = yolo_detector.detect_faces(
            image,
            model_name="face11n",
            conf_threshold=conf_threshold,
            enable_smart_crop=True
        )

        # 转换结果为可序列化格式
        result = {
            "success": True,
            "face_count": len(faces),
            "model_used": "yolo",
            "model_info": yolo_detector.get_model_info(),
            "confidence_threshold": conf_threshold,
            "detailed_result": detailed,
            "faces": []
        }

        for i, face in enumerate(faces):
            face_info = {
                "id": face.get("id", i),
                "bbox": face["bbox"],
                "confidence": round(face["confidence"], 3),
                "keypoints": [[round(x, 3), round(y, 3)] for x, y in face["keypoints"]]
            }

            # 添加详细信息（如果可用）
            if detailed:
                face_info.update({
                    "detection_type": face.get("detection_type", "yolo"),
                    "area": face.get("area"),
                    "aspect_ratio": face.get("aspect_ratio"),
                    "center": face.get("center"),
                    "landmarks_estimate": face.get("landmarks_estimate", [])
                })

            result["faces"].append(face_info)

        logger.info(f"YOLO 检测到 {len(faces)} 个人脸")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO 人脸检测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YOLO 人脸检测失败: {str(e)}")

@app.post("/detect_faces_multi_yolo")
async def detect_faces_multi_yolo(
    file: UploadFile = File(...),
    model: str = Query("face11s", description="YOLO 模型: face11n, face11s, face11m, face11l, face11x, face11sss"),
    conf_threshold: float = Query(0.5, description="置信度阈值"),
    detailed: bool = Query(False, description="是否返回详细信息")
):
    """
    使用指定的 YOLO 模型检测人脸

    Args:
        file: 上传的图像文件
        model: YOLO 模型名称
        conf_threshold: 置信度阈值
        detailed: 是否返回详细信息

    Returns:
        指定 YOLO 模型的人脸检测结果
    """
    try:
        # 验证 YOLO 检测器是否可用
        if yolo_detector is None:
            raise HTTPException(status_code=503, detail="YOLO 检测器不可用")

        # 验证模型是否可用
        if model not in yolo_detector.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"模型 {model} 不可用。可用模型: {', '.join(yolo_detector.available_models)}"
            )

        # 验证文件类型
        logger.info(f"接收到文件: {file.filename}, MIME类型: {file.content_type}")
        logger.info(f"使用 YOLO 模型: {model}")

        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 使用指定的 YOLO 模型检测人脸（启用智能裁剪）
        logger.info(f"使用 {model} 模型检测人脸，置信度阈值: {conf_threshold}")

        faces = yolo_detector.detect_faces(
            image,
            model_name=model,
            conf_threshold=conf_threshold,
            enable_smart_crop=True
        )

        # 获取图像信息
        height, width = image.shape[:2]
        smart_crop_used = width > 800 or height > 800

        # 转换结果为可序列化格式
        result = {
            "success": True,
            "face_count": len(faces),
            "model_used": model,
            "model_info": yolo_detector.get_model_info(model),
            "confidence_threshold": conf_threshold,
            "detailed_result": detailed,
            "image_info": {
                "width": width,
                "height": height,
                "smart_crop_used": smart_crop_used,
                "smart_crop_enabled": True
            },
            "faces": []
        }

        for i, face in enumerate(faces):
            face_info = {
                "id": face.get("id", i),
                "bbox": face["bbox"],
                "confidence": round(face["confidence"], 3),
                "keypoints": [[round(x, 3), round(y, 3)] for x, y in face["keypoints"]]
            }

            # 添加详细信息（如果可用）
            if detailed:
                face_info.update({
                    "detection_type": face.get("detection_type", f"yolo_{model}"),
                    "area": face.get("area"),
                    "aspect_ratio": face.get("aspect_ratio"),
                    "center": face.get("center"),
                    "model_used": face.get("model_used", model),
                    "landmarks_estimate": face.get("landmarks_estimate", [])
                })

            result["faces"].append(face_info)

        logger.info(f"{model} 检测到 {len(faces)} 个人脸")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多模型 YOLO 人脸检测错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"多模型 YOLO 人脸检测失败: {str(e)}")

@app.post("/benchmark_yolo_models")
async def benchmark_yolo_models(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, description="置信度阈值")
):
    """
    对所有可用的 YOLO 模型进行性能基准测试

    Args:
        file: 上传的图像文件
        conf_threshold: 置信度阈值

    Returns:
        所有 YOLO 模型的性能对比结果
    """
    try:
        # 验证 YOLO 检测器是否可用
        if yolo_detector is None:
            raise HTTPException(status_code=503, detail="YOLO 检测器不可用")

        # 验证文件类型
        logger.info(f"开始 YOLO 模型基准测试，文件: {file.filename}")

        supported_types = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/webp',
            'image/bmp', 'image/tiff', 'image/gif'
        ]

        supported_extensions = [
            '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif'
        ]

        file_ext = os.path.splitext(file.filename)[1].lower()
        is_valid_type = (file.content_type in supported_types or
                        file_ext in supported_extensions or
                        file.content_type == 'application/octet-stream')

        if not is_valid_type:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图像格式: {file.content_type}。支持的格式: {', '.join(supported_types)}"
            )

        # 读取图像数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")

        # 运行基准测试
        logger.info("开始运行 YOLO 模型基准测试...")
        benchmark_results = multi_yolo_detector.benchmark_models(image, conf_threshold)

        # 获取所有模型信息
        all_models_info = multi_yolo_detector.get_available_models()

        result = {
            "success": True,
            "benchmark_config": {
                "image_file": file.filename,
                "image_size": f"{image.shape[1]}x{image.shape[0]}",
                "confidence_threshold": conf_threshold
            },
            "available_models": all_models_info,
            "benchmark_results": benchmark_results
        }

        # 添加性能排名
        successful_results = {k: v for k, v in benchmark_results.items() if 'avg_response_time_ms' in v}
        if successful_results:
            # 按响应时间排序
            ranked_by_speed = sorted(successful_results.items(), key=lambda x: x[1]['avg_response_time_ms'])
            result["rankings"] = {
                "fastest": ranked_by_speed[0][0],
                "slowest": ranked_by_speed[-1][0],
                "speed_ranking": [{"model": k, "response_time_ms": v["avg_response_time_ms"]} for k, v in ranked_by_speed]
            }

        logger.info(f"基准测试完成，测试了 {len(benchmark_results)} 个模型")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO 模型基准测试错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YOLO 模型基准测试失败: {str(e)}")

if __name__ == "__main__":
    import os

    # 支持环境变量配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # 生产环境关闭reload
        log_level="info"
    )