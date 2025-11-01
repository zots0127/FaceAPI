"""
FaceAPI - 基于 MediaPipe + YOLO 的高性能人脸检测API

支持多模型选择、468个关键点检测和智能重叠裁剪技术
"""

__version__ = "1.0.0"
__author__ = "FaceAPI Team"
__email__ = "team@faceapi.com"
__license__ = "MIT"

# 核心导入
try:
    from .core import MediaPipeFaceDetector, MultiYOLODetector, SmartCropper
    from .utils import detect_faces, get_landmarks, extract_face
    __all__ = [
        "MediaPipeFaceDetector",
        "MultiYOLODetector",
        "SmartCropper",
        "detect_faces",
        "get_landmarks",
        "extract_face",
        "__version__",
    ]
except ImportError:
    # 模块未正确安装，仅提供版本信息
    __all__ = ["__version__"]

def get_version():
    """获取版本号"""
    return __version__

def get_info():
    """获取项目信息"""
    return {
        "name": "faceapi",
        "version": __version__,
        "description": "基于 MediaPipe + YOLO 的高性能人脸检测API",
        "author": __author__,
        "license": __license__,
        "features": [
            "MediaPipe 人脸检测",
            "YOLO 多模型支持 (6个模型)",
            "468个关键点检测",
            "智能重叠裁剪",
            "RESTful API 接口",
            "Gradio Web界面"
        ]
    }