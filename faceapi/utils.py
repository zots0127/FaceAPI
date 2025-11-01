"""
FaceAPI 工具模块
提供便捷的工具函数和实用方法
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_face(image: np.ndarray, bbox: List[int], margin: int = 20) -> np.ndarray:
    """
    从图像中提取人脸区域

    Args:
        image: 输入图像
        bbox: 边界框 [x, y, width, height]
        margin: 边距像素

    Returns:
        提取的人脸图像
    """
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


def draw_landmarks(image: np.ndarray, landmarks: List[List[float]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   point_size: int = 1, line_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    在图像上绘制关键点

    Args:
        image: 输入图像
        landmarks: 关键点坐标列表 [[x1, y1], [x2, y2], ...]
        color: 点的颜色
        point_size: 点的大小
        line_color: 连线的颜色

    Returns:
        绘制了关键点的图像
    """
    result_image = image.copy()

    # 绘制关键点
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(result_image, (int(x), int(y)), point_size, color, -1)

    # 绘制面部轮廓
    if len(landmarks) == 468:
        # 面部轮廓点索引
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340,
                   346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 287,
                   273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165,
                   167, 164, 393, 391, 322, 410, 287, 273, 335, 321, 308, 324,
                   318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13,
                   312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        # 绘制轮廓连线
        for i in range(len(face_oval)):
            start_idx = face_oval[i]
            end_idx = face_oval[(i + 1) % len(face_oval)]

            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                cv2.line(result_image, start_point, end_point, line_color, 1)

    return result_image


def save_detection_result(image: np.ndarray, faces: List[Dict[str, Any]],
                         output_path: str, draw_confidence: bool = True) -> bool:
    """
    保存检测结果到图像文件

    Args:
        image: 原始图像
        faces: 检测到的人脸列表
        output_path: 输出文件路径
        draw_confidence: 是否绘制置信度

    Returns:
        保存是否成功
    """
    try:
        result_image = image.copy()

        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            confidence = face.get('confidence', 0.0)

            # 绘制边界框
            color = (0, 255, 0)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

            # 绘制人脸ID和置信度
            label = f"Face {i+1}"
            if draw_confidence:
                label += f": {confidence:.2f}"

            cv2.putText(result_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存图像
        cv2.imwrite(output_path, result_image)
        logger.info(f"检测结果已保存到: {output_path}")
        return True

    except Exception as e:
        logger.error(f"保存检测结果失败: {e}")
        return False


def calculate_face_angle(landmarks: List[List[float]]) -> float:
    """
    根据关键点计算人脸角度

    Args:
        landmarks: 468个关键点坐标

    Returns:
        人脸角度（度）
    """
    try:
        if len(landmarks) < 468:
            return 0.0

        # 使用眼部关键点计算角度
        # 左眼关键点 (参考MediaPipe Face Mesh)
        left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # 右眼关键点
        right_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

        # 计算左眼中心
        left_eye_x = sum(landmarks[i][0] for i in left_eye_points if i < len(landmarks)) / len(left_eye_points)
        left_eye_y = sum(landmarks[i][1] for i in left_eye_points if i < len(landmarks)) / len(left_eye_points)

        # 计算右眼中心
        right_eye_x = sum(landmarks[i][0] for i in right_eye_points if i < len(landmarks)) / len(right_eye_points)
        right_eye_y = sum(landmarks[i][1] for i in right_eye_points if i < len(landmarks)) / len(right_eye_points)

        # 计算角度
        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y

        angle = math.degrees(math.atan2(dy, dx))

        return angle

    except Exception as e:
        logger.error(f"计算人脸角度失败: {e}")
        return 0.0


def align_face(image: np.ndarray, landmarks: List[List[float]]) -> np.ndarray:
    """
    根据关键点对齐人脸

    Args:
        image: 输入图像
        landmarks: 468个关键点坐标

    Returns:
        对齐后的人脸图像
    """
    try:
        if len(landmarks) < 468:
            return image

        # 计算人脸角度
        angle = calculate_face_angle(landmarks)

        # 获取人脸边界框
        h, w = image.shape[:2]
        min_x = min(landmark[0] for landmark in landmarks)
        max_x = max(landmark[0] for landmark in landmarks)
        min_y = min(landmark[1] for landmark in landmarks)
        max_y = max(landmark[1] for landmark in landmarks)

        # 计算人脸中心
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

        # 旋转图像
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        return aligned_image

    except Exception as e:
        logger.error(f"人脸对齐失败: {e}")
        return image


def create_detection_video(input_video_path: str, output_video_path: str,
                          detector_func, output_fps: int = 30) -> bool:
    """
    创建检测结果视频

    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        detector_func: 检测函数
        output_fps: 输出视频帧率

    Returns:
        创建是否成功
    """
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {input_video_path}")
            return False

        # 获取视频信息
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔几帧处理一次（根据原帧率调整）
            if frame_count % max(1, original_fps // output_fps) == 0:
                # 检测人脸
                faces = detector_func(frame)

                # 绘制检测结果
                result_frame = frame.copy()
                for face in faces:
                    x, y, w, h = face['bbox']
                    confidence = face.get('confidence', 0.0)

                    # 绘制边界框
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(result_frame, f"{confidence:.2f}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                out.write(result_frame)
                processed_frames += 1

            frame_count += 1

            # 显示进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")

        # 释放资源
        cap.release()
        out.release()

        logger.info(f"视频处理完成: {output_video_path}")
        logger.info(f"总帧数: {total_frames}, 处理帧数: {processed_frames}")
        return True

    except Exception as e:
        logger.error(f"创建检测视频失败: {e}")
        return False


def batch_process_images(input_dir: str, output_dir: str,
                        detector_func, file_extensions: List[str] = None) -> Dict[str, Any]:
    """
    批量处理图像

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        detector_func: 检测函数
        file_extensions: 支持的文件扩展名

    Returns:
        处理结果统计
    """
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有图像文件
        image_files = []
        for ext in file_extensions:
            image_files.extend([f for f in os.listdir(input_dir)
                              if f.lower().endswith(ext.lower())])

        if not image_files:
            logger.warning(f"在目录 {input_dir} 中未找到支持的图像文件")
            return {"total_files": 0, "processed": 0, "failed": 0, "results": []}

        results = []
        processed_count = 0
        failed_count = 0

        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"detected_{filename}")

            try:
                # 读取图像
                image = cv2.imread(input_path)
                if image is None:
                    logger.warning(f"无法读取图像: {input_path}")
                    failed_count += 1
                    continue

                # 检测人脸
                faces = detector_func(image)

                # 保存结果
                success = save_detection_result(image, faces, output_path)

                if success:
                    processed_count += 1
                    results.append({
                        "filename": filename,
                        "faces_count": len(faces),
                        "output_path": output_path,
                        "success": True
                    })
                else:
                    failed_count += 1
                    results.append({
                        "filename": filename,
                        "error": "保存失败",
                        "success": False
                    })

            except Exception as e:
                logger.error(f"处理文件 {filename} 失败: {e}")
                failed_count += 1
                results.append({
                    "filename": filename,
                    "error": str(e),
                    "success": False
                })

        logger.info(f"批量处理完成: 总计 {len(image_files)} 个文件")
        logger.info(f"成功: {processed_count}, 失败: {failed_count}")

        return {
            "total_files": len(image_files),
            "processed": processed_count,
            "failed": failed_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return {"total_files": 0, "processed": 0, "failed": 0, "results": []}