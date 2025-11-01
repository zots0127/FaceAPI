#!/usr/bin/env python3
"""
FaceAPI Gradio Web界面
提供直观的Web界面进行人脸检测和关键点提取
"""

import gradio as gr
import cv2
import numpy as np
import json
import tempfile
import os
from typing import Tuple, List, Optional

try:
    from .core import MediaPipeFaceDetector, MultiYOLODetector
    from .utils import extract_face
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# 初始化检测器
if CORE_AVAILABLE:
    mediapipe_detector = MediaPipeFaceDetector()
    yolo_detector = MultiYOLODetector()
else:
    mediapipe_detector = None
    yolo_detector = None


def detect_faces_interface(image: np.ndarray, model: str, confidence: float,
                          enable_smart_crop: bool) -> Tuple[np.ndarray, str]:
    """
    人脸检测界面函数
    """
    if image is None:
        return None, "请上传图像"

    if not CORE_AVAILABLE:
        return image, "❌ 核心模块未正确加载"

    try:
        result_image = image.copy()
        faces = []

        # 选择检测模型
        if model == "MediaPipe":
            faces = mediapipe_detector.detect_faces(image)
        elif model.startswith("YOLO"):
            model_name = model.replace("YOLO ", "").lower()
            if model_name in yolo_detector.available_models:
                faces = yolo_detector.detect_faces(
                    image, model_name=model_name,
                    conf_threshold=confidence, enable_smart_crop=enable_smart_crop
                )
            else:
                return image, f"❌ 模型 {model_name} 不可用"

        # 绘制检测结果
        if faces:
            for i, face in enumerate(faces):
                x, y, w, h = face['bbox']
                conf = face['confidence']

                # 绘制边界框
                color = (0, 255, 0) if conf >= confidence else (0, 165, 255)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

                # 绘制标签
                label = f"Face {i+1}: {conf:.2f}"
                cv2.putText(result_image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            info = f"✅ 检测到 {len(faces)} 个人脸\n"
            info += f"使用模型: {model}\n"
            info += f"置信度阈值: {confidence}"
        else:
            info = "❌ 未检测到人脸"

        return result_image, info

    except Exception as e:
        return image, f"❌ 检测失败: {str(e)}"


def landmarks_interface(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    关键点检测界面函数
    """
    if image is None:
        return None, "请上传图像"

    if not CORE_AVAILABLE or mediapipe_detector is None:
        return image, "❌ MediaPipe检测器未加载"

    try:
        result_image = image.copy()
        landmarks_data = mediapipe_detector.get_landmarks(image)

        if landmarks_data:
            # 绘制关键点
            for face_data in landmarks_data:
                landmarks = face_data['landmarks']

                # 绘制所有关键点
                for i, (x, y) in enumerate(landmarks):
                    cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1)

                # 绘制面部轮廓
                # 面部轮廓点索引
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

                # 绘制轮廓连线
                for i in range(len(face_oval)):
                    start_point = landmarks[face_oval[i]]
                    end_point = landmarks[face_oval[(i + 1) % len(face_oval)]]
                    cv2.line(result_image, start_point, end_point, (255, 0, 0), 1)

            info = f"✅ 提取到 {len(landmarks_data)} 组关键点\n"
            info += f"每组关键点: 468个\n"
            info += f"总计: {len(landmarks_data) * 468} 个点"
        else:
            info = "❌ 未检测到人脸关键点"

        return result_image, info

    except Exception as e:
        return image, f"❌ 关键点提取失败: {str(e)}"


def extract_face_interface(image: np.ndarray, face_id: int, margin: int) -> Tuple[Optional[np.ndarray], str]:
    """
    人脸提取界面函数
    """
    if image is None:
        return None, "请上传图像"

    if not CORE_AVAILABLE or mediapipe_detector is None:
        return image, "❌ MediaPipe检测器未加载"

    try:
        # 检测人脸
        faces = mediapipe_detector.detect_faces(image)

        if not faces:
            return None, "❌ 未检测到人脸"

        if face_id >= len(faces):
            return None, f"❌ 人脸ID超出范围 (检测到 {len(faces)} 个人脸)"

        # 提取指定人脸
        bbox = faces[face_id]['bbox']
        extracted_face = extract_face(image, bbox, margin)

        info = f"✅ 成功提取人脸 {face_id + 1}\n"
        info += f"边界框: {bbox}\n"
        info += f"边距: {margin}px\n"
        info += f"提取尺寸: {extracted_face.shape[:2]}"

        return extracted_face, info

    except Exception as e:
        return None, f"❌ 人脸提取失败: {str(e)}"


def benchmark_interface(image: np.ndarray, confidence: float) -> str:
    """
    模型基准测试界面函数
    """
    if image is None:
        return "请上传图像"

    if not CORE_AVAILABLE:
        return "❌ 核心模块未正确加载"

    try:
        results = {}

        # MediaPipe测试
        if mediapipe_detector:
            start_time = cv2.getTickCount()
            faces_mediapipe = mediapipe_detector.detect_faces(image)
            end_time = cv2.getTickCount()
            time_mediapipe = (end_time - start_time) / cv2.getTickFrequency() * 1000

            results['MediaPipe'] = {
                'faces': len(faces_mediapipe),
                'time': f"{time_mediapipe:.1f}ms"
            }

        # YOLO模型测试
        if yolo_detector and yolo_detector.available_models:
            for model_name in yolo_detector.available_models[:3]:  # 只测试前3个模型
                start_time = cv2.getTickCount()
                faces_yolo = yolo_detector.detect_faces(
                    image, model_name=model_name, conf_threshold=confidence
                )
                end_time = cv2.getTickCount()
                time_yolo = (end_time - start_time) / cv2.getTickFrequency() * 1000

                results[f'YOLO {model_name}'] = {
                    'faces': len(faces_yolo),
                    'time': f"{time_yolo:.1f}ms"
                }

        # 格式化结果
        result_text = "🏁 基准测试结果\n" + "="*30 + "\n\n"

        for model, data in results.items():
            result_text += f"🤖 {model}:\n"
            result_text += f"   检测人脸: {data['faces']} 个\n"
            result_text += f"   处理时间: {data['time']}\n\n"

        return result_text

    except Exception as e:
        return f"❌ 基准测试失败: {str(e)}"


def create_interface():
    """创建Gradio界面"""

    # 检查可用模型
    available_models = ["MediaPipe"]
    if CORE_AVAILABLE and yolo_detector:
        for model in yolo_detector.available_models:
            available_models.append(f"YOLO {model.upper()}")

    # 创建界面
    with gr.Blocks(title="FaceAPI - 智能人脸检测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 FaceAPI - 智能人脸检测

        基于 MediaPipe + YOLO 的高性能人脸检测系统，支持多模型选择和智能重叠裁剪技术。
        """)

        with gr.Tabs():
            # 人脸检测标签页
            with gr.TabItem("🔍 人脸检测"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="numpy", label="输入图像")

                        with gr.Row():
                            model_choice = gr.Dropdown(
                                choices=available_models,
                                value="MediaPipe",
                                label="检测模型"
                            )
                            confidence_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                                label="置信度阈值"
                            )

                        smart_crop_checkbox = gr.Checkbox(
                            label="启用智能裁剪 (适用于大图像)",
                            value=True
                        )

                        detect_btn = gr.Button("🔍 检测人脸", variant="primary")

                    with gr.Column():
                        output_image = gr.Image(type="numpy", label="检测结果")
                        detect_info = gr.Textbox(label="检测信息", lines=3)

                detect_btn.click(
                    fn=detect_faces_interface,
                    inputs=[input_image, model_choice, confidence_slider, smart_crop_checkbox],
                    outputs=[output_image, detect_info]
                )

            # 关键点检测标签页
            with gr.TabItem("📍 关键点检测"):
                with gr.Row():
                    with gr.Column():
                        landmarks_input = gr.Image(type="numpy", label="输入图像")
                        landmarks_btn = gr.Button("📍 提取关键点", variant="primary")

                    with gr.Column():
                        landmarks_output = gr.Image(type="numpy", label="关键点可视化")
                        landmarks_info = gr.Textbox(label="关键点信息", lines=3)

                landmarks_btn.click(
                    fn=landmarks_interface,
                    inputs=[landmarks_input],
                    outputs=[landmarks_output, landmarks_info]
                )

            # 人脸提取标签页
            with gr.TabItem("✂️ 人脸提取"):
                with gr.Row():
                    with gr.Column():
                        extract_input = gr.Image(type="numpy", label="输入图像")

                        with gr.Row():
                            face_id_slider = gr.Slider(
                                minimum=0, maximum=10, value=0, step=1,
                                label="人脸ID"
                            )
                            margin_slider = gr.Slider(
                                minimum=0, maximum=100, value=20, step=5,
                                label="边距 (像素)"
                            )

                        extract_btn = gr.Button("✂️ 提取人脸", variant="primary")

                    with gr.Column():
                        extract_output = gr.Image(type="numpy", label="提取的人脸")
                        extract_info = gr.Textbox(label="提取信息", lines=4)

                extract_btn.click(
                    fn=extract_face_interface,
                    inputs=[extract_input, face_id_slider, margin_slider],
                    outputs=[extract_output, extract_info]
                )

            # 基准测试标签页
            with gr.TabItem("🏁 基准测试"):
                with gr.Row():
                    with gr.Column():
                        benchmark_input = gr.Image(type="numpy", label="测试图像")
                        benchmark_confidence = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                            label="置信度阈值"
                        )
                        benchmark_btn = gr.Button("🏁 开始测试", variant="primary")

                    with gr.Column():
                        benchmark_output = gr.Textbox(
                            label="测试结果",
                            lines=15,
                            max_lines=20
                        )

                benchmark_btn.click(
                    fn=benchmark_interface,
                    inputs=[benchmark_input, benchmark_confidence],
                    outputs=[benchmark_output]
                )

        # 底部信息
        gr.Markdown("""
        ---
        ### 📋 使用说明
        1. **人脸检测**: 选择模型并调整置信度阈值，点击检测按钮
        2. **关键点检测**: 提取468个面部关键点并可视化
        3. **人脸提取**: 选择特定人脸ID并调整边距进行提取
        4. **基准测试**: 对比不同模型的性能表现

        ### 🚀 技术特性
        - 🤖 **多模型支持**: MediaPipe + 6个YOLO模型
        - 📍 **468关键点**: 精确的面部特征点检测
        - ✂️ **智能裁剪**: 大图像自动分块处理
        - ⚡ **高性能**: 优化的推理速度
        """)

    return demo


def launch(host: str = "0.0.0.0", port: int = 7860, share: bool = False, debug: bool = False):
    """启动Gradio应用"""
    if not CORE_AVAILABLE:
        print("❌ 核心模块未正确加载，请检查依赖安装")
        return

    demo = create_interface()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True
    )


def main():
    """Gradio应用入口点"""
    launch()


if __name__ == "__main__":
    main()