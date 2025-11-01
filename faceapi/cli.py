#!/usr/bin/env python3
"""
FaceAPI 命令行接口
提供便捷的命令行工具来启动各种服务
"""

import click
import sys
import os
from typing import Optional

@click.group()
@click.version_option(version="1.0.0", prog_name="faceapi")
def main():
    """
    FaceAPI - 基于 MediaPipe + YOLO 的智能人脸检测API

    支持多模型选择和智能重叠裁剪技术的完整人脸检测解决方案。
    """
    pass

@main.command()
@click.option('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
@click.option('--port', default=8000, help='服务器端口 (默认: 8000)')
@click.option('--reload', is_flag=True, help='启用自动重载 (开发模式)')
@click.option('--workers', default=1, help='工作进程数 (默认: 1)')
@click.option('--log-level', default='info', help='日志级别 (默认: info)')
def fastapi(host: str, port: int, reload: bool, workers: int, log_level: str):
    """启动 FastAPI 服务器"""
    click.echo(f"🚀 启动 FaceAPI FastAPI 服务器...")
    click.echo(f"📡 地址: http://{host}:{port}")
    click.echo(f"📖 API文档: http://{host}:{port}/docs")

    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=log_level
        )
    except ImportError:
        click.echo("❌ uvicorn 未安装，请运行: pip install uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ 启动失败: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
@click.option('--port', default=7860, help='服务器端口 (默认: 7860)')
@click.option('--share', is_flag=True, help='创建公共分享链接')
@click.option('--debug', is_flag=True, help='启用调试模式')
def gradio(host: str, port: int, share: bool, debug: bool):
    """启动 Gradio Web 界面"""
    click.echo(f"🎨 启动 FaceAPI Gradio Web 界面...")
    click.echo(f"📡 地址: http://{host}:{port}")

    try:
        # 动态导入gradio_app模块
        from . import gradio_app
        gradio_app.launch(host=host, port=port, share=share, debug=debug)
    except ImportError:
        click.echo("❌ gradio 未安装，请运行: pip install faceapi[gradio]", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ 启动失败: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--image', '-i', required=True, help='输入图像路径')
@click.option('--model', default='mediapipe', help='检测模型 (mediapipe, face11n, face11s, etc.)')
@click.option('--output', '-o', help='输出图像路径')
@click.option('--confidence', default=0.5, help='置信度阈值 (默认: 0.5)')
@click.option('--draw', is_flag=True, help='绘制检测结果')
def detect(image: str, model: str, output: Optional[str], confidence: float, draw: bool):
    """检测图像中的人脸"""
    if not os.path.exists(image):
        click.echo(f"❌ 图像文件不存在: {image}", err=True)
        sys.exit(1)

    click.echo(f"🔍 检测图像: {image}")
    click.echo(f"🤖 使用模型: {model}")
    click.echo(f"📊 置信度阈值: {confidence}")

    try:
        import cv2
        import numpy as np

        # 读取图像
        img = cv2.imread(image)
        if img is None:
            click.echo(f"❌ 无法读取图像: {image}", err=True)
            sys.exit(1)

        # 选择检测器
        if model.lower() == 'mediapipe':
            from .core import MediaPipeFaceDetector
            detector = MediaPipeFaceDetector()
            faces = detector.detect_faces(img)
        else:
            from .core import MultiYOLODetector
            detector = MultiYOLODetector()
            faces = detector.detect_faces(img, model_name=model, conf_threshold=confidence)

        click.echo(f"✅ 检测到 {len(faces)} 个人脸")

        # 绘制结果
        if draw or output:
            for face in faces:
                x, y, w, h = face['bbox']
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{face['confidence']:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存结果
        if output:
            cv2.imwrite(output, img)
            click.echo(f"💾 结果已保存: {output}")
        elif draw:
            output_name = f"detected_{os.path.basename(image)}"
            cv2.imwrite(output_name, img)
            click.echo(f"💾 结果已保存: {output_name}")

    except Exception as e:
        click.echo(f"❌ 检测失败: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--image', '-i', required=True, help='输入图像路径')
@click.option('--output', '-o', help='输出文件路径 (默认: landmarks.json)')
def landmarks(image: str, output: Optional[str]):
    """提取人脸关键点"""
    if not os.path.exists(image):
        click.echo(f"❌ 图像文件不存在: {image}", err=True)
        sys.exit(1)

    click.echo(f"📍 提取关键点: {image}")

    try:
        import cv2
        import json

        # 读取图像
        img = cv2.imread(image)
        if img is None:
            click.echo(f"❌ 无法读取图像: {image}", err=True)
            sys.exit(1)

        # 检测关键点
        from .core import MediaPipeFaceDetector
        detector = MediaPipeFaceDetector()
        landmarks_data = detector.get_landmarks(img)

        click.echo(f"✅ 提取到 {len(landmarks_data)} 组关键点")

        # 保存结果
        output_file = output or "landmarks.json"
        with open(output_file, 'w') as f:
            json.dump(landmarks_data, f, indent=2)

        click.echo(f"💾 关键点已保存: {output_file}")

    except Exception as e:
        click.echo(f"❌ 提取失败: {e}", err=True)
        sys.exit(1)

@main.command()
def models():
    """显示可用模型信息"""
    click.echo("🤖 可用模型信息:")
    click.echo("=" * 50)

    try:
        from .core import MediaPipeFaceDetector, MultiYOLODetector

        # MediaPipe信息
        click.echo("📱 MediaPipe:")
        click.echo("  - Face Detection: 人脸边界框检测")
        click.echo("  - Face Mesh: 468个面部关键点")
        click.echo("  - Refine Landmarks: 精细化关键点")
        click.echo("")

        # YOLO模型信息
        yolo_detector = MultiYOLODetector()
        if yolo_detector.available_models:
            click.echo("🎯 YOLO 模型:")
            for model in yolo_detector.available_models:
                click.echo(f"  - {model}")
        else:
            click.echo("❌ YOLO 模型未加载")

    except Exception as e:
        click.echo(f"❌ 获取模型信息失败: {e}", err=True)

@main.command()
def version():
    """显示版本信息"""
    click.echo("FaceAPI 版本信息:")
    click.echo(f"版本: 1.0.0")
    click.echo("作者: FaceAPI Team")
    click.echo("许可证: MIT")
    click.echo("")
    click.echo("核心功能:")
    click.echo("  - MediaPipe 人脸检测")
    click.echo("  - YOLO 多模型支持")
    click.echo("  - 468个面部关键点检测")
    click.echo("  - 智能重叠裁剪技术")
    click.echo("  - RESTful API 接口")
    click.echo("  - Gradio Web界面")

if __name__ == '__main__':
    main()