#!/usr/bin/env python3
"""
综合人脸检测测试脚本
整合MediaPipe和YOLO多模型的完整测试功能
"""

import requests
import json
import time
import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveFaceDetector:
    """综合人脸检测器"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.models = ["face11sss", "face11n", "face11s", "face11m", "face11l", "face11x"]

    def test_single_image(self, image_path: str, model_name: str = None) -> Dict:
        """测试单张图片"""
        try:
            logger.info(f"测试图片: {image_path}, 模型: {model_name or '默认'}")

            with open(image_path, 'rb') as f:
                if model_name:
                    response = requests.post(
                        f"{self.api_url}/detect_faces_multi_yolo",
                        files={"file": f},
                        params={"model": model_name}
                    )
                else:
                    response = requests.post(
                        f"{self.api_url}/detect_faces",
                        files={"file": f}
                    )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API调用失败: {response.status_code}"}

        except Exception as e:
            logger.error(f"测试失败: {str(e)}")
            return {"error": str(e)}

    def test_all_models(self, image_path: str) -> Dict:
        """测试所有YOLO模型"""
        results = {}

        for model in self.models:
            logger.info(f"测试模型: {model}")
            result = self.test_single_image(image_path, model)
            results[model] = result

            if "error" not in result:
                logger.info(f"✅ {model}: 检测到 {result.get('face_count', 0)} 个人脸")
            else:
                logger.error(f"❌ {model}: {result['error']}")

        return results

    def run_benchmark(self, image_path: str) -> Dict:
        """运行基准测试"""
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/benchmark_yolo_models",
                    files={"file": f}
                )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"基准测试失败: {response.status_code}"}

        except Exception as e:
            logger.error(f"基准测试失败: {str(e)}")
            return {"error": str(e)}

    def get_available_models(self) -> Dict:
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.api_url}/models")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"获取模型列表失败: {response.status_code}"}
        except Exception as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return {"error": str(e)}

    def generate_report(self, results: Dict, image_path: str, output_file: str = "comprehensive_test_report.json"):
        """生成测试报告"""
        report = {
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "image_size": self.get_image_size(image_path),
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        # 生成markdown报告
        self.generate_markdown_report(report, output_file.replace('.json', '.md'))

        logger.info(f"报告已保存: {output_file}")

    def get_image_size(self, image_path: str) -> Tuple[int, int]:
        """获取图片尺寸"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                return (width, height)
        except:
            pass
        return (0, 0)

    def generate_markdown_report(self, report: Dict, output_file: str):
        """生成markdown格式报告"""
        try:
            with open(output_file, 'w') as f:
                f.write("# 🎯 人脸检测综合测试报告\n\n")
                f.write(f"**测试时间**: {report['test_time']}\n")
                f.write(f"**测试图片**: {report['image_path']}\n")
                f.write(f"**图片尺寸**: {report['image_size'][0]}x{report['image_size'][1]}\n\n")

                if "benchmark_results" in report["results"]:
                    # 基准测试结果
                    benchmark = report["results"]["benchmark_results"]
                    f.write("## 📊 基准测试结果\n\n")
                    f.write("| 模型 | 检测数量 | 响应时间(ms) | 状态 |\n")
                    f.write("|------|----------|--------------|------|\n")

                    for model, data in benchmark.get("benchmark_results", {}).items():
                        face_count = data.get("avg_faces_detected", 0)
                        response_time = data.get("avg_response_time_ms", 0)
                        f.write(f"| {model} | {face_count} | {response_time:.2f} | ✅ |\n")

                f.write("\n## 💡 使用说明\n\n")
                f.write("### API端点\n")
                f.write("- `GET /models` - 获取可用模型列表\n")
                f.write("- `POST /detect_faces` - MediaPipe人脸检测\n")
                f.write("- `POST /detect_faces_multi_yolo?model=<model_name>` - YOLO多模型检测\n")
                f.write("- `POST /benchmark_yolo_models` - YOLO模型基准测试\n\n")

                f.write("### Python示例\n")
                f.write("```python\n")
                f.write("import requests\n\n")
                f.write("# 测试单张图片\n")
                f.write("with open('image.jpg', 'rb') as f:\n")
                f.write("    response = requests.post(\n")
                f.write("        'http://localhost:8000/detect_faces',\n")
                f.write("        files={'file': f}\n")
                f.write("    )\n\n")
                f.write("# 使用指定YOLO模型\n")
                f.write("with open('image.jpg', 'rb') as f:\n")
                f.write("    response = requests.post(\n")
                f.write("        'http://localhost:8000/detect_faces_multi_yolo',\n")
                f.write("        files={'file': f},\n")
                f.write("        params={'model': 'face11n'}\n")
                f.write("    )\n")
                f.write("```\n")

        except Exception as e:
            logger.error(f"生成markdown报告失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合人脸检测测试工具')
    parser.add_argument('--image', '-i', help='测试图片路径', default='selfie.jpg')
    parser.add_argument('--model', '-m', help='指定模型名称')
    parser.add_argument('--benchmark', '-b', action='store_true', help='运行基准测试')
    parser.add_argument('--url', '-u', help='API服务地址', default='http://localhost:8000')

    args = parser.parse_args()

    detector = ComprehensiveFaceDetector(args.url)

    # 检查API是否可用
    models = detector.get_available_models()
    if "error" in models:
        logger.error(f"API服务不可用: {models['error']}")
        logger.info("请确保API服务正在运行: uv run python main.py")
        return

    logger.info("🚀 开始综合人脸检测测试")
    logger.info(f"API地址: {args.url}")
    logger.info(f"测试图片: {args.image}")

    if not os.path.exists(args.image):
        logger.error(f"图片文件不存在: {args.image}")
        return

    # 运行测试
    if args.benchmark:
        logger.info("运行基准测试...")
        results = detector.run_benchmark(args.image)
    elif args.model:
        logger.info(f"测试指定模型: {args.model}")
        results = {"single_model": detector.test_single_image(args.image, args.model)}
    else:
        logger.info("测试所有模型...")
        results = detector.test_all_models(args.image)

    # 生成报告
    detector.generate_report(results, args.image)
    logger.info("✅ 测试完成!")

if __name__ == "__main__":
    main()