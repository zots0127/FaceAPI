#!/usr/bin/env python3
"""
FaceAPI 模型下载脚本
自动下载所需的YOLO人脸检测模型文件
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import time

# 模型配置
MODELS_CONFIG = {
    "face11sss.pt": {
        "url": "https://d.9.af/files/face/face11/face11sss.pt",
        "size": 933738,  # ~0.9MB (实际大小)
        "description": "超轻量级模型 - 43.4 FPS"
    },
    "face11n.pt": {
        "url": "https://d.9.af/files/face/face11/face11n.pt",
        "size": 5485347,  # ~5.2MB
        "description": "Nano版本 - 23.2 FPS"
    },
    "face11s.pt": {
        "url": "https://d.9.af/files/face/face11/face11s.pt",
        "size": 19259829,  # ~18.3MB
        "description": "小型版本 - 14.6 FPS"
    },
    "face11m.pt": {
        "url": "https://d.9.af/files/face/face11/face11m.pt",
        "size": 40697829,  # ~38.6MB
        "description": "中型版本 - 6.8 FPS"
    },
    "face11l.pt": {
        "url": "https://d.9.af/files/face/face11/face11l.pt",
        "size": 51322877,  # ~48.8MB
        "description": "大型版本 - 5.0 FPS"
    },
    "face11x.pt": {
        "url": "https://d.9.af/files/face/face11/face11x.pt",
        "size": 228698877,  # ~217MB
        "description": "超大型版本 - 3.1 FPS"
    }
}

class ModelDownloader:
    """模型下载器"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, filepath: Path, expected_size: int,
                     timeout: int = 30, max_retries: int = 3) -> bool:
        """下载单个文件"""
        for attempt in range(max_retries):
            try:
                print(f"  📥 下载中... (尝试 {attempt + 1}/{max_retries})")

                # 流式下载
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 开始下载
                downloaded_size = 0
                start_time = time.time()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                speed = downloaded_size / (time.time() - start_time) / 1024 / 1024  # MB/s
                                print(f"    进度: {progress:.1f}% ({speed:.1f} MB/s)", end='\r')

                print()  # 换行

                # 验证文件大小
                actual_size = filepath.stat().st_size
                if abs(actual_size - expected_size) / expected_size > 0.01:  # 允许1%误差
                    print(f"  ⚠️  文件大小不匹配: 期望 {expected_size} 字节, 实际 {actual_size} 字节")
                    filepath.unlink()
                    return False

                print(f"  ✅ 下载完成: {actual_size:,} 字节")
                return True

            except requests.exceptions.RequestException as e:
                print(f"  ❌ 下载失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"  ⏳ {2 ** attempt}秒后重试...")
                    time.sleep(2 ** attempt)
                else:
                    if filepath.exists():
                        filepath.unlink()
                    return False
            except Exception as e:
                print(f"  ❌ 未知错误: {e}")
                if filepath.exists():
                    filepath.unlink()
                return False

        return False

    def verify_model(self, model_name: str, filepath: Path) -> bool:
        """验证模型文件"""
        if not filepath.exists():
            return False

        # 检查文件大小
        actual_size = filepath.stat().st_size
        expected_size = MODELS_CONFIG[model_name]["size"]

        if abs(actual_size - expected_size) / expected_size > 0.01:
            print(f"  ⚠️  {model_name} 文件大小异常")
            return False

        print(f"  ✅ {model_name} 验证通过")
        return True

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """下载单个模型"""
        if model_name not in MODELS_CONFIG:
            print(f"❌ 未知模型: {model_name}")
            return False

        config = MODELS_CONFIG[model_name]
        filepath = self.models_dir / model_name

        # 检查文件是否已存在
        if filepath.exists() and not force:
            if self.verify_model(model_name, filepath):
                print(f"✅ {model_name} 已存在且有效")
                return True
            else:
                print(f"🔄 {model_name} 文件损坏，重新下载")

        print(f"📦 下载 {model_name}")
        print(f"   描述: {config['description']}")
        print(f"   大小: {config['size']:,} 字节 ({config['size'] / 1024 / 1024:.1f} MB)")
        print(f"   URL: {config['url']}")

        success = self.download_file(config['url'], filepath, config['size'])

        if success:
            print(f"🎉 {model_name} 下载成功!")
        else:
            print(f"💥 {model_name} 下载失败!")

        return success

    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """下载所有模型"""
        print("🚀 开始下载 FaceAPI YOLO 模型")
        print("=" * 50)

        results = {}
        total_size = sum(config['size'] for config in MODELS_CONFIG.values())

        print(f"📊 总计 {len(MODELS_CONFIG)} 个模型, {total_size:,} 字节 ({total_size / 1024 / 1024:.1f} MB)")
        print()

        start_time = time.time()

        for model_name in MODELS_CONFIG.keys():
            results[model_name] = self.download_model(model_name, force)
            print()

        end_time = time.time()

        # 统计结果
        success_count = sum(results.values())
        failed_models = [name for name, success in results.items() if not success]

        print("📋 下载结果统计")
        print("=" * 30)
        print(f"✅ 成功: {success_count}/{len(MODELS_CONFIG)} 个模型")
        print(f"⏱️  耗时: {end_time - start_time:.1f} 秒")

        if failed_models:
            print(f"❌ 失败: {len(failed_models)} 个模型")
            print(f"   失败模型: {', '.join(failed_models)}")

        return results

    def list_models(self) -> None:
        """列出所有模型状态"""
        print("📋 FaceAPI YOLO 模型列表")
        print("=" * 40)

        for model_name, config in MODELS_CONFIG.items():
            filepath = self.models_dir / model_name
            status = "✅ 已下载" if self.verify_model(model_name, filepath) else "❌ 缺失"
            size_mb = config['size'] / 1024 / 1024

            print(f"{model_name:<12} {status:<8} {size_mb:>6.1f} MB  {config['description']}")

        print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="FaceAPI YOLO 模型下载工具")
    parser.add_argument("--models-dir", "-d", default="models",
                       help="模型存储目录 (默认: models)")
    parser.add_argument("--model", "-m",
                       choices=list(MODELS_CONFIG.keys()) + ["all"],
                       default="all", help="下载指定模型 (默认: all)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="强制重新下载已存在的文件")
    parser.add_argument("--list", "-l", action="store_true",
                       help="列出模型状态")

    args = parser.parse_args()

    # 创建下载器
    downloader = ModelDownloader(args.models_dir)

    # 列出模型状态
    if args.list:
        downloader.list_models()
        return

    # 下载模型
    print(f"📁 模型目录: {downloader.models_dir.absolute()}")
    print()

    if args.model == "all":
        results = downloader.download_all(args.force)

        # 检查是否全部成功
        if all(results.values()):
            print("\n🎉 所有模型下载完成！FaceAPI 已准备就绪！")
            print("💡 运行 './start.sh' 启动API服务器")
            print("💡 运行 './gradio.sh' 启动Web界面")
        else:
            print("\n⚠️  部分模型下载失败，但FaceAPI仍可正常工作")
            print("💡 您可以稍后重新运行此脚本下载失败的模型")
            sys.exit(1)
    else:
        success = downloader.download_model(args.model, args.force)
        if success:
            print(f"\n🎉 {args.model} 下载完成！")
        else:
            print(f"\n💥 {args.model} 下载失败！")
            sys.exit(1)


if __name__ == "__main__":
    main()