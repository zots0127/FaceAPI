#!/usr/bin/env python3
"""
FaceAPI 安装验证脚本
验证所有组件是否正确安装和配置
"""

import sys
import os
import importlib
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python 版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✅ Python 版本符合要求 (>= 3.10)")
        return True
    else:
        print("❌ Python 版本不符合要求，需要 >= 3.10")
        return False

def check_package(package_name, description=""):
    """检查Python包是否可导入"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} 已安装 {description}")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装 {description}")
        return False

def check_file_exists(filepath, description=""):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"✅ {filepath} 存在 ({size:.1f}MB) {description}")
        return True
    else:
        print(f"❌ {filepath} 不存在 {description}")
        return False

def check_directory_exists(dirpath, description=""):
    """检查目录是否存在"""
    if os.path.exists(dirpath):
        print(f"✅ {dirpath} 目录存在 {description}")
        return True
    else:
        print(f"❌ {dirpath} 目录不存在 {description}")
        return False

def check_command_exists(command, description=""):
    """检查命令是否存在"""
    try:
        result = subprocess.run([command, '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ {command} 可用 ({version}) {description}")
            return True
        else:
            print(f"❌ {command} 不可用 {description}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {command} 不可用 {description}")
        return False

def main():
    """主验证函数"""
    print("🔍 FaceAPI 安装验证")
    print("=" * 40)
    print()

    # 检查Python版本
    python_ok = check_python_version()
    print()

    # 检查命令行工具
    print("🛠️  命令行工具检查:")
    print("-" * 20)
    uv_ok = check_command_exists('uv', '包管理器')
    print()

    # 检查核心Python包
    print("📦 Python包检查:")
    print("-" * 20)
    packages = {
        'fastapi': 'Web框架',
        'uvicorn': 'ASGI服务器',
        'cv2': 'OpenCV图像处理',
        'numpy': '数值计算',
        'mediapipe': 'MediaPipe人脸检测',
        'ultralytics': 'YOLO模型',
        'torch': 'PyTorch深度学习框架',
        'PIL': 'Pillow图像处理'
    }

    package_results = {}
    for package, desc in packages.items():
        package_results[package] = check_package(package, f'({desc})')
    print()

    # 检查项目文件
    print("📁 项目文件检查:")
    print("-" * 20)
    files = {
        'main.py': '主API服务',
        'comprehensive_test.py': '测试脚本',
        'pyproject.toml': '项目配置',
        'requirements.txt': '依赖列表',
        'start.sh': '启动脚本',
        'install.sh': '安装脚本',
        'README.md': '项目文档',
        '.env.example': '环境变量模板'
    }

    file_results = {}
    for filepath, desc in files.items():
        file_results[filepath] = check_file_exists(filepath, f'({desc})')
    print()

    # 检查目录结构
    print("📂 目录结构检查:")
    print("-" * 20)
    dirs = {
        'models': '模型文件目录',
        '.venv': '虚拟环境目录'
    }

    dir_results = {}
    for dirpath, desc in dirs.items():
        dir_results[dirpath] = check_directory_exists(dirpath, f'({desc})')
    print()

    # 检查模型文件
    print("🤖 模型文件检查:")
    print("-" * 20)
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
        if model_files:
            print(f"✅ 找到 {len(model_files)} 个模型文件:")
            for model in model_files:
                size = os.path.getsize(f'models/{model}') / (1024*1024)
                print(f"  - {model} ({size:.1f}MB)")
            models_ok = True
        else:
            print("❌ models目录存在但未找到.pt模型文件")
            models_ok = False
    else:
        print("❌ models目录不存在")
        models_ok = False
    print()

    # 总结验证结果
    print("📊 验证结果总结:")
    print("=" * 40)

    all_checks = [
        python_ok,
        uv_ok,
        all(package_results.values()),
        all(file_results.values()),
        all(dir_results.values()),
        models_ok
    ]

    if all(all_checks):
        print("🎉 所有检查通过！FaceAPI 已正确安装")
        print()
        print("🚀 启动服务:")
        print("  ./start.sh")
        print("  或: uv run python main.py")
        print()
        print("📖 API 文档: http://localhost:8000/docs")
        return True
    else:
        print("⚠️  部分检查失败，请检查上述错误信息")
        print()
        print("🔧 修复建议:")
        if not python_ok:
            print("  - 升级Python到3.10+版本")
        if not uv_ok:
            print("  - 安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
        if not all(package_results.values()):
            failed_packages = [p for p, ok in package_results.items() if not ok]
            print(f"  - 安装缺失的包: uv sync")
        if not models_ok:
            print("  - 下载模型文件: uv run python download_models.py")
        print()
        print("💡 或运行一键安装: ./install.sh")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)