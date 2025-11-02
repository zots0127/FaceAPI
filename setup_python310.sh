#!/bin/bash
# FaceAPI Python 3.10 环境设置脚本

set -e

echo "🐍 FaceAPI Python 3.10 环境设置"
echo "================================="

# 检查当前Python版本
echo "📋 当前环境信息:"
echo "Python版本: $(python --version 2>/dev/null || echo '未安装')"
echo "UV版本: $(uv --version 2>/dev/null || echo '未安装')"
echo "当前目录: $(pwd)"
echo ""

# 检查是否在正确的目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误: 请在FaceAPI项目根目录运行此脚本"
    exit 1
fi

# 方案1: 使用conda (如果可用)
if command -v conda &> /dev/null; then
    echo "✅ 检测到conda，准备创建Python 3.10环境..."

    # 检查是否已有faceapi环境
    if conda env list | grep -q "faceapi"; then
        echo "🔄 发现已存在的faceapi环境，正在激活..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate faceapi
    else
        echo "🆕 创建新的faceapi环境 (Python 3.10)..."
        conda create -n faceapi python=3.10 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate faceapi
    fi

    echo "✅ Python版本: $(python --version)"

# 方案2: 使用pyenv (如果可用)
elif command -v pyenv &> /dev/null; then
    echo "✅ 检测到pyenv，准备设置Python 3.10..."

    # 安装Python 3.10 (如果未安装)
    if ! pyenv versions | grep -q "3.10"; then
        echo "📦 安装Python 3.10..."
        pyenv install 3.10.12
    fi

    # 设置本地Python版本
    echo "🔧 设置本地Python版本为3.10..."
    pyenv local 3.10.12

    echo "✅ Python版本: $(python --version)"

else
    echo "⚠️  未检测到conda或pyenv"
    echo "💡 请手动安装Python 3.10:"
    echo "   - 使用conda: conda create -n faceapi python=3.10"
    echo "   - 使用pyenv: pyenv install 3.10.12 && pyenv local 3.10.12"
    echo "   - 或从官网安装: https://www.python.org/downloads/release/python-31012/"
    echo ""
    read -p "是否继续使用当前Python版本? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 已取消设置"
        exit 1
    fi
fi

# 检查UV安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装UV包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📋 环境设置完成后的信息:"
echo "Python版本: $(python --version)"
echo "UV版本: $(uv --version)"
echo ""

# 安装依赖
echo "📦 使用UV安装项目依赖..."
if [ -f "uv.lock" ]; then
    echo "🔄 检测到uv.lock文件，执行精确同步..."
    uv sync
else
    echo "🆕 执行全新安装..."
    uv sync --dev
fi

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "🚀 下一步操作:"
echo "1. 激活环境 (如果使用conda): conda activate faceapi"
echo "2. 运行快速测试: uv run python quick_test.py"
echo "3. 运行完整测试: uv run python examples/run_comprehensive_tests.py"
echo "4. 启动API服务: uv run python main.py"
echo ""

# 验证关键模块
echo "🔍 验证关键模块..."
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import mediapipe
    print('✅ MediaPipe: OK')
except ImportError as e:
    print(f'❌ MediaPipe: {e}')

try:
    import facenet_pytorch
    print('✅ FaceNet: OK')
except ImportError as e:
    print(f'❌ FaceNet: {e}')

try:
    import torch
    print('✅ PyTorch: OK')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import cv2
    print('✅ OpenCV: OK')
except ImportError as e:
    print(f'❌ OpenCV: {e}')
"

echo ""
echo "🎉 设置脚本执行完成！"