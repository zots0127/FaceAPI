#!/bin/bash

# FaceAPI Gradio Web界面启动脚本

echo "🎨 启动 FaceAPI Gradio Web界面"
echo "================================="

# 确保 uv 在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，请先运行: ./install.sh"
    exit 1
fi

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查是否在正确的目录
if [ ! -f "faceapi/__init__.py" ]; then
    echo "❌ 请在FaceAPI项目根目录中运行此脚本"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动Gradio界面
echo "🌐 启动Gradio Web界面..."
echo "📡 访问地址: http://localhost:7860"
echo ""
uv run python -m faceapi.gradio_app --host 0.0.0.0 --port 7860