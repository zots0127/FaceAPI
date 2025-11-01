#!/bin/bash

# 人脸识别 API 启动脚本

echo "🚀 启动人脸识别 API..."

# 确保 uv 在 PATH 中
export PATH="$HOME/.local/bin:$PATH"

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，请先安装 uv"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "或者运行: ./install.sh"
    exit 1
fi

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if [[ $(echo "$python_version >= 3.10" | bc) -eq 0 ]]; then
    echo "❌ Python 版本过低，需要 Python 3.10+，当前版本: $python_version"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖..."
if uv sync; then
    echo "✅ uv 安装完成"
else
    echo "⚠️ uv 安装失败，尝试使用 pip"
    if pip install -r requirements.txt; then
        echo "✅ pip 安装完成"
    else
        echo "❌ 依赖安装失败"
        exit 1
    fi
fi

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件配置相关参数"
fi

# 启动服务
echo "🌟 启动服务..."

# 检查端口8000是否被占用
if command -v lsof &> /dev/null && lsof -i:8000 &> /dev/null; then
    echo "⚠️ 端口8000已被占用，尝试使用端口8001..."
    export PORT=8001
else
    export PORT=8000
fi

if command -v uv &> /dev/null; then
    echo "使用 uv 启动服务 (端口: $PORT)..."
    if uv run python main.py; then
        echo "✅ 使用 uv 启动成功"
    else
        echo "❌ uv 启动失败，尝试使用 python..."
        if python main.py; then
            echo "✅ 使用 python 启动成功"
        else
            echo "❌ 服务启动失败"
            exit 1
        fi
    fi
else
    echo "使用 python 启动服务 (端口: $PORT)..."
    if python main.py; then
        echo "✅ 使用 python 启动成功"
    else
        echo "❌ 服务启动失败"
        exit 1
    fi
fi