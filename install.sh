#!/bin/bash

# 🚀 FaceAPI 一键安装脚本
# 支持 macOS 和 Linux

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查操作系统
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# 主安装函数
main() {
    echo "🎯 FaceAPI 一键安装脚本"
    echo "=========================="
    echo ""

    # 检测操作系统
    OS=$(detect_os)
    print_info "检测到操作系统: $OS"

    # 检查 Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python 已安装: $PYTHON_VERSION"

        # 检查 Python 版本是否为 3.10.x
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 10 ]; then
            print_success "Python 版本符合要求 (= 3.10.x)"
        else
            print_error "Python 版本不符合要求，需要 3.10.x，当前版本: $PYTHON_VERSION"
            echo "请安装 Python 3.10.x 后重试"
            exit 1
        fi
    else
        print_error "Python3 未安装"
        echo "请先安装 Python 3.10.x 后重试"
        exit 1
    fi

    # 检查并安装 uv
    if command_exists uv; then
        UV_VERSION=$(uv --version)
        print_success "uv 已安装: $UV_VERSION"
    else
        print_info "正在安装 uv..."
        if command_exists curl; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # 重新加载环境变量 - uv安装在 $HOME/.local/bin
            export PATH="$HOME/.local/bin:$PATH"
            if [ -f "$HOME/.bashrc" ]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            fi
            if [ -f "$HOME/.zshrc" ]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            fi
            if [ -f "$HOME/.profile" ]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
            fi
            print_success "uv 安装完成"

            # 重新检测uv是否可用
            if command_exists uv; then
                UV_VERSION=$(uv --version)
                print_success "uv 验证成功: $UV_VERSION"
            else
                print_error "uv 安装后无法找到，尝试重新加载PATH"
                export PATH="$HOME/.local/bin:$PATH"
                if command_exists uv; then
                    UV_VERSION=$(uv --version)
                    print_success "uv 重新加载成功: $UV_VERSION"
                else
                    print_error "uv 安装失败，请手动安装"
                    exit 1
                fi
            fi
        else
            print_error "curl 未安装，无法自动安装 uv"
            echo "请手动安装 uv: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi
    fi

    # 检查系统依赖 (Linux)
    if [ "$OS" = "linux" ]; then
        print_info "检查系统依赖..."

        # 检查必要的系统包管理器
        if command_exists apt-get; then
            print_info "检测到apt包管理器，检查系统依赖..."
            # 检查是否安装了必要的系统包
            if ! dpkg -l | grep -q libgl1-mesa-glx; then
                print_info "安装OpenGL库..."
                apt-get update -qq && apt-get install -y libgl1-mesa-glx libglib2.0-0 || true
            fi
            if ! dpkg -l | grep -q libsm6; then
                print_info "安装X11库..."
                apt-get install -y libsm6 libxext6 libxrender-dev libfontconfig1 libice6 || true
            fi
        elif command_exists yum; then
            print_info "检测到yum包管理器，检查系统依赖..."
            yum install -y mesa-libGL glib2 libSM libXext libXrender fontconfig libICE || true
        fi
    fi

    # 检查并下载模型文件
    print_info "检查模型文件..."
    if [ -d "models" ]; then
        MODEL_COUNT=$(ls models/*.pt 2>/dev/null | wc -l)
        if [ "$MODEL_COUNT" -gt 0 ]; then
            print_success "找到 $MODEL_COUNT 个模型文件"
            ls -lh models/*.pt | awk '{print "  - " $9 " (" $5 ")"}'
        else
            print_warning "models 目录存在但未找到模型文件"
            echo "将自动下载 YOLO 模型文件..."
        fi
    else
        print_info "创建 models 目录..."
        mkdir -p models
        print_info "将自动下载 YOLO 模型文件..."
    fi

    # 下载模型文件
    print_info "下载 YOLO 模型文件..."
    if command_exists uv && uv run python download_models.py --list; then
        print_info "开始下载模型..."
        if uv run python download_models.py --model all; then
            print_success "所有模型下载完成"
        else
            print_warning "部分模型下载失败，但 FaceAPI 仍可正常工作"
            print_info "您可以稍后运行: uv run python download_models.py"
        fi
    else
        print_warning "模型下载脚本运行失败，尝试使用 pip"
        if pip install requests && python download_models.py --list; then
            print_info "开始下载模型..."
            if python download_models.py --model all; then
                print_success "所有模型下载完成"
            else
                print_warning "部分模型下载失败，但 FaceAPI 仍可正常工作"
                print_info "您可以稍后运行: python download_models.py"
            fi
        else
            print_warning "模型下载脚本运行失败"
            print_info "您可以稍后手动下载模型文件"
        fi
    fi

    # 安装依赖
    print_info "正在安装项目依赖..."
    if uv sync; then
        print_success "依赖安装完成"
    else
        print_warning "uv sync 失败，尝试使用 pip 安装"
        if pip install -r requirements.txt; then
            print_success "pip 安装完成"
        else
            print_error "依赖安装失败"
            exit 1
        fi
    fi

    # 设置执行权限
    print_info "设置脚本执行权限..."
    chmod +x start.sh
    chmod +x comprehensive_test.py

    # 创建环境变量文件
    if [ ! -f ".env" ]; then
        print_info "创建环境变量文件..."
        cp .env.example .env
        print_success "已创建 .env 文件，可根据需要修改配置"
    else
        print_info ".env 文件已存在，跳过创建"
    fi

    # 验证安装
    print_info "验证安装..."

    # 使用正确的命令验证依赖
    PYTHON_CMD="uv run python"
    if ! command_exists uv; then
        PYTHON_CMD="python"
    fi

    # 检查主要依赖
    if $PYTHON_CMD -c "import fastapi, uvicorn" 2>/dev/null; then
        print_success "FastAPI 和 Uvicorn 安装成功"
    else
        print_error "FastAPI 或 Uvicorn 安装失败"
        exit 1
    fi

    if $PYTHON_CMD -c "import cv2" 2>/dev/null; then
        print_success "OpenCV 安装成功"
    else
        print_warning "OpenCV 安装失败，这是Linux服务器常见问题"
        print_info "正在自动修复OpenCV安装..."

        # 卸载可能有问题的版本
        print_info "卸载旧版本..."
        pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true

        # 安装无GUI版本 (适合服务器环境)
        print_info "安装OpenCV无GUI版本..."
        if command_exists uv; then
            uv add opencv-python-headless 2>/dev/null || pip install opencv-python-headless
        else
            pip install opencv-python-headless
        fi

        # 再次验证
        if $PYTHON_CMD -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')" 2>/dev/null; then
            print_success "OpenCV 修复成功！"
        else
            print_error "OpenCV 自动修复失败，请手动执行："
            print_info "  pip uninstall opencv-python opencv-contrib-python"
            print_info "  pip install opencv-python-headless"
            print_info "如果仍然失败，可能需要安装系统依赖："
            if [ "$OS" = "linux" ]; then
                if command_exists apt-get; then
                    print_info "  sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6"
                elif command_exists yum; then
                    print_info "  sudo yum install mesa-libGL glib2 libSM libXext"
                fi
            fi
            exit 1
        fi
    fi

    if $PYTHON_CMD -c "import mediapipe" 2>/dev/null; then
        print_success "MediaPipe 安装成功"
    else
        print_warning "MediaPipe 安装失败，YOLO 功能仍可用"
    fi

    if $PYTHON_CMD -c "import ultralytics" 2>/dev/null; then
        print_success "Ultralytics YOLO 安装成功"
    else
        print_error "Ultralytics YOLO 安装失败"
        exit 1
    fi

    # 安装完成
    echo ""
    echo "🎉 FaceAPI 安装完成！"
    echo "==================="
    echo ""
    echo "🚀 启动服务:"
    echo "  ./start.sh"
    echo "  或: uv run python main.py"
    echo ""
    echo "📖 API 文档:"
    echo "  http://localhost:8000/docs"
    echo ""
    echo "🧪 运行测试:"
    echo "  uv run python comprehensive_test.py --help"
    echo ""
    echo "📁 项目结构:"
    echo "  main.py                    # 主 API 服务"
    echo "  comprehensive_test.py     # 测试脚本"
    echo "  FACE_API_COMPLETE_REPORT.md # 完整报告"
    echo "  models/                   # 模型文件目录"
    echo ""
    print_success "安装完成！可以开始使用 FaceAPI 了"
}

# 运行主函数
main "$@"