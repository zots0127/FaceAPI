# 🎯 FaceAPI GitHub 发布说明

## 📦 发布信息

- **项目名称**: FaceAPI - 智能人脸检测系统
- **GitHub仓库**: https://github.com/zots0127/FaceAPI
- **当前版本**: 1.0.0
- **许可证**: MIT License
- **发布状态**: 已准备好发布

## 🚀 安装方式

### 方法1: 一键安装 (推荐)
```bash
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI
./install.sh
```

### 方法2: 手动安装
```bash
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI

# 检查Python版本 (需要 >= 3.10)
python --version

# 安装依赖
pip install -r requirements.txt

# 验证安装 (会自动下载所有模型)
python verify_installation.py
```

### 方法3: 使用uv (推荐开发者)
```bash
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI

# 使用uv管理依赖
uv sync

# 启动服务
uv run python main.py
```

## 🎮 使用方式

### 启动服务
```bash
# 启动FastAPI服务器
./start.sh

# 启动Gradio Web界面
./gradio.sh

# 或者直接运行
python main.py
```

### 命令行工具
```bash
# 人脸检测
python -m faceapi.cli detect -i image.jpg --model mediapipe

# 关键点检测
python -m faceapi.cli landmarks -i image.jpg

# 查看模型信息
python -m faceapi.cli models

# 显示版本
python -m faceapi.cli version
```

### Python API
```python
from faceapi import detect_faces, get_landmarks
import cv2

# 读取图像
image = cv2.imread('test.jpg')

# 人脸检测
faces = detect_faces(image)
print(f"检测到 {len(faces)} 个人脸")

# 获取关键点
landmarks = get_landmarks(image)
print(f"提取到 {len(landmarks)} 组关键点")
```

## 📊 核心功能

### 🤖 多模型支持
- **MediaPipe**: 468个关键点检测
- **YOLO系列**: 6个不同规模的预训练模型
  - face11sss (0.9MB, 43.4 FPS)
  - face11n (5.2MB, 23.2 FPS)
  - face11s (18.3MB, 14.6 FPS)
  - face11m (38.6MB, 6.8 FPS)
  - face11l (48.8MB, 5.0 FPS)
  - face11x (217MB, 3.1 FPS)

### 🎯 主要特性
- **智能重叠裁剪**: 大图像检测提升2-3倍
- **468个关键点**: 精确面部特征检测
- **多人脸检测**: 最多同时检测10个人脸
- **Gradio界面**: 直观的Web交互界面
- **RESTful API**: 完整的HTTP接口
- **命令行工具**: 便捷的CLI工具

### 📡 API端点
- `GET /models` - 获取可用模型列表
- `POST /detect_faces` - 人脸检测
- `POST /face_landmarks` - 关键点检测
- `POST /extract_face` - 人脸提取
- `POST /detect_and_draw` - 结果可视化
- `POST /benchmark_yolo_models` - 模型基准测试

## 📈 性能数据

### SelfieBenchmark 评估结果
| 模型 | F1分数 | 精确率 | 召回率 | 文件大小 | FPS |
|------|--------|--------|--------|----------|-----|
| face11x | **0.808** | 0.917 | 0.722 | 217MB | 3.1 |
| face11n | **0.801** | 0.915 | 0.713 | 5.2MB | 23.2 |
| face11l | **0.800** | 0.922 | 0.707 | 48.8MB | 5.0 |
| face11s | 0.800 | 0.920 | 0.708 | 18.3MB | 14.6 |
| face11m | 0.793 | 0.920 | 0.696 | 38.6MB | 6.8 |
| face11sss | 0.745 | 0.916 | 0.627 | 0.9MB | 43.4 |

## 🔧 系统要求

### 最低要求
- **Python**: 3.10+
- **操作系统**: macOS, Linux, Windows
- **内存**: 建议4GB+
- **存储**: 2GB (包含模型文件)

### 推荐配置
- **Python**: 3.11+
- **内存**: 8GB+
- **GPU**: NVIDIA CUDA (可选，用于加速)
- **存储**: 5GB+

## 📁 项目结构

```
FaceAPI/
├── 📄 main.py                    # 主API服务
├── 📄 comprehensive_test.py         # 测试脚本
├── 📄 README.md                  # 项目说明
├── 📄 pyproject.toml             # 项目配置
├── 📄 requirements.txt            # 依赖列表
├── 📄 install.sh                 # 一键安装脚本
├── 📄 start.sh                   # 启动脚本
├── 📄 gradio.sh                  # Gradio启动脚本
├── 📄 verify_installation.py     # 安装验证
├── 📄 download_models.py         # 模型下载脚本
├── 📁 faceapi/                   # Python包
│   ├── 📄 __init__.py
│   ├── 📄 core.py                   # 核心模块
│   ├── 📄 utils.py                  # 工具函数
│   ├── 📄 cli.py                    # 命令行接口
│   ├── 📄 gradio_app.py            # Gradio界面
│   └── 📄 server.py                # 服务器入口
└── 📁 models/                   # YOLO模型文件
    ├── face11sss.pt (0.9MB)
    ├── face11n.pt (5.2MB)
    ├── face11s.pt (18.3MB)
    ├── face11m.pt (38.6MB)
    ├── face11l.pt (48.8MB)
    └── face11x.pt (217MB)
```

## 🐛 常见问题

### Q: 如何解决依赖安装问题？
A: 使用一键安装脚本 `./install.sh`，它会自动检测Python版本并安装所有依赖。脚本支持uv和pip两种安装方式，会自动fallback。

### Q: Linux服务器上安装失败怎么办？
A: 确保:
1. 安装了curl: `apt-get install curl` (Ubuntu) 或 `yum install curl` (CentOS)
2. Python版本 >= 3.10
3. 重新运行: `./install.sh`
4. 如果uv命令未找到，手动添加PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Q: 模型文件加载失败怎么办？
A: 运行 `uv run python download_models.py --model all` 重新下载所有模型，或使用 `uv run python download_models.py --list` 检查模型状态。

### Q: 如何提高检测精度？
A: 调整置信度阈值到0.3-0.7之间，使用face11x模型获得最高精度。

### Q: 如何提高检测速度？
A: 使用face11n或face11sss模型，禁用智能裁剪功能。

### Q: GPU加速如何启用？
A: 安装CUDA版本的PyTorch，系统会自动检测并使用GPU。

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📞 支持

- **Issues**: https://github.com/zots0127/FaceAPI/issues
- **Documentation**: 查看 `README.md` 和项目内的完整报告
- **API文档**: 启动服务后访问 `/docs`

## 📄 许可证

本项目采用 MIT 许可证，详情请查看 [LICENSE](LICENSE) 文件。

---

**感谢使用 FaceAPI！如果觉得有用，请给个 ⭐**