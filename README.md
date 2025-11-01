# 🎯 FaceAPI - 智能人脸检测系统

基于 MediaPipe + YOLO 的高性能人脸检测API，支持多模型选择、468个关键点检测和智能重叠裁剪技术。

## 🚀 核心特性

### 🤖 多模型支持

#### MediaPipe 人脸检测引擎
- **Face Detection**: 精确人脸边界框检测
- **Face Mesh**: **468个面部关键点**检测
- **Refine Landmarks**: 精细化关键点定位
- **Multi-face**: 支持多人脸同时检测 (最多10个)
- **高精度**: 置信度阈值0.5，精确度极高

#### YOLO系列: 6个不同规模的预训练模型
  - face11sss (0.9MB) - 超轻量级，43.4 FPS
  - face11n (5.2MB) - Nano版本，23.2 FPS
  - face11s (18.3MB) - 小型版本，14.6 FPS
  - face11m (38.6MB) - 中型版本，6.8 FPS
  - face11l (48.8MB) - 大型版本，5.0 FPS
  - face11x (217MB) - 超大型版本，3.1 FPS

### 🔧 技术特性
- **智能重叠裁剪**: 大图像自动分块处理，提升检测覆盖率2-3倍
- **多维度评估**: 符合SelfieBenchmark学术标准
- **RESTful API**: 完整的HTTP接口设计
- **高精度检测**: F1分数高达0.808
- **实时性能**: 支持实时视频流处理
- **Web界面**: Gradio交互式界面
- **命令行工具**: 便捷的CLI工具

## 📊 性能表现

### 🏆 综合性能排名 (基于SelfieBenchmark)
| 排名 | 模型 | F1分数 | 精确率 | 召回率 | 文件大小 | FPS |
|------|------|--------|--------|--------|----------|-----|
| 🥇 | **face11x** | **0.808** | 0.917 | 0.722 | 217MB | 3.1 |
| 🥈 | **face11n** | **0.801** | 0.915 | 0.713 | 5.2MB | 23.2 |
| 🥉 | **face11l** | **0.800** | 0.922 | 0.707 | 48.8MB | 5.0 |
| 4 | face11s | 0.800 | 0.920 | 0.708 | 18.3MB | 14.6 |
| 5 | face11m | 0.793 | 0.920 | 0.696 | 38.6MB | 6.8 |
| 6 | face11sss | 0.745 | 0.916 | 0.627 | 0.9MB | 43.4 |

### 🎯 应用场景推荐
- **实时应用**: face11sss (43.4 FPS) 或 face11n (23.2 FPS)
- **学术研究**: face11x (F1: 0.808, 最高精度)
- **视频处理**: face11s (14.6 FPS)
- **资源受限**: face11sss (0.9MB, 最佳性价比)

## 📦 安装方式

### 🚀 方式1: GitHub克隆安装 (推荐)

```bash
# 克隆项目
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI

# 一键安装和配置 (自动下载模型)
./install.sh

# 验证安装
python verify_installation.py

# 启动服务
./start.sh
```

### 📋 方式2: 本地pip开发模式

```bash
# 克隆项目后
cd FaceAPI

# 安装项目依赖 (开发模式)
pip install -e .

# 安装可选依赖
pip install -e ".[gradio]"  # Gradio界面
pip install -e ".[dev]"      # 开发工具
pip install -e ".[all]"       # 全功能
```

### 📋 方式3: uv包管理器 (推荐开发者)

```bash
# 克隆项目
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI

# 使用uv安装
uv sync

# 启动服务
uv run python main.py
```

## 🎮 使用方式

### 💻 命令行工具

```bash
# 克隆项目后使用
cd FaceAPI

# 启动FastAPI服务器
python -m faceapi.cli fastapi --host 0.0.0.0 --port 8000

# 启动Gradio Web界面
python -m faceapi.cli gradio --host 0.0.0.0 --port 7860 --share

# 检测图像中的人脸
python -m faceapi.cli detect -i image.jpg --model mediapipe --draw

# 提取人脸关键点
python -m faceapi.cli landmarks -i image.jpg -o landmarks.json

# 查看可用模型
python -m faceapi.cli models

# 显示版本信息
python -m faceapi.cli version

# 或者使用便捷脚本
./start.sh              # 启动FastAPI服务器
./gradio.sh             # 启动Gradio界面

# 手动下载/更新模型
uv run python download_models.py --list
uv run python download_models.py --model all
```

### 🌐 Web界面

1. **FastAPI接口**: `http://localhost:8000/docs`
2. **Gradio界面**: `http://localhost:7860`

### 📡 API接口

#### 核心端点

| 端点 | 方法 | 功能 | 参数 |
|------|------|------|------|
| `/models` | GET | 获取可用模型列表 | 无 |
| `/detect_faces` | POST | MediaPipe人脸检测 | model, conf_threshold |
| `/face_landmarks` | POST | **468个关键点检测** | 无 |
| `/extract_face` | POST | 人脸区域提取 | face_id, margin |
| `/detect_and_draw` | POST | 检测结果可视化 | 无 |
| `/detect_faces_yolo` | POST | YOLO默认模型检测 | conf_threshold, detailed |
| `/detect_faces_multi_yolo` | POST | 指定YOLO模型检测 | model, conf_threshold |
| `/benchmark_yolo_models` | POST | 多模型基准测试 | 无 |
| `/health` | GET | 健康检查 | 无 |

#### 使用示例

##### MediaPipe人脸检测
```bash
curl -X POST "http://localhost:8000/detect_faces?model=mediapipe&conf_threshold=0.5" \
     -F "file=@image.jpg"
```

##### 468个关键点检测
```bash
curl -X POST "http://localhost:8000/face_landmarks" \
     -F "file=@image.jpg"
```

##### 提取人脸区域
```bash
curl -X POST "http://localhost:8000/extract_face?face_id=0&margin=20" \
     -F "file=@image.jpg" \
     --output extracted_face.jpg
```

##### 检测并绘制可视化结果
```bash
curl -X POST "http://localhost:8000/detect_and_draw" \
     -F "file=@image.jpg" \
     --output result_with_faces.jpg
```

##### YOLO多模型检测
```bash
curl -X POST "http://localhost:8000/detect_faces_multi_yolo?model=face11n&conf_threshold=0.5" \
     -F "file=@image.jpg"
```

### 📊 API响应格式

#### MediaPipe人脸检测响应
```json
{
  "success": true,
  "face_count": 2,
  "model_used": "mediapipe",
  "faces": [
    {
      "id": 0,
      "bbox": [100, 150, 80, 100],
      "confidence": 0.95,
      "keypoints": []
    }
  ]
}
```

#### MediaPipe关键点检测响应
```json
{
  "success": true,
  "landmarks_count": 2,
  "landmarks": [
    {
      "id": 0,
      "count": 468,
      "landmarks": [[161, 134], [160, 133], ...]
    }
  ]
}
```

#### YOLO检测响应
```json
{
  "success": true,
  "face_count": 3,
  "model_used": "face11n",
  "model_info": {
    "name": "face11n",
    "available_models": ["face11n", "face11s", ...],
    "smart_crop_enabled": true
  },
  "image_info": {
    "width": 2048,
    "height": 1152,
    "smart_crop_used": true,
    "smart_crop_enabled": true
  },
  "faces": [
    {
      "id": 0,
      "bbox": [100, 150, 80, 100],
      "confidence": 0.925,
      "keypoints": []
    }
  ]
}
```

## 📈 智能重叠裁剪

### 🎯 技术原理
- **触发条件**: 图像尺寸 >800px 自动启用
- **裁剪尺寸**: 640×640，适合YOLO输入
- **重叠率**: 20%，确保边界覆盖
- **去重策略**: IoU阈值0.5，去除重复检测

### 📊 性能提升
| 图片尺寸 | 直接检测 | 智能裁剪 | 提升倍数 |
|----------|----------|----------|----------|
| 1920×1080 | 1.8人脸 | 5.2人脸 | **2.89x** |
| 2048×1152 | 2.1人脸 | 6.3人脸 | **3.00x** |
| 2560×1440 | 2.5人脸 | 7.1人脸 | **2.84x** |

## 🧪 编程接口

### Python API使用 (克隆项目后)

```python
# 克隆项目并进入目录
# git clone https://github.com/zots0127/FaceAPI.git
# cd FaceAPI

# 方式1: 使用便捷函数
from faceapi import detect_faces, get_landmarks, extract_face
import cv2

# 读取图像
image = cv2.imread('test.jpg')

# 人脸检测
faces = detect_faces(image, model='mediapipe')
print(f"检测到 {len(faces)} 个人脸")

# 获取关键点
landmarks = get_landmarks(image)
print(f"提取到 {len(landmarks)} 组关键点")

# 提取人脸
if faces:
    face_image = extract_face(image, faces[0]['bbox'], margin=20)
    cv2.imwrite('extracted_face.jpg', face_image)

# 方式2: 使用核心类
from faceapi.core import MediaPipeFaceDetector, MultiYOLODetector

# MediaPipe检测器
mediapipe_detector = MediaPipeFaceDetector()
faces = mediapipe_detector.detect_faces(image)

# YOLO检测器
yolo_detector = MultiYOLODetector()
faces = yolo_detector.detect_faces(image, model_name='face11n')

# 方式3: 使用工具函数
from faceapi.utils import save_detection_result, draw_landmarks

# 保存检测结果
save_detection_result(image, faces, 'result.jpg')

# 绘制关键点
result_image = draw_landmarks(image, landmarks[0]['landmarks'])
cv2.imwrite('landmarks_result.jpg', result_image)
```

## 🔧 配置选项

### 环境变量
- `LOG_LEVEL`: 日志级别 (默认: INFO)
- `HOST`: 服务主机地址 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 8000)
- `SMART_CROP_THRESHOLD`: 智能裁剪触发阈值 (默认: 800px)

### 模型参数
- `conf_threshold`: 检测置信度阈值 (默认: 0.5)
- `iou_threshold`: IoU去重阈值 (默认: 0.5)
- `smart_crop_overlap`: 裁剪重叠率 (默认: 0.2)

## 🏗️ 项目结构

```
FaceAPI/
├── 📄 main.py                           # 主API服务 (37KB)
├── 📄 comprehensive_test.py             # 测试脚本 (8KB)
├── 📄 FACE_API_COMPLETE_REPORT.md       # 完整评估报告 (15KB)
├── 📄 README.md                         # 项目说明
├── 📄 pyproject.toml                    # 项目配置
├── 📄 requirements.txt                  # 依赖列表
├── 📄 start.sh                          # 启动脚本
├── 📄 install.sh                        # 一键安装脚本
├── 📄 verify_installation.py           # 安装验证脚本
├── 📄 download_models.py                # 模型下载脚本
├── 📁 faceapi/                          # Python包
│   ├── 📄 __init__.py                   # 包初始化
│   ├── 📄 core.py                       # 核心检测模块
│   ├── 📄 utils.py                      # 工具函数
│   ├── 📄 cli.py                        # 命令行接口
│   ├── 📄 gradio_app.py                 # Gradio Web界面
│   └── 📄 server.py                     # 服务器入口
└── 📁 models/                           # YOLO模型目录 (1.5GB)
    ├── face11sss.pt (0.9MB)            # 超轻量级模型
    ├── face11n.pt (5.2MB)              # Nano版本
    ├── face11s.pt (18.3MB)             # 小型版本
    ├── face11m.pt (38.6MB)             # 中型版本
    ├── face11l.pt (48.8MB)             # 大型版本
    └── face11x.pt (217MB)              # 超大型版本
```

## 📖 API文档

启动服务后，访问以下地址查看交互式 API 文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型文件
   ls -la models/

   # 重新下载模型
   uv run python download_models.py --model all

   # 列出模型状态
   uv run python download_models.py --list
   ```

2. **GPU加速问题**
   ```bash
   # 检查CUDA可用性
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **内存不足**
   - 使用较小的模型 (face11n, face11s)
   - 调整智能裁剪参数
   - 减少并发请求数

4. **性能优化**
   - 启用智能裁剪处理大图像
   - 根据应用场景选择合适的模型
   - 调整置信度阈值平衡精度和速度

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

- **完整报告**: 查看 [FACE_API_COMPLETE_REPORT.md](./FACE_API_COMPLETE_REPORT.md) 获取详细的性能分析
- **API文档**: 启动服务后访问 `/docs` 查看交互式文档
- **问题反馈**: 请提交 GitHub Issue

---

**🎯 基于SelfieBenchmark标准评估，包含856个标注数据集的多维度测试结果**

## 🎮 快速体验

```bash
# 克隆项目
git clone https://github.com/zots0127/FaceAPI.git
cd FaceAPI

# 一键安装
./install.sh

# 启动Web界面
./gradio.sh

# 或启动API服务器
./start.sh

# 命令行检测
python -m faceapi.cli detect -i your_photo.jpg --model mediapipe --draw

# 开始使用吧！🚀
```

## 🌐 GitHub发布信息

- **仓库地址**: https://github.com/zots0127/FaceAPI
- **Star项目**: 如果觉得有用，请给个⭐
- **Issues**: 报告问题或建议新功能
- **Pull Requests**: 欢迎贡献代码

### 📋 发布说明
本项目目前发布在GitHub上，暂未上传到PyPI。用户需要先克隆项目到本地才能使用。

### 🔄 未来计划
- [ ] 上传到PyPI，支持 `pip install faceapi`
- [ ] 添加Docker支持
- [ ] 提供预编译的二进制文件
- [ ] 添加更多预训练模型