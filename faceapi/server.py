#!/usr/bin/env python3
"""
FaceAPI FastAPI 服务器入口
"""

import uvicorn
import sys
import os

def main():
    """启动FastAPI服务器"""
    # 添加当前目录到Python路径，以便导入main模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

    try:
        # 启动服务器
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    except ImportError:
        print("❌ 无法导入main模块，请确保在正确的目录中运行")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()