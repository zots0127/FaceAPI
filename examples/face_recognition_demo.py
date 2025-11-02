"""
人脸识别实用示例
演示实际应用场景中的人脸识别功能
"""

import cv2
import os
import sys
from pathlib import Path
import time
from typing import Dict, List

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from faceapi.face_recognition import FaceRecognition
from faceapi.simple_database import SimpleFaceDatabase


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceRecognitionDemo:
    """人脸识别演示类"""

    def __init__(self):
        self.recognizer = FaceRecognition()
        self.database = SimpleFaceDatabase()

    def register_person_from_image(self, name: str, image_path: str, overwrite: bool = True):
        """
        从图片注册人脸

        Args:
            name: 人员姓名
            image_path: 图片路径
            overwrite: 已存在同名条目时是否覆盖
        """
        try:
            # 读取图片
            if not os.path.exists(image_path):
                print(f"❌ 图片不存在: {image_path}")
                return False

            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图片: {image_path}")
                return False

            # 使用现有的faceapi进行人脸检测
            from faceapi.core import detect_faces

            print(f"正在处理图片: {image_path}")
            faces = detect_faces(image, model='mediapipe')

            if not faces:
                print(f"❌ 未在图片中检测到人脸: {image_path}")
                return False

            # 处理检测到的第一个人脸
            face = faces[0]
            bbox = face['bbox']

            # 提取人脸区域
            x, y, w, h = bbox
            face_region = image[y:y+h, x:x+w]

            # 提取特征
            feature = self.recognizer.extract_feature(face_region)

            # 注册到数据库
            success = self.database.register_face(
                name,
                feature,
                metadata={
                    'source_image': image_path,
                    'bbox': bbox,
                    'registered_at': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                overwrite=overwrite
            )

            if success:
                print(f"✅ 成功注册人脸: {name}")
                return True
            else:
                print(f"❌ 注册失败: {name}")
                return False

        except Exception as e:
            print(f"❌ 注册过程出错: {e}")
            return False

    def identify_person(self, image_path: str, threshold: float = 0.6):
        """
        识别图片中的人脸

        Args:
            image_path: 图片路径
            threshold: 相似度阈值
        """
        try:
            if not os.path.exists(image_path):
                print(f"❌ 图片不存在: {image_path}")
                return None

            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图片: {image_path}")
                return None

            # 人脸检测
            from faceapi.core import detect_faces
            faces = detect_faces(image, model='mediapipe')

            if not faces:
                print(f"❌ 未在图片中检测到人脸: {image_path}")
                return None

            results = []

            for i, face in enumerate(faces):
                bbox = face['bbox']
                x, y, w, h = bbox
                face_region = image[y:y+h, x:x+w]

                # 提取特征
                feature = self.recognizer.extract_feature(face_region)

                # 在数据库中搜索
                match_result = self.database.find_best_match(feature, threshold=threshold)

                result = {
                    'face_index': i,
                    'bbox': bbox,
                    'match': match_result,
                    'confidence': match_result[1] if match_result else 0.0
                }

                results.append(result)

                if match_result:
                    name, similarity = match_result
                    print(f"✅ 人脸 {i+1}: 识别为 {name} (相似度: {similarity:.4f})")
                else:
                    print(f"❓ 人脸 {i+1}: 未识别 (未匹配到已知人脸)")

            return results

        except Exception as e:
            print(f"❌ 识别过程出错: {e}")
            return None

    def verify_faces(self, image1_path: str, image2_path: str, threshold: float = 0.6):
        """
        验证两张图片是否为同一个人

        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径
            threshold: 相似度阈值
        """
        try:
            # 读取图片
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)

            if img1 is None or img2 is None:
                print("❌ 无法读取图片")
                return False

            # 人脸检测
            from faceapi.core import detect_faces

            faces1 = detect_faces(img1, model='mediapipe')
            faces2 = detect_faces(img2, model='mediapipe')

            if not faces1 or not faces2:
                print("❌ 其中一张图片未检测到人脸")
                return False

            # 使用第一个人脸进行比较
            face1 = faces1[0]
            face2 = faces2[0]

            # 提取人脸区域
            bbox1 = face1['bbox']
            bbox2 = face2['bbox']

            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            face_region1 = img1[y1:y1+h1, x1:x1+w1]
            face_region2 = img2[y2:y2+h2, x2:x2+w2]

            # 比较人脸
            result = self.recognizer.compare_faces(
                self.recognizer.extract_feature(face_region1),
                self.recognizer.extract_feature(face_region2),
                threshold=threshold
            )

            print(f"\n人脸验证结果:")
            print(f"图片1: {image1_path}")
            print(f"图片2: {image2_path}")
            print(f"相似度: {result['similarity']:.4f}")
            print(f"阈值: {threshold}")
            print(f"结果: {'✅ 同一个人' if result['is_same_person'] else '❌ 不同的人'}")

            return result['is_same_person']

        except Exception as e:
            print(f"❌ 验证过程出错: {e}")
            return False

    def list_registered_faces(self):
        """列出所有已注册的人脸"""
        faces = self.database.get_all_names()
        stats = self.database.get_statistics()

        print(f"\n已注册的人脸 ({len(faces)} 个):")
        print("-" * 40)

        for name in faces:
            metadata = self.database.get_metadata(name)
            source = metadata.get('source_image', '未知') if metadata else '未知'
            print(f"• {name} - 来源: {source}")

        print(f"\n数据库统计:")
        print(f"总人数: {stats['total_faces']}")
        print(f"特征维度: {list(stats['feature_dimensions'])}")
        print(f"数据库大小: {stats['database_size_mb']:.2f} MB")

    def save_database(self, path: str = None):
        """保存数据库"""
        if path:
            success = self.database.save_database(path)
        else:
            success = self.database.save_database()

        if success:
            print(f"✅ 数据库已保存")
        else:
            print("❌ 数据库保存失败")

    def load_database(self, path: str = None):
        """加载数据库"""
        if path:
            success = self.database.load_database(path)
        else:
            success = self.database.load_database()

        if success:
            print(f"✅ 数据库已加载，共 {len(self.database)} 个人脸")
        else:
            print("❌ 数据库加载失败")


def list_image_files(directory: Path) -> List[Path]:
    """列出目录中的有效图像文件"""
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )


def collect_registration_images(register_dir: Path) -> Dict[str, List[Path]]:
    """收集注册用的人脸图片，支持按子目录或文件命名"""
    mapping: Dict[str, List[Path]] = {}

    if not register_dir.exists():
        return mapping

    for entry in register_dir.iterdir():
        if entry.is_dir():
            images = list_image_files(entry)
            if images:
                mapping[entry.name] = images
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            mapping.setdefault(entry.stem, []).append(entry)

    return mapping


def print_sample_setup_instructions(base_dir: Path) -> None:
    """输出示例数据准备指南"""
    register_dir = base_dir / "register"
    probe_dir = base_dir / "probe"

    print("❗ 未找到示例人脸数据，请先准备真实照片后再运行脚本。")
    print("建议的目录结构如下：")
    print(f"{base_dir}/")
    print(f"  ├── register/   # 必须：注册用真人人脸照片")
    print(f"  │     ├── Alice/      # 每个人一个子目录，名称即登记姓名")
    print(f"  │     │     ├── img1.jpg")
    print(f"  │     │     └── img2.jpg")
    print(f"  │     └── Bob/")
    print(f"  └── probe/      # 可选：识别/验证待测图片")
    print(f"        ├── meeting_room.jpg")
    print(f"        └── entrance.png")
    print("")
    print("注意事项：")
    print("- 请确保照片清晰、光照均匀，单人正脸效果最佳")
    print("- 命名使用 ASCII 字母/数字，避免中文或空格导致路径解析问题")
    print(f"- 支持的图片格式：{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}")
    print("")
    print(f"将您的照片放入 {register_dir} 后再次运行本脚本。")


def main():
    """主演示函数"""
    print("人脸识别实用演示")
    print("=" * 50)

    base_dir = Path(__file__).parent / "sample_faces"
    register_dir = base_dir / "register"
    probe_dir = base_dir / "probe"

    registration_map = collect_registration_images(register_dir)
    if not registration_map:
        print_sample_setup_instructions(base_dir)
        return

    demo = FaceRecognitionDemo()

    print("\n" + "=" * 50)
    print("1. 注册人脸到数据库")
    print("=" * 50)

    for name, image_paths in registration_map.items():
        representative_image = image_paths[0]
        if len(image_paths) > 1:
            print(f"ℹ️ {name} 共提供 {len(image_paths)} 张照片，默认使用 {representative_image.name} 进行注册")

        demo.register_person_from_image(name, str(representative_image), overwrite=True)

    print("\n" + "=" * 50)
    print("2. 已注册人员概览")
    print("=" * 50)
    demo.list_registered_faces()

    print("\n" + "=" * 50)
    print("3. 人脸识别演示")
    print("=" * 50)

    probe_images = list_image_files(probe_dir)
    if not probe_images:
        print(f"未在 {probe_dir} 找到待识别图片，可添加后重新运行。")
    else:
        for image_path in probe_images:
            print(f"\n📷 检测文件: {image_path.name}")
            demo.identify_person(str(image_path))

    print("\n" + "=" * 50)
    print("4. 1:1 人脸验证演示")
    print("=" * 50)

    if len(probe_images) >= 2:
        print("示例：对 probe 目录前两张图片执行验证，结果仅供参考。")
        demo.verify_faces(str(probe_images[0]), str(probe_images[1]))
    else:
        print("请在 probe 目录准备至少两张待比对图片，以体验 1:1 验证流程。")

    print("\n" + "=" * 50)
    print("5. 数据库存档")
    print("=" * 50)
    demo.save_database(str(base_dir / "faces_demo.db"))

    print("\n演示完成! 🎉")
    print("下一步建议：")
    print("- 使用更多姿态/光照的真实照片扩充注册集")
    print("- 将示例代码整合到自动考勤或门禁业务流程中")
    print("- 搭配活体检测、访问控制等逻辑提升安全性")


if __name__ == "__main__":
    main()