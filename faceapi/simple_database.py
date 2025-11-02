"""
简单人脸数据库管理模块
基于内存存储的人脸特征数据库，支持基础的注册、搜索和管理功能
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleFaceDatabase:
    """简单人脸数据库类 - 基于内存存储"""

    def __init__(self, auto_save: bool = True, save_path: Optional[str] = None):
        """
        初始化人脸数据库

        Args:
            auto_save: 是否自动保存到文件
            save_path: 保存路径，如果为None则使用默认路径
        """
        self.faces: Dict[str, np.ndarray] = {}  # {name: feature_vector}
        self.metadata: Dict[str, Dict] = {}     # {name: metadata}
        self.auto_save = auto_save
        self.save_path = save_path or "data/faces_database.pkl"

        # 确保保存目录存在
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        # 尝试加载已有数据
        self.load_database()

        logger.info(f"人脸数据库初始化完成，当前存储 {len(self.faces)} 个人脸")

    def register_face(self, name: str, feature: np.ndarray,
                     metadata: Optional[Dict] = None, overwrite: bool = False) -> bool:
        """
        注册人脸到数据库

        Args:
            name: 人脸标识名称
            feature: 128维特征向量
            metadata: 额外元数据
            overwrite: 是否覆盖已存在的人脸

        Returns:
            注册是否成功
        """
        try:
            # 检查是否已存在
            if name in self.faces and not overwrite:
                logger.warning(f"人脸 '{name}' 已存在，使用 overwrite=True 来覆盖")
                return False

            # 验证特征向量
            if not isinstance(feature, np.ndarray) or feature.ndim != 1:
                raise ValueError("特征向量必须是1维numpy数组")

            # 存储特征和元数据
            self.faces[name] = feature.copy()

            # 创建元数据
            if metadata is None:
                metadata = {}

            metadata.update({
                'registered_at': datetime.now().isoformat(),
                'feature_dimension': feature.shape[0],
                'feature_norm': float(np.linalg.norm(feature))
            })

            self.metadata[name] = metadata

            # 自动保存
            if self.auto_save:
                self.save_database()

            logger.info(f"人脸 '{name}' 注册成功")
            return True

        except Exception as e:
            logger.error(f"人脸注册失败: {e}")
            return False

    def remove_face(self, name: str) -> bool:
        """
        从数据库中删除人脸

        Args:
            name: 要删除的人脸名称

        Returns:
            删除是否成功
        """
        try:
            if name not in self.faces:
                logger.warning(f"人脸 '{name}' 不存在")
                return False

            del self.faces[name]
            del self.metadata[name]

            if self.auto_save:
                self.save_database()

            logger.info(f"人脸 '{name}' 删除成功")
            return True

        except Exception as e:
            logger.error(f"人脸删除失败: {e}")
            return False

    def get_face(self, name: str) -> Optional[np.ndarray]:
        """
        获取指定名称的人脸特征

        Args:
            name: 人脸名称

        Returns:
            特征向量，如果不存在则返回None
        """
        return self.faces.get(name)

    def get_metadata(self, name: str) -> Optional[Dict]:
        """
        获取人脸元数据

        Args:
            name: 人脸名称

        Returns:
            元数据字典
        """
        return self.metadata.get(name)

    def get_all_faces(self) -> Dict[str, np.ndarray]:
        """
        获取所有人脸特征

        Returns:
            所有人脸特征字典
        """
        return self.faces.copy()

    def get_all_names(self) -> List[str]:
        """
        获取所有人脸名称

        Returns:
            人脸名称列表
        """
        return list(self.faces.keys())

    def find_best_match(self, query_feature: np.ndarray,
                       threshold: float = 0.6, method: str = 'cosine') -> Optional[Tuple[str, float]]:
        """
        在数据库中查找最佳匹配

        Args:
            query_feature: 查询特征向量
            threshold: 相似度阈值
            method: 相似度计算方法，'cosine'或'euclidean'

        Returns:
            (最佳匹配名称, 相似度) 如果没有匹配则返回None
        """
        try:
            if not self.faces:
                logger.warning("数据库为空")
                return None

            best_match = None
            best_score = -1 if method == 'cosine' else float('inf')

            for name, stored_feature in self.faces.items():
                if method == 'cosine':
                    # 余弦相似度 (越大越好)
                    similarity = self._cosine_similarity(query_feature, stored_feature)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = name

                elif method == 'euclidean':
                    # 欧氏距离 (越小越好)
                    distance = self._euclidean_distance(query_feature, stored_feature)
                    if distance < best_score:
                        best_score = distance
                        best_match = name
                else:
                    raise ValueError(f"不支持的相似度方法: {method}")

            # 检查是否达到阈值
            if method == 'cosine':
                if best_score >= threshold:
                    return best_match, best_score
            else:  # euclidean
                if best_score <= threshold:
                    return best_match, 1 - (best_score / 2.0)  # 转换为相似度

            return None

        except Exception as e:
            logger.error(f"查找匹配失败: {e}")
            return None

    def find_all_matches(self, query_feature: np.ndarray,
                        threshold: float = 0.6, method: str = 'cosine',
                        max_results: int = 10) -> List[Tuple[str, float]]:
        """
        查找所有匹配的人脸

        Args:
            query_feature: 查询特征向量
            threshold: 相似度阈值
            method: 相似度计算方法
            max_results: 最大返回结果数

        Returns:
            匹配结果列表 [(名称, 相似度), ...] 按相似度排序
        """
        try:
            matches = []

            for name, stored_feature in self.faces.items():
                if method == 'cosine':
                    similarity = self._cosine_similarity(query_feature, stored_feature)
                    if similarity >= threshold:
                        matches.append((name, similarity))
                elif method == 'euclidean':
                    distance = self._euclidean_distance(query_feature, stored_feature)
                    if distance <= threshold:
                        similarity = max(0, 1 - distance / 2.0)
                        matches.append((name, similarity))

            # 按相似度排序
            matches.sort(key=lambda x: x[1], reverse=True)

            return matches[:max_results]

        except Exception as e:
            logger.error(f"查找所有匹配失败: {e}")
            return []

    def update_metadata(self, name: str, new_metadata: Dict) -> bool:
        """
        更新人脸元数据

        Args:
            name: 人脸名称
            new_metadata: 新的元数据

        Returns:
            更新是否成功
        """
        try:
            if name not in self.faces:
                logger.warning(f"人脸 '{name}' 不存在")
                return False

            self.metadata[name].update(new_metadata)
            self.metadata[name]['updated_at'] = datetime.now().isoformat()

            if self.auto_save:
                self.save_database()

            return True

        except Exception as e:
            logger.error(f"更新元数据失败: {e}")
            return False

    def save_database(self, path: Optional[str] = None) -> bool:
        """
        保存数据库到文件

        Args:
            path: 保存路径，如果为None则使用默认路径

        Returns:
            保存是否成功
        """
        try:
            save_path = path or self.save_path

            data = {
                'faces': {name: feature.tolist() for name, feature in self.faces.items()},
                'metadata': self.metadata,
                'version': '1.0',
                'saved_at': datetime.now().isoformat()
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"数据库已保存到: {save_path}")
            return True

        except Exception as e:
            logger.error(f"数据库保存失败: {e}")
            return False

    def load_database(self, path: Optional[str] = None) -> bool:
        """
        从文件加载数据库

        Args:
            path: 加载路径，如果为None则使用默认路径

        Returns:
            加载是否成功
        """
        try:
            load_path = path or self.save_path

            if not Path(load_path).exists():
                logger.info("数据库文件不存在，创建新数据库")
                return True

            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            # 恢复数据
            self.faces = {name: np.array(feature) for name, feature in data['faces'].items()}
            self.metadata = data.get('metadata', {})

            logger.info(f"数据库已从 {load_path} 加载，共 {len(self.faces)} 个人脸")
            return True

        except Exception as e:
            logger.error(f"数据库加载失败: {e}")
            return False

    def export_to_json(self, path: str) -> bool:
        """
        导出数据库为JSON格式

        Args:
            path: 导出路径

        Returns:
            导出是否成功
        """
        try:
            export_data = {
                'faces': {name: feature.tolist() for name, feature in self.faces.items()},
                'metadata': self.metadata,
                'export_time': datetime.now().isoformat()
            }

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"数据库已导出到: {path}")
            return True

        except Exception as e:
            logger.error(f"数据库导出失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'total_faces': len(self.faces),
            'names': list(self.faces.keys()),
            'feature_dimensions': set(),
            'database_size_mb': 0,
            'oldest_registration': None,
            'newest_registration': None
        }

        if self.faces:
            # 特征维度统计
            for feature in self.faces.values():
                stats['feature_dimensions'].add(feature.shape[0])

            # 注册时间统计
            registration_times = []
            for metadata in self.metadata.values():
                if 'registered_at' in metadata:
                    registration_times.append(metadata['registered_at'])

            if registration_times:
                registration_times.sort()
                stats['oldest_registration'] = registration_times[0]
                stats['newest_registration'] = registration_times[-1]

            # 数据库大小估算
            total_size = sum(feature.nbytes for feature in self.faces.values())
            stats['database_size_mb'] = total_size / (1024 * 1024)

        return stats

    @staticmethod
    def _cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算余弦相似度"""
        feature1_norm = feature1 / np.linalg.norm(feature1)
        feature2_norm = feature2 / np.linalg.norm(feature2)
        return float(np.dot(feature1_norm, feature2_norm))

    @staticmethod
    def _euclidean_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算欧氏距离"""
        return float(np.linalg.norm(feature1 - feature2))

    def clear_database(self) -> bool:
        """
        清空数据库

        Returns:
            清空是否成功
        """
        try:
            self.faces.clear()
            self.metadata.clear()

            if self.auto_save:
                self.save_database()

            logger.info("数据库已清空")
            return True

        except Exception as e:
            logger.error(f"数据库清空失败: {e}")
            return False

    def __len__(self) -> int:
        """返回数据库中的人脸数量"""
        return len(self.faces)

    def __contains__(self, name: str) -> bool:
        """检查人脸是否存在于数据库中"""
        return name in self.faces

    def __str__(self) -> str:
        """字符串表示"""
        return f"SimpleFaceDatabase(faces={len(self.faces)})"


if __name__ == "__main__":
    # 简单测试
    print("人脸数据库模块测试")

    try:
        # 创建数据库
        db = SimpleFaceDatabase(auto_save=False, save_path="test_faces.pkl")

        # 测试注册
        test_feature = np.random.rand(128)
        test_feature = test_feature / np.linalg.norm(test_feature)  # 归一化

        success = db.register_face("test_person", test_feature,
                                  metadata={"description": "测试人脸"})
        print(f"注册测试: {success}")

        # 测试查找
        best_match = db.find_best_match(test_feature, threshold=0.5)
        print(f"查找测试: {best_match}")

        # 测试统计
        stats = db.get_statistics()
        print(f"统计信息: {stats}")

        print("✅ 数据库模块测试通过")

    except Exception as e:
        print(f"❌ 数据库模块测试失败: {e}")