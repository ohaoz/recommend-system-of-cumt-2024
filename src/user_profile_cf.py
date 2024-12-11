import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

class UserProfileCF:
    def __init__(self, n_components=20, min_ratings=5, min_similarity=0.01):
        self.n_components = n_components
        self.min_ratings = min_ratings
        self.min_similarity = min_similarity
        self.user_profiles = None
        self.item_profiles = None
        self.user_features = None
        self.item_features = None
        self.mean_rating = None
        self.user_means = None
        self.item_means = None
        self.rating_std = None
        
    def fit(self, rating_matrix):
        """训练模型
        
        Args:
            rating_matrix: 用户-物品评分矩阵 (scipy.sparse.csr_matrix)
        """
        # 计算评分统计信息
        self.mean_rating = rating_matrix.data.mean()
        self.rating_std = rating_matrix.data.std()
        
        # 计算用户和物品的平均评分
        user_ratings_sum = rating_matrix.sum(axis=1).A.ravel()
        user_counts = rating_matrix.getnnz(axis=1)
        self.user_means = np.zeros_like(user_ratings_sum)
        mask = user_counts >= self.min_ratings
        self.user_means[mask] = user_ratings_sum[mask] / user_counts[mask]
        self.user_means[~mask] = self.mean_rating
        
        item_ratings_sum = rating_matrix.sum(axis=0).A.ravel()
        item_counts = rating_matrix.getnnz(axis=0)
        self.item_means = np.zeros_like(item_ratings_sum)
        mask = item_counts >= self.min_ratings
        self.item_means[mask] = item_ratings_sum[mask] / item_counts[mask]
        self.item_means[~mask] = self.mean_rating
        
        # 提取用户特征
        self.user_features = self._extract_features(rating_matrix)
        
        # 提取物品特征
        self.item_features = self._extract_features(rating_matrix.T)
        
        # 构建用户画像
        self.user_profiles = self._build_profiles(rating_matrix, self.user_features)
        
        # 构建物品画像
        self.item_profiles = self._build_profiles(rating_matrix.T, self.item_features)
        
    def _extract_features(self, matrix):
        """提取特征
        
        Args:
            matrix: 评分矩阵
            
        Returns:
            特征矩阵
        """
        # 标准化
        matrix_dense = matrix.toarray()
        
        # 处理评分数据
        row_means = matrix.mean(axis=1).A.ravel()
        row_means[np.isnan(row_means)] = self.mean_rating
        
        # 中心化和标准化
        for i in range(matrix_dense.shape[0]):
            mask = matrix_dense[i] != 0
            if np.sum(mask) > 0:
                matrix_dense[i, mask] = (matrix_dense[i, mask] - row_means[i]) / self.rating_std
        
        # PCA降维
        pca = PCA(n_components=min(self.n_components, matrix_dense.shape[1]))
        features = pca.fit_transform(matrix_dense)
        
        # 标准化特征
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
        
    def _build_profiles(self, matrix, features):
        """构建画像
        
        Args:
            matrix: 评分矩阵
            features: 特征矩阵
            
        Returns:
            画像矩阵
        """
        # 结合评分行为和特征
        n_features = features.shape[1]
        profiles = np.zeros((matrix.shape[0], n_features * 2))
        
        # 评分行为特征
        rating_features = matrix.toarray()
        
        # 处理评分数据
        row_means = matrix.mean(axis=1).A.ravel()
        row_means[np.isnan(row_means)] = self.mean_rating
        
        # 中心化和标准化
        for i in range(rating_features.shape[0]):
            mask = rating_features[i] != 0
            if np.sum(mask) > 0:
                rating_features[i, mask] = (rating_features[i, mask] - row_means[i]) / self.rating_std
        
        # 使用PCA降维评分特征
        pca = PCA(n_components=n_features)
        rating_features_pca = pca.fit_transform(rating_features)
        
        # 标准化PCA特征
        scaler = StandardScaler()
        rating_features_scaled = scaler.fit_transform(rating_features_pca)
        
        # 组合特征
        profiles[:, :n_features] = features
        profiles[:, n_features:] = rating_features_scaled
        
        return profiles
        
    def predict(self, user_id, item_id):
        """预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测的评分
        """
        if user_id >= len(self.user_profiles) or item_id >= len(self.item_profiles):
            return self.mean_rating
            
        # 获取用户和物品的画像
        user_profile = self.user_profiles[user_id].reshape(1, -1)
        item_profile = self.item_profiles[item_id].reshape(1, -1)
        
        # 计算相似度
        sim = cosine_similarity(user_profile, item_profile)[0][0]
        
        # 如果相似度太低，返回平均评分
        if abs(sim) < self.min_similarity:
            return self.mean_rating
            
        # 考虑用户和物品的评分偏差
        user_bias = self.user_means[user_id] - self.mean_rating
        item_bias = self.item_means[item_id] - self.mean_rating
        
        # 结合相似度和偏差进行预测
        pred = self.mean_rating + user_bias + item_bias + sim * self.rating_std
        
        # 确保预测值在合理范围内
        pred = np.clip(pred, 1, 5)
        
        return float(pred) 