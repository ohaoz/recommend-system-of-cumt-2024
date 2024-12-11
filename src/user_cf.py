import numpy as np 
from scipy.sparse import csr_matrix
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

class UserCF:
    def __init__(self, n_neighbors=20, min_support=5):
        self.n_neighbors = n_neighbors
        self.min_support = min_support
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        
    def fit(self, train_matrix):
        self.train_matrix = train_matrix
        self.n_users, self.n_items = train_matrix.shape
        
        # 计算全局平均分和用户/物品偏置
        self._compute_baselines()
        
        # 计算归一化矩阵
        normalized_matrix = self._normalize_matrix()
        
        # 计算用户相似度
        self.user_similarity = self._compute_similarity(normalized_matrix)

    def _compute_baselines(self):
        # 计算全局平均分
        self.global_mean = self.train_matrix.data.mean()
        
        # 计算用户偏置
        self.user_means = np.zeros(self.n_users)
        for u in range(self.n_users):
            user_ratings = self.train_matrix[u].data
            if len(user_ratings) > 0:
                self.user_means[u] = user_ratings.mean() - self.global_mean
            
        # 计算物品偏置
        self.item_means = np.zeros(self.n_items)
        for i in range(self.n_items):
            item_ratings = self.train_matrix[:,i].data
            if len(item_ratings) > 0:
                self.item_means[i] = item_ratings.mean() - self.global_mean

    def _normalize_matrix(self):
        normalized = self.train_matrix.copy()
        rows, cols = normalized.nonzero()
        
        # 减去基准预测值
        normalized.data = normalized.data - (self.global_mean + 
                                          self.user_means[rows] + 
                                          self.item_means[cols])
        return normalized

    def _compute_similarity(self, normalized_matrix):
        """计算用户相似度矩阵"""
        # 使用余弦相似度直接计算
        similarity = cosine_similarity(normalized_matrix)
        
        # 将对角线（自相似度）设为0
        np.fill_diagonal(similarity, 0)
        
        return similarity
        
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        # 如果用户已经对该物品评分，直接返回已有评分
        if self.train_matrix[user_id, item_id] != 0:
            return self.train_matrix[user_id, item_id]
        
        # 获取评分过该物品的用户列表
        rated_users = self.train_matrix[:, item_id].nonzero()[0]
        
        # 如果没有用户评价过该物品，返回全局平均分
        if len(rated_users) == 0:
            return np.clip(self.global_mean, 1, 5)
        
        # 计算与这些用户的相似度
        similarities = self.user_similarity[user_id, rated_users]
        
        # 选择相似度最高的N个用户
        top_n_idx = np.argsort(similarities)[-self.n_neighbors:]
        similar_users = rated_users[top_n_idx]
        sim_weights = similarities[top_n_idx]
        
        # 过滤掉相似度为0或负数的用户
        mask = sim_weights > 0
        similar_users = similar_users[mask]
        sim_weights = sim_weights[mask]
        
        # 如果没有足够的相似用户，返回物品平均分
        if len(similar_users) < self.min_support:
            item_mean = np.mean(self.train_matrix[:, item_id].data)
            return np.clip(item_mean, 1, 5)
        
        # 获取相似用户对该物品的评分
        ratings = np.array([self.train_matrix[u, item_id] for u in similar_users])
        
        # 计算加权预测评分
        pred = np.sum(sim_weights * ratings) / np.sum(sim_weights)
        
        return np.clip(pred, 1, 5)
