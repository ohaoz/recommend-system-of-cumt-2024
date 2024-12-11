import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self, n_neighbors=10, batch_size=1000):
        """初始化ItemCF推荐器
        
        Args:
            n_neighbors (int): 近邻物品数量
            batch_size (int): 批处理大小
        """
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.user_item_matrix = None
        self.top_similar_items = {}
        self.top_similarities = {}
        
    def fit(self, user_item_matrix):
        """训练模型
        
        Args:
            user_item_matrix (csr_matrix): 用户-物品评分矩阵
        """
        self.user_item_matrix = user_item_matrix
        n_items = user_item_matrix.shape[1]
        
        # 分批计算物品相似度
        for i in range(0, n_items, self.batch_size):
            batch_items = range(i, min(i + self.batch_size, n_items))
            batch_matrix = user_item_matrix[:, batch_items].T
            
            # 计算这批物品与所有物品的相似度
            similarities = cosine_similarity(batch_matrix, user_item_matrix.T)
            
            # 为每个物品存储top-N个最相似的物品
            for idx, item_id in enumerate(batch_items):
                item_similarities = similarities[idx]
                item_similarities[item_id] = 0  # 将自己的相似度置为0
                
                # 获取top-N个最相似物品
                top_indices = np.argsort(item_similarities)[:-self.n_neighbors-1:-1]
                top_sims = item_similarities[top_indices]
                
                # 存储结果
                self.top_similar_items[item_id] = top_indices
                self.top_similarities[item_id] = top_sims
        
    def predict(self, user_id, item_id):
        """预测用户对物品的评分
        
        Args:
            user_id (int): 用户ID
            item_id (int): 物品ID
            
        Returns:
            float: 预测的评分
        """
        if self.user_item_matrix is None:
            raise ValueError("请先调用fit方法训练模型")
            
        # 获取物品的top-N个相似物品
        if item_id not in self.top_similar_items:
            return 3.0  # 如果是未知物品，返回默认评分
            
        similar_items = self.top_similar_items[item_id]
        similarities = self.top_similarities[item_id]
        
        # 获取用户对这些物品的评分
        user_ratings = self.user_item_matrix[user_id, similar_items].toarray().flatten()
        
        # 如果用户没有对任何相似物品评分，返回默认评分
        if np.sum(similarities) == 0 or np.sum(user_ratings) == 0:
            return 3.0
            
        # 计算加权评分
        predicted_rating = np.sum(similarities * user_ratings) / np.sum(similarities)
        
        return predicted_rating
        
    def predict_for_user(self, user_id, item_ids):
        """为用户预测多个物品的评分
        
        Args:
            user_id (int): 用户ID
            item_ids (list): 物品ID列表
            
        Returns:
            dict: 物品ID到预测评分的映射
        """
        predictions = {}
        for item_id in item_ids:
            predictions[item_id] = self.predict(user_id, item_id)
        return predictions 