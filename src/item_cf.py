import numpy as np
from scipy.sparse import csr_matrix
import warnings
import time
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

class ItemCF:
    def __init__(self, n_neighbors=10, similarity_metric='cosine'):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_means = None
        self.top_similar_items = {}
        self.top_similarities = {}
    
    def _normalize_ratings(self, ratings):
        """对物品评分进行归一化处理"""
        print("计算物品平均评分...")
        ratings_sum = np.array(ratings.sum(axis=0)).ravel()
        ratings_count = np.array(ratings.getnnz(axis=0))
        
        self.item_means = np.zeros(ratings.shape[1])
        mask = ratings_count > 0
        self.item_means[mask] = ratings_sum[mask] / ratings_count[mask]
        
        print("创建归一化矩阵...")
        normalized = ratings.copy()
        rows, cols = normalized.nonzero()
        normalized.data = normalized.data - self.item_means[cols]
        
        print("归一化完成!")
        return normalized

    def _compute_similarity(self, normalized_matrix, batch_size=1024):
        """计算物品相似度"""
        n_items = normalized_matrix.shape[1]
        
        print("\n开始计算物品相似度...")
        # 计算IDF权重
        n_users_per_item = np.array((normalized_matrix != 0).sum(axis=0)).ravel()
        idf = np.log(normalized_matrix.shape[0] / (n_users_per_item + 1))
        
        # 应用IDF权重到评分矩阵，并确保结果是CSR格式
        weighted_matrix = normalized_matrix.copy()
        weighted_matrix = weighted_matrix.multiply(idf.reshape(1, -1)).tocsr()
        
        # 计算物品评分数量
        item_ratings_count = np.array(weighted_matrix.getnnz(axis=0))
        
        # 转置矩阵以计算物品相似度
        item_matrix = weighted_matrix.T.tocsr()
        
        # 减小批处理大小以降低内存使用
        batch_size = min(batch_size, 512)
        
        # 分批处理以节省内存
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            if start == 0:
                print(f"处理物品批次 {start+1}-{end}/{n_items}")
            
            # 获取当前批次数据
            batch_matrix = item_matrix[start:end].toarray()
            batch_counts = item_ratings_count[start:end]
            
            # 计算均值
            batch_means = np.mean(batch_matrix, axis=1, keepdims=True)
            batch_matrix = batch_matrix - batch_means
            
            # 分批计算相似度以节省内存
            similarities = np.zeros((end - start, n_items))
            for j in range(0, n_items, batch_size):
                j_end = min(j + batch_size, n_items)
                
                # 获取另一批次数据
                other_matrix = item_matrix[j:j_end].toarray()
                other_means = np.mean(other_matrix, axis=1, keepdims=True)
                other_matrix = other_matrix - other_means
                
                # 计算调整余弦相似度
                numerator = np.dot(batch_matrix, other_matrix.T)
                denominator = np.outer(
                    np.sqrt(np.sum(batch_matrix**2, axis=1)),
                    np.sqrt(np.sum(other_matrix**2, axis=1))
                )
                
                # 避免除零
                mask = denominator > 1e-8
                batch_similarities = np.zeros_like(numerator)
                batch_similarities[mask] = numerator[mask] / denominator[mask]
                
                # 计算评分数量惩罚项
                batch_penalty = np.sqrt(
                    np.minimum(batch_counts.reshape(-1, 1), item_ratings_count[j:j_end].reshape(1, -1)) /
                    np.maximum(batch_counts.reshape(-1, 1), item_ratings_count[j:j_end].reshape(1, -1))
                )
                
                # 应用惩罚项
                batch_similarities *= batch_penalty
                similarities[:, j:j_end] = batch_similarities
            
            # 为每个物品找到最相似的邻居
            for i in range(end - start):
                item_id = start + i
                item_sims = similarities[i]
                
                # 排除自身
                item_sims[item_id] = -np.inf
                
                # 获取top-k个最相似物品
                top_indices = np.argpartition(item_sims, -self.n_neighbors)[-self.n_neighbors:]
                top_indices = top_indices[np.argsort(-item_sims[top_indices])]
                
                self.top_similar_items[item_id] = top_indices
                self.top_similarities[item_id] = item_sims[top_indices]
            
            if (start // batch_size + 1) % 10 == 0:
                print(f"已完成 {end}/{n_items} 个物品的相似度计算...")

    def fit(self, user_item_matrix):
        """训练模型"""
        print(f"开始处理用户-物品矩阵 (shape: {user_item_matrix.shape})...")
        print(f"非零元素数量: {user_item_matrix.nnz}")
        print(f"矩阵稀疏度: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4%}")
        
        print("\n正在归一化物品评分...")
        self.user_item_matrix = user_item_matrix
        normalized_matrix = self._normalize_ratings(user_item_matrix)
        
        print("\n正在计算物品相似度...")
        start_time = time.time()
        self._compute_similarity(normalized_matrix)
        end_time = time.time()
        print(f"\n相似度计算完成! 总耗时: {(end_time-start_time)/60:.2f} 分钟")

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if item_id not in self.top_similar_items:
            return self.item_means[item_id] if self.item_means is not None else 3.0

        # 获取相似物品及其相似度
        similar_items = np.array(self.top_similar_items[item_id])
        similarities = np.array(self.top_similarities[item_id])
        
        # 获取用户对相似物品的评分
        user_ratings = self.user_item_matrix[user_id, similar_items].toarray().ravel()
        
        # 找出用户有评分的物品
        rated_mask = user_ratings > 0
        
        if np.sum(rated_mask) == 0:
            return self.item_means[item_id] if self.item_means is not None else 3.0
        
        # 计算加权平均评分
        relevant_similarities = similarities[rated_mask]
        relevant_ratings = user_ratings[rated_mask]
        
        # 添加置信度权重
        confidence_weights = 1 / (1 + np.exp(-len(relevant_ratings)))
        rating_variance = np.var(relevant_ratings) if len(relevant_ratings) > 1 else 1.0
        confidence_weights *= 1 / (1 + rating_variance)
        
        # 计算预测评分
        weighted_sum = np.sum(relevant_ratings * relevant_similarities * confidence_weights)
        similarity_sum = np.sum(np.abs(relevant_similarities) * confidence_weights)
        
        if similarity_sum > 0:
            base_pred = weighted_sum / similarity_sum
        else:
            base_pred = self.item_means[item_id] if self.item_means is not None else 3.0
        
        # 考虑评分分布和用户偏好
        user_mean = np.mean(relevant_ratings)
        prediction = 0.7 * base_pred + 0.3 * user_mean
        
        # 确保预测值在合理范围内
        return np.clip(prediction, 1, 5)
        
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