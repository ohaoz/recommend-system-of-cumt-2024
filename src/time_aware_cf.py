import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time

class TimeAwareCF:
    def __init__(self, time_decay_factor=0.1, n_neighbors=20, batch_size=1000, min_similarity=0.01):
        self.time_decay_factor = time_decay_factor
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.min_similarity = min_similarity
        self.user_similarity = None
        self.item_similarity = None
        self.rating_matrix = None
        self.time_matrix = None
        self.mean_rating = None
        self.user_means = None
        self.item_means = None
        self.current_time = time.time()
        
    def fit(self, rating_matrix, timestamp_matrix=None):
        """训练模型
        
        Args:
            rating_matrix: 用户-物品评分矩阵
            timestamp_matrix: 用户-物品评分时间矩阵 (可选)
        """
        self.rating_matrix = rating_matrix
        self.mean_rating = rating_matrix.data.mean()
        
        # 计算用户和物品的平均评分
        user_ratings_sum = rating_matrix.sum(axis=1).A.ravel()
        user_counts = rating_matrix.getnnz(axis=1)
        self.user_means = np.zeros_like(user_ratings_sum)
        mask = user_counts > 0
        self.user_means[mask] = user_ratings_sum[mask] / user_counts[mask]
        self.user_means[~mask] = self.mean_rating
        
        item_ratings_sum = rating_matrix.sum(axis=0).A.ravel()
        item_counts = rating_matrix.getnnz(axis=0)
        self.item_means = np.zeros_like(item_ratings_sum)
        mask = item_counts > 0
        self.item_means[mask] = item_ratings_sum[mask] / item_counts[mask]
        self.item_means[~mask] = self.mean_rating
        
        # 如果没有时间信息，使用当前时间
        if timestamp_matrix is None:
            self.time_matrix = csr_matrix(rating_matrix.shape)
            self.time_matrix.data = np.full_like(rating_matrix.data, self.current_time)
        else:
            self.time_matrix = timestamp_matrix
            
        print("计算时间权重...")
        time_weights = self._calculate_time_weights()
        
        # 计算加权评分矩阵
        print("计算加权评分矩阵...")
        weighted_matrix = self.rating_matrix.copy()
        weighted_matrix.data = weighted_matrix.data * time_weights
        
        # 分批计算相似度
        print("计算用户相似度...")
        self.user_similarity = self._calculate_similarity_batch(weighted_matrix)
        
        print("计算物品相似度...")
        self.item_similarity = self._calculate_similarity_batch(weighted_matrix.T)
        
    def _calculate_time_weights(self):
        """计算时间权重
        
        Returns:
            时间权重数组
        """
        time_diff = self.current_time - self.time_matrix.data
        weights = np.exp(-self.time_decay_factor * time_diff / (24 * 3600))  # 转换为天
        return weights
        
    def _calculate_similarity_batch(self, matrix):
        """分批计算相似度矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            相似度矩阵
        """
        n_rows = matrix.shape[0]
        similarity = np.zeros((n_rows, n_rows))
        
        # 标准化
        print("正在标准化矩阵...")
        matrix_norm = matrix.copy()
        row_means = matrix.mean(axis=1).A.ravel()
        row_means[np.isnan(row_means)] = 0
        
        # 分批处理标准化
        for start in range(0, n_rows, self.batch_size):
            end = min(start + self.batch_size, n_rows)
            print(f"标准化进度: {start}/{n_rows}")
            
            for i in range(start, end):
                if matrix[i].nnz > 0:
                    row = matrix[i].toarray().ravel()
                    nonzero_indices = row.nonzero()[0]
                    row[nonzero_indices] -= row_means[i]
                    matrix_norm[i] = csr_matrix(row)
        
        # 分批计算相似度
        for i in range(0, n_rows, self.batch_size):
            end_i = min(i + self.batch_size, n_rows)
            batch_i = matrix_norm[i:end_i].toarray()
            
            print(f"计算相似度进度: {i}/{n_rows}")
            
            for j in range(0, n_rows, self.batch_size):
                end_j = min(j + self.batch_size, n_rows)
                batch_j = matrix_norm[j:end_j].toarray()
                
                # 计算批次间的相似度
                norms_i = np.sqrt(np.sum(batch_i * batch_i, axis=1))
                norms_j = np.sqrt(np.sum(batch_j * batch_j, axis=1))
                
                # 避免除零
                norms_i[norms_i == 0] = 1
                norms_j[norms_j == 0] = 1
                
                # 计算余弦相似度
                sim_batch = np.dot(batch_i, batch_j.T) / np.outer(norms_i, norms_j)
                
                # 处理数值问题
                sim_batch = np.nan_to_num(sim_batch, 0)
                sim_batch = np.clip(sim_batch, -1, 1)
                
                # 过滤低相似度
                sim_batch[sim_batch < self.min_similarity] = 0
                
                similarity[i:end_i, j:end_j] = sim_batch
                
                # 清理内存
                del sim_batch
        
        return similarity
        
    def predict(self, user_id, item_id):
        """预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测的评分
        """
        if user_id >= self.rating_matrix.shape[0] or item_id >= self.rating_matrix.shape[1]:
            return self.mean_rating
            
        # 获取用户的评分记录
        user_ratings = self.rating_matrix[user_id].toarray().ravel()
        user_times = self.time_matrix[user_id].toarray().ravel()
        
        # 获取物品的评分记录
        item_ratings = self.rating_matrix[:, item_id].toarray().ravel()
        item_times = self.time_matrix[:, item_id].toarray().ravel()
        
        # 基于用户的预测
        user_pred = self.mean_rating
        user_mask = user_ratings > 0
        if np.sum(user_mask) > 0:
            # 获取相似用户的评分
            user_sims = self.user_similarity[user_id]
            # 选择最相似的N个用户
            top_similar_users = np.argsort(-user_sims)[:self.n_neighbors]
            # 过滤掉未评分的用户
            valid_users = [u for u in top_similar_users if item_ratings[u] > 0]
            
            if len(valid_users) > 0:
                similar_ratings = item_ratings[valid_users]
                similar_times = item_times[valid_users]
                similar_sims = user_sims[valid_users]
                
                # 计算时间权重
                time_weights = np.exp(-self.time_decay_factor * 
                                    (self.current_time - similar_times) / (24 * 3600))
                
                # 结合相似度和时间权重
                weights = similar_sims * time_weights
                if weights.sum() > 0:
                    # 考虑评分偏差和时间衰减
                    rating_biases = similar_ratings - self.user_means[valid_users]
                    time_decay = np.exp(-self.time_decay_factor * 
                                      (self.current_time - np.mean(similar_times)) / (24 * 3600))
                    user_pred = self.user_means[user_id] + time_decay * np.average(rating_biases, weights=weights)
        
        # 基于物品的预测
        item_pred = self.mean_rating
        item_mask = item_ratings > 0
        if np.sum(item_mask) > 0:
            # 获取用户评分过的物品
            rated_items = np.nonzero(user_ratings)[0]
            if len(rated_items) > 0:
                # 获取物品相似度
                item_sims = self.item_similarity[item_id, rated_items]
                # 选择最相似的N个物品
                top_similar_items = rated_items[np.argsort(-item_sims)[:self.n_neighbors]]
                
                if len(top_similar_items) > 0:
                    similar_ratings = user_ratings[top_similar_items]
                    similar_times = user_times[top_similar_items]
                    similar_sims = item_sims[np.argsort(-item_sims)[:self.n_neighbors]]
                    
                    # 计算时间权重
                    time_weights = np.exp(-self.time_decay_factor * 
                                        (self.current_time - similar_times) / (24 * 3600))
                    
                    # 结合相似度和时间权重
                    weights = similar_sims * time_weights
                    if weights.sum() > 0:
                        # 考虑评分���差和时间衰减
                        rating_biases = similar_ratings - self.item_means[top_similar_items]
                        time_decay = np.exp(-self.time_decay_factor * 
                                          (self.current_time - np.mean(similar_times)) / (24 * 3600))
                        item_pred = self.item_means[item_id] + time_decay * np.average(rating_biases, weights=weights)
        
        # 组合预测结果
        if user_pred != self.mean_rating and item_pred != self.mean_rating:
            # 如果两种预测都有效，根据可靠性加权
            user_confidence = len([u for u in valid_users if user_sims[u] >= self.min_similarity])
            item_confidence = len([i for i in top_similar_items if item_sims[i] >= self.min_similarity])
            
            total_confidence = user_confidence + item_confidence
            if total_confidence > 0:
                user_weight = user_confidence / total_confidence
                item_weight = item_confidence / total_confidence
                pred = user_pred * user_weight + item_pred * item_weight
            else:
                pred = self.mean_rating
        elif user_pred != self.mean_rating:
            pred = user_pred
        else:
            pred = item_pred
            
        # 确保预测值在合理范围内
        pred = np.clip(pred, 1, 5)
        
        # 添加置信度阈值
        confidence = max(user_confidence if user_pred != self.mean_rating else 0,
                        item_confidence if item_pred != self.mean_rating else 0)
        
        if confidence < self.n_neighbors / 4:  # 如果置信度太低
            pred = self.mean_rating  # 返回平均评分
        
        return float(pred) 