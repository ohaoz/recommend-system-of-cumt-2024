import numpy as np
from scipy.sparse import csr_matrix
import warnings
import time
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

class UserCF:
    def __init__(self, n_neighbors=10, similarity_metric='cosine', use_gpu=False):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_means = None
        self.top_similar_users = {}
        self.top_similarities = {}
    
    def _normalize_ratings(self, ratings):
        """对用户评分进行归一化处理"""
        print("计算用户平均评分...")
        ratings_sum = np.array(ratings.sum(axis=1)).ravel()
        ratings_count = np.array(ratings.getnnz(axis=1))
        
        self.user_means = np.zeros(ratings.shape[0])
        mask = ratings_count > 0
        self.user_means[mask] = ratings_sum[mask] / ratings_count[mask]
        
        print("创建归一化矩阵...")
        normalized = ratings.copy()
        rows, cols = normalized.nonzero()
        normalized.data = normalized.data - self.user_means[rows]
        
        print("归一化完成!")
        return normalized

    def _compute_similarity(self, normalized_matrix, batch_size=100):
        """使用CPU计算用户相似度"""
        n_users = normalized_matrix.shape[0]
        
        print("\n开始计算用户相似度...")
        
        # 计算IDF权重
        n_users_per_item = np.array((normalized_matrix != 0).sum(axis=0)).ravel()
        idf = np.log(n_users / (n_users_per_item + 1))
        
        # 应用IDF权重到评分矩阵
        weighted_matrix = normalized_matrix.copy()
        weighted_matrix = weighted_matrix.multiply(idf.reshape(1, -1)).tocsr()
        
        # 计算用户评分数量
        user_ratings_count = np.array(weighted_matrix.getnnz(axis=1)).reshape(-1, 1)
        
        # 分批处理以节省内存
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            if start == 0:
                print(f"处理用户批次 {start+1}-{end}/{n_users}")
            
            # 获取当前批次数据
            batch_matrix = weighted_matrix[start:end].toarray()
            batch_counts = user_ratings_count[start:end]
            
            # 初始化当前批次的top-k
            for i in range(end - start):
                user_id = start + i
                self.top_similar_users[user_id] = []
                self.top_similarities[user_id] = []
            
            # 分块计算相似度
            for j in range(0, n_users, batch_size):
                j_end = min(j + batch_size, n_users)
                
                # 获取另一批次数据
                other_matrix = weighted_matrix[j:j_end].toarray()
                other_counts = user_ratings_count[j:j_end]
                
                # 计算Pearson相关系数
                mean_batch = np.mean(batch_matrix, axis=1, keepdims=True)
                mean_other = np.mean(other_matrix, axis=1, keepdims=True)
                
                centered_batch = batch_matrix - mean_batch
                centered_other = other_matrix - mean_other
                
                # 计算相关系数
                numerator = np.dot(centered_batch, centered_other.T)
                denominator = np.outer(
                    np.sqrt(np.sum(centered_batch**2, axis=1)),
                    np.sqrt(np.sum(centered_other**2, axis=1))
                )
                
                # 避免除零
                mask = denominator > 1e-8
                similarities = np.zeros_like(numerator)
                similarities[mask] = numerator[mask] / denominator[mask]
                
                # 添加评分数量惩罚项
                rating_penalty = np.minimum(batch_counts, other_counts.T) / np.maximum(batch_counts, other_counts.T)
                similarities *= np.sqrt(rating_penalty)
                
                # 更新每个用户的top-k
                for i in range(end - start):
                    user_id = start + i
                    user_sims = similarities[i]
                    
                    # 排除自身
                    if start + i >= j and start + i < j_end:
                        user_sims[start + i - j] = -np.inf
                    
                    # 找出当前批次中的top-k
                    if len(self.top_similar_users[user_id]) < self.n_neighbors:
                        # 初始填充
                        top_indices = np.argsort(user_sims)[-self.n_neighbors:]
                        for idx in top_indices:
                            if user_sims[idx] > -np.inf:
                                self.top_similar_users[user_id].append(j + idx)
                                self.top_similarities[user_id].append(float(user_sims[idx]))
                    else:
                        # 更新现有top-k
                        current_min = min(self.top_similarities[user_id])
                        better_indices = np.where(user_sims > current_min)[0]
                        for idx in better_indices:
                            if user_sims[idx] > -np.inf:
                                min_idx = np.argmin(self.top_similarities[user_id])
                                self.top_similar_users[user_id][min_idx] = j + idx
                                self.top_similarities[user_id][min_idx] = float(user_sims[idx])
            
            if (start // batch_size + 1) % 10 == 0:
                print(f"已完成 {end}/{n_users} 个用户的相似度计算...")

    def fit(self, user_item_matrix):
        """训练模型"""
        print(f"开始处理用户-物品矩阵 (shape: {user_item_matrix.shape})...")
        print(f"非零元素数量: {user_item_matrix.nnz}")
        print(f"矩阵稀疏度: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4%}")
        
        print("\n正在归一化用户评分...")
        self.user_item_matrix = user_item_matrix
        normalized_matrix = self._normalize_ratings(user_item_matrix)
        
        print("\n正在计算用户相似度...")
        start_time = time.time()
        self._compute_similarity(normalized_matrix)
        end_time = time.time()
        print(f"\n相似度计算完成! 总耗时: {(end_time-start_time)/60:.2f} 分钟")

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id not in self.top_similar_users:
            return self.user_means[user_id] if self.user_means is not None else 3.0

        # 获取相似用户及其相似度
        similar_users = np.array(self.top_similar_users[user_id])
        similarities = np.array(self.top_similarities[user_id])
        
        # 获取相似用户对目标物品的评分
        neighbor_ratings = self.user_item_matrix[similar_users, item_id].toarray().ravel()
        
        # 找出有评分的用户
        rated_mask = neighbor_ratings > 0
        
        if np.sum(rated_mask) == 0:
            return self.user_means[user_id] if self.user_means is not None else 3.0
        
        # 计算加权平均评分
        relevant_similarities = similarities[rated_mask]
        relevant_ratings = neighbor_ratings[rated_mask]
        
        # 添加置信度权重
        confidence_weights = 1 / (1 + np.exp(-len(relevant_ratings)))
        
        # 计算预测评分
        weighted_sum = np.sum(relevant_ratings * relevant_similarities * confidence_weights)
        similarity_sum = np.sum(np.abs(relevant_similarities) * confidence_weights)
        
        if similarity_sum > 0:
            base_pred = weighted_sum / similarity_sum
        else:
            base_pred = self.user_means[user_id] if self.user_means is not None else 3.0
        
        # 考虑评分分布
        user_mean = np.mean(relevant_ratings)
        prediction = 0.7 * base_pred + 0.3 * user_mean
        
        # 确保预测值在合理范围内
        return np.clip(prediction, 1, 5)