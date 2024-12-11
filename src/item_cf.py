import numpy as np
from scipy.sparse import csr_matrix
import warnings
from sklearn.metrics.pairwise import cosine_similarity
import time

warnings.filterwarnings('ignore')

class ItemCF:
    def __init__(self, n_neighbors=20, min_similarity=0.01, verbose=False):
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.verbose = verbose
        self.item_ratings = None
        self.item_similarity = None
        self.mean_ratings = None
        self.precomputed_mean_ratings = None
        
    def fit(self, ratings_matrix):
        if self.verbose:
            print(f"开始处理用户-物品矩阵 (shape: {ratings_matrix.shape})...")
            print(f"非零元素数量: {ratings_matrix.nnz}")
            print(f"矩阵稀疏度: {ratings_matrix.nnz/(ratings_matrix.shape[0]*ratings_matrix.shape[1])*100:.4f}%")
        
        self.item_ratings = ratings_matrix.copy()
        
        # 计算物品平均评分
        if self.verbose:
            print("\n计算物品平均评分...")
        ratings_sum = ratings_matrix.sum(axis=0).A1  # Efficiently handle sparse matrix
        ratings_count = ratings_matrix.getnnz(axis=0)
        self.mean_ratings = np.zeros_like(ratings_sum, dtype=float)
        mask = ratings_count > 0
        self.mean_ratings[mask] = ratings_sum[mask] / ratings_count[mask]
        self.precomputed_mean_ratings = self.mean_ratings.copy()  # Cache mean ratings for quick access
        if self.verbose:
            print("物品平均评分计算完成!")
            
        # 归一化评分矩阵
        if self.verbose:
            print("归一化评分矩阵...")
        normalized_matrix = self._normalize_ratings(ratings_matrix)
        if self.verbose:
            print("评分矩阵归一化完成!")
            
        # 计算物品相似度
        if self.verbose:
            print("\n计算物品相似度...")
        start_time = time.time()
        self.item_similarity = self._compute_similarity(normalized_matrix)
        end_time = time.time()
        if self.verbose:
            print(f"\n相似度计算完成! 总耗时: {(end_time-start_time)/60:.2f} 分钟")
    
    def _normalize_ratings(self, ratings_matrix):
        if self.verbose:
            print("开始归一化评分矩阵...")
        normalized = ratings_matrix.copy()
        rows, cols = normalized.nonzero()
        normalized.data = normalized.data - self.mean_ratings[cols]
        if self.verbose:
            print("评分矩阵归一化完成!")
        return normalized
    
    def _compute_similarity(self, normalized_matrix, min_common_users=3, batch_size=1000):
        if self.verbose:
            print("开始计算物品相似度...")
        n_items = normalized_matrix.shape[1]
        similarity_matrix = csr_matrix((n_items, n_items), dtype=np.float32)
        
        item_matrix = normalized_matrix.T.tocsr()
        
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            if self.verbose:
                print(f"处理物品批次 {start+1}-{end}/{n_items}")
            
            # 获取当前批次的物品向量
            batch_vectors = item_matrix[start:end]
            
            # 计算点积
            dot_product = batch_vectors @ item_matrix.T
            
            # 计算范数
            norms_i = np.sqrt(batch_vectors.multiply(batch_vectors).sum(axis=1)).A1
            norms_j = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1)).A1
            
            # 计算共同评分用户数
            common_users = (batch_vectors != 0).dot((item_matrix != 0).T).toarray()
            
            # 计算相似度
            similarity = np.zeros((batch_vectors.shape[0], n_items), dtype=np.float32)
            norms_outer = np.outer(norms_i, norms_j)
            valid_mask = (norms_outer != 0) & (common_users >= min_common_users)
            
            if valid_mask.any():
                dot_product_dense = dot_product.toarray()
                similarity[valid_mask] = dot_product_dense[valid_mask] / norms_outer[valid_mask]
                # 惩罚共同评分用户数较少的情况
                similarity[valid_mask] *= np.minimum(1, common_users[valid_mask] / (min_common_users * 2))
            
            similarity_matrix[start:end, :] = csr_matrix(similarity)
            if self.verbose and (start + batch_size) % (batch_size * 10) == 0:
                print(f"已完成 {end}/{n_items} 个物品的相似度计算...")
        
        if self.verbose:
            print("物品相似度计算完成!")
        return similarity_matrix
        
    def predict(self, user_id, item_id):
        if self.verbose:
            print(f"开始预测用户 {user_id} 对物品 {item_id} 的评分...")
            print(f"用户评分数: {len(np.nonzero(self.item_ratings[user_id].toarray().flatten())[0])}")
            print(f"物品平均评分: {self.precomputed_mean_ratings[item_id]:.2f}")
        
        if self.item_ratings is None:
            raise ValueError("模型未训练")
        
        # 检查是否已有评分
        actual_rating = self.item_ratings[user_id, item_id]
        if actual_rating != 0:
            if self.verbose:
                print(f"用户 {user_id} 已对物品 {item_id} 评分，实际评分为 {actual_rating}")
            return actual_rating
            
        # 获取用户的评分记录
        user_ratings = self.item_ratings[user_id].toarray().flatten()
        rated_items = np.nonzero(user_ratings)[0]
        
        # 如果用户没有任何评分记录，返回物品的平均评分
        if len(rated_items) == 0:
            if self.verbose:
                print(f"用户 {user_id} 未对任何物品评分，返回物品 {item_id} 的平均评分")
            return self.precomputed_mean_ratings[item_id]
        
        # 获取最相似的物品
        item_similarities = self.item_similarity[item_id].toarray().flatten()
        
        # 过滤掉相似度低于阈值的物品
        valid_items = rated_items[item_similarities[rated_items] >= self.min_similarity]
        if len(valid_items) > 0:
            top_similar_items = valid_items[np.argsort(-item_similarities[valid_items])][:self.n_neighbors]
            if self.verbose:
                print(f"找到 {len(valid_items)} 个相似物品，选择前 {len(top_similar_items)} 个")
                print(f"相似度范围: {item_similarities[top_similar_items].min():.3f} - {item_similarities[top_similar_items].max():.3f}")
        else:
            if self.verbose:
                print(f"没有找到物品 {item_id} 的相似物品（相似度阈值: {self.min_similarity}），返回平均评分")
            return self.precomputed_mean_ratings[item_id]
        
        # 计算预测评分
        similarities = item_similarities[top_similar_items]
        ratings = user_ratings[top_similar_items]
        
        # 使用加权平均计算预测评分
        weights = similarities  # 直接使用相似度作为权重
        if weights.sum() > 0:
            # 考虑评分偏差和用户平均评分
            user_mean = np.mean(user_ratings[rated_items])
            rating_biases = ratings - self.mean_ratings[top_similar_items]
            prediction = user_mean + np.sum(rating_biases * weights) / np.sum(weights)
            prediction = np.clip(prediction, 1, 5)  # 限制预测值在合理范围内
            
            if self.verbose:
                print(f"用户平均评分: {user_mean:.2f}")
                print(f"评分偏差范围: {rating_biases.min():.2f} - {rating_biases.max():.2f}")
                print(f"预测完成! 预测评分为 {prediction:.2f}")
            return prediction
        
        if self.verbose:
            print(f"无法计算有效预测，返回物品平均评分")
        return self.precomputed_mean_ratings[item_id]
