import numpy as np
from scipy.sparse import csr_matrix
import warnings
import time
import cupy as cp
import faiss
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

class UserCF:
    def __init__(self, n_neighbors=10, similarity_metric='cosine', use_gpu=True):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.use_gpu = use_gpu
        self.user_item_matrix = None
        self.user_means = None
        self.top_similar_users = {}
        self.top_similarities = {}
        
        # 初始化GPU
        if self.use_gpu:
            self.gpu_device = cp.cuda.Device(0)
            
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

    def _compute_similarity_gpu(self, normalized_matrix, batch_size=1024):
        """使用GPU计算用户相似度"""
        n_users = normalized_matrix.shape[0]
        
        # 将稀疏矩阵转换为密集矩阵并移至GPU
        print("将数据转移到GPU...")
        with self.gpu_device:
            # 分批处理以节省GPU内存
            for start in range(0, n_users, batch_size):
                end = min(start + batch_size, n_users)
                if start == 0:
                    print(f"处理用户批次 {start+1}-{end}/{n_users}")
                
                # 将当前批次数据转换为密集矩阵
                batch_matrix = normalized_matrix[start:end].toarray()
                
                # 计算L2范数
                norms = np.linalg.norm(batch_matrix, axis=1)
                valid_mask = norms > 0
                
                if np.any(valid_mask):
                    # 标准化向量
                    batch_matrix[valid_mask] /= norms[valid_mask, np.newaxis]
                    
                    # 初始化FAISS索引
                    if start == 0:
                        d = normalized_matrix.shape[1]  # 向量维度
                        index = faiss.IndexFlatIP(d)  # 内积相似度
                        if self.use_gpu:
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, 0, index)
                    
                    # 添加向量到索引
                    index.add(batch_matrix.astype('float32'))
                
                if (start // batch_size + 1) % 10 == 0:
                    print(f"已处理 {end}/{n_users} 个用户...")
        
        print("\n开始查找最近邻...")
        # 分批查询最近邻
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            query_matrix = normalized_matrix[start:end].toarray().astype('float32')
            
            # 查询最近邻
            similarities, indices = index.search(query_matrix, self.n_neighbors + 1)
            
            # 保存结果
            for i, (user_sims, user_indices) in enumerate(zip(similarities, indices)):
                user_id = start + i
                # 移除自身（如果在结果中）
                mask = user_indices != user_id
                self.top_similar_users[user_id] = user_indices[mask][:self.n_neighbors]
                self.top_similarities[user_id] = user_sims[mask][:self.n_neighbors]
            
            if (start // batch_size + 1) % 10 == 0:
                print(f"已完成 {end}/{n_users} 个用户的近邻搜索...")

    def fit(self, user_item_matrix):
        """训练模型"""
        print(f"开始处理用户-物品矩阵 (shape: {user_item_matrix.shape})...")
        print(f"非零元素数量: {user_item_matrix.nnz}")
        print(f"矩阵稀疏度: {user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4%}")
        
        print("\n正在归一化用户评分...")
        self.user_item_matrix = user_item_matrix
        normalized_matrix = self._normalize_ratings(user_item_matrix)
        
        print("\n正在使用GPU计算用户相似度...")
        start_time = time.time()
        self._compute_similarity_gpu(normalized_matrix)
        end_time = time.time()
        print(f"\n相似度计算完成! 总耗时: {(end_time-start_time)/60:.2f} 分钟")

    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分
        
        参数:
            user_id (int): 用户ID
            item_id (int): 物品ID
            
        返回:
            float: 预测的评分
        """
        if user_id not in self.top_similar_users:
            return self.user_means[user_id] if self.user_means is not None else 3.0

        neighbors = self.top_similar_users[user_id]
        similarities = self.top_similarities[user_id]
        
        # 获取邻居评分
        neighbor_ratings = self.user_item_matrix[neighbors, item_id].toarray().ravel()
        rated_mask = neighbor_ratings > 0
        
        if np.sum(rated_mask) == 0:
            return self.user_means[user_id] if self.user_means is not None else 3.0
        
        # 计算加权平均评分
        relevant_similarities = similarities[rated_mask]
        relevant_ratings = neighbor_ratings[rated_mask]
        
        # 加回用户均值以还原预测评分
        prediction = (np.dot(relevant_ratings, relevant_similarities) / 
                     np.sum(relevant_similarities))
        prediction += self.user_means[user_id]
        
        # 限制预测评分范围
        return np.clip(prediction, 1, 5)