import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class DataProcessor:
    def __init__(self, ratings_path):
        """初始化数据处理器
        
        Args:
            ratings_path (str): ratings.csv 文件路径
        """
        self.ratings_df = pd.read_csv(ratings_path)
        self.user_item_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        
    def sample_data(self, user_sample_size=10000, min_ratings=5):
        """对数据进行采样，保留活跃用户
        
        Args:
            user_sample_size (int): 采样用户数量
            min_ratings (int): 用户最少评分数
            
        Returns:
            DataFrame: 采样后的数据
        """
        print(f"原始数据规模: {len(self.ratings_df)} 条评分")
        
        # 获取用户评分数统计
        user_ratings = self.ratings_df['userId'].value_counts()
        
        # 筛选评分数达到要求的用户
        qualified_users = user_ratings[user_ratings >= min_ratings].index
        print(f"评分数>={min_ratings}的用户数: {len(qualified_users)}")
        
        # 如果合格用户数量大于采样大小，进行随机采样
        if len(qualified_users) > user_sample_size:
            sampled_users = np.random.choice(qualified_users, size=user_sample_size, replace=False)
        else:
            sampled_users = qualified_users
            
        # 获取采样用户的评分数据
        sampled_data = self.ratings_df[self.ratings_df['userId'].isin(sampled_users)]
        
        # 获取评分数大于1的物品
        item_ratings = sampled_data['movieId'].value_counts()
        qualified_items = item_ratings[item_ratings > 1].index
        
        # 只保留有效物品的评分
        sampled_data = sampled_data[sampled_data['movieId'].isin(qualified_items)]
        
        print(f"采样后数据规模: {len(sampled_data)} 条评分")
        print(f"采样后用户数: {len(sampled_data['userId'].unique())}")
        print(f"采样后物品数: {len(sampled_data['movieId'].unique())}")
        
        self.ratings_df = sampled_data
        return sampled_data
        
    def process(self, test_size=0.2, random_state=42, sample_users=10000, min_ratings=5):
        """处理数据并划分训练集和测试集
        
        Args:
            test_size (float): 测试集比例
            random_state (int): 随机种子
            sample_users (int): 采样用户数量
            min_ratings (int): 用户最少评分数
            
        Returns:
            tuple: (训练集, 测试集)
        """
        # 数据采样
        self.sample_data(user_sample_size=sample_users, min_ratings=min_ratings)
        
        # 创建用户和物品的映射
        unique_users = self.ratings_df['userId'].unique()
        unique_items = self.ratings_df['movieId'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # 转换用户ID和物品ID为矩阵索引
        user_idx = self.ratings_df['userId'].map(self.user_mapping)
        item_idx = self.ratings_df['movieId'].map(self.item_mapping)
        
        # 创建稀疏矩阵
        self.user_item_matrix = csr_matrix(
            (self.ratings_df['rating'], (user_idx, item_idx)),
            shape=(len(unique_users), len(unique_items))
        )
        
        # 获取所有评分记录
        ratings_records = np.column_stack((
            user_idx,
            item_idx,
            self.ratings_df['rating'].values
        ))
        
        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            ratings_records,
            test_size=test_size,
            random_state=random_state
        )
        
        return train_data, test_data
    
    def get_user_item_matrix(self):
        """获取用户-物品评分矩阵
        
        Returns:
            csr_matrix: 用户-物品评分矩阵
        """
        return self.user_item_matrix
    
    def get_users(self):
        """获取所有用户ID列表
        
        Returns:
            list: 用户ID列表
        """
        return list(self.user_mapping.keys())
    
    def get_items(self):
        """获取所有物品ID列表
        
        Returns:
            list: 物品ID列表
        """
        return list(self.item_mapping.keys())
    
    def get_user_ratings(self, user_id):
        """获取指定用户的所有评分记录
        
        Args:
            user_id (int): 用户ID
            
        Returns:
            DataFrame: 用户的评分记录
        """
        return self.ratings_df[self.ratings_df['userId'] == user_id]
    
    def get_item_ratings(self, item_id):
        """获取指定物品的所有评分记录
        
        Args:
            item_id (int): 物品ID
            
        Returns:
            DataFrame: 物品的评分记录
        """
        return self.ratings_df[self.ratings_df['movieId'] == item_id] 