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
        
    def process(self, test_size=0.2, random_state=42):
        """处理数据并划分训练集和测试集
        
        Args:
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        Returns:
            tuple: (训练集, 测试集)
        """
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