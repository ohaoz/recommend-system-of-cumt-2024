import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DeepCFDataset(Dataset):
    def __init__(self, rating_matrix):
        self.users, self.items = rating_matrix.nonzero()
        self.ratings = rating_matrix[self.users, self.items].A.ravel()
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }

class DeepCFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors=100, layers=[64, 32]):
        super(DeepCFModel, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP层
        self.layers = nn.ModuleList()
        input_size = n_factors * 2
        for size in layers:
            self.layers.append(nn.Linear(input_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(nn.Dropout(0.2))
            input_size = size
            
        # 输出层
        self.output = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        # 获取嵌入
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # 连接用户和物品嵌入
        x = torch.cat([user_embeds, item_embeds], dim=1)
        
        # 通过MLP层
        for layer in self.layers:
            x = layer(x)
            
        # 输出预测
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x.squeeze()

class DeepCF:
    def __init__(self, n_factors=100, layers=[64, 32], batch_size=256, n_epochs=10, lr=0.001):
        self.n_factors = n_factors
        self.layers = layers
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.mean_rating = None
        
    def fit(self, rating_matrix):
        """训练模型
        
        Args:
            rating_matrix: 用户-物品评分矩阵
        """
        self.mean_rating = rating_matrix.data.mean()
        n_users, n_items = rating_matrix.shape
        
        # 创建数据集和数据加载器
        dataset = DeepCFDataset(rating_matrix)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型
        self.model = DeepCFModel(n_users, n_items, self.n_factors, self.layers).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 训练模型
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch in dataloader:
                user_ids = batch['user'].to(self.device)
                item_ids = batch['item'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # 前向传播
                pred = self.model(user_ids, item_ids)
                loss = criterion(pred, ratings)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}')
            
    def predict(self, user_id, item_id):
        """预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测的评分
        """
        if self.model is None:
            return self.mean_rating
            
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_id], dtype=torch.long).to(self.device)
            pred = self.model(user_tensor, item_tensor)
            
            # 将预测值映射到评分范围 (1-5)
            pred = pred.item() * 4 + 1
            
        return pred 