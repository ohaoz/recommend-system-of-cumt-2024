import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self):
        """初始化评估器"""
        self.metrics = {}
        
    def evaluate(self, y_true, y_pred):
        """评估预测结果
        
        Args:
            y_true (array-like): 真实评分
            y_pred (array-like): 预测评分
            
        Returns:
            dict: 评估指标
        """
        # 计算MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 计算准确率（将评分四舍五入到最近的0.5，计算完全匹配的比例）
        y_true_rounded = np.round(y_true * 2) / 2
        y_pred_rounded = np.round(y_pred * 2) / 2
        accuracy = np.mean(y_true_rounded == y_pred_rounded)
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy': accuracy
        }
        
        return self.metrics
    
    def plot_prediction_distribution(self, y_true, y_pred):
        """绘制预测值与真实值的分布对比图
        
        Args:
            y_true (array-like): 真实评分
            y_pred (array-like): 预测评分
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制真实值分布
        plt.hist(y_true, bins=20, alpha=0.5, label='True Ratings')
        
        # 绘制预测值分布
        plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted Ratings')
        
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Distribution of True vs Predicted Ratings')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_error_distribution(self, y_true, y_pred):
        """绘制预测误差分布图
        
        Args:
            y_true (array-like): 真实评分
            y_pred (array-like): 预测评分
        """
        errors = y_pred - y_true
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True)
        plt.show()
    
    def get_metrics(self):
        """获取评估指标
        
        Returns:
            dict: 评估指标
        """
        return self.metrics 