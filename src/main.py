import os
import numpy as np
from data_processor import DataProcessor
from user_cf import UserCF
from item_cf import ItemCF
from evaluator import Evaluator

def main():
    # 设置数据路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ratings_path = os.path.join(project_root, 'ml-32m', 'ml-32m', 'ratings.csv')
    
    # 初始化数据处理器
    print("Loading data...")
    processor = DataProcessor(ratings_path)
    
    # 处理数据
    train_data, test_data = processor.process(test_size=0.2)
    user_item_matrix = processor.get_user_item_matrix()
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 训练和评估 UserCF
    print("\nTraining UserCF model...")
    user_cf = UserCF(
        n_neighbors=20,  # 增加邻居数量以提高稳定性
        similarity_metric='pearson'  # 使用皮尔逊相关系数
    )
    user_cf.fit(user_item_matrix)
    
    # 在测试集上进行预测
    print("Evaluating UserCF model...")
    print(f"Test data shape: {test_data.shape}")
    
    user_cf_predictions = []
    for i, (user_id, item_id, rating) in enumerate(test_data):
        pred = user_cf.predict(user_id, item_id)
        user_cf_predictions.append(pred)
        if i % 100000 == 0:
            print(f"Processed {i}/{len(test_data)} predictions")
    
    user_cf_predictions = np.array(user_cf_predictions)
    print(f"Predictions shape: {user_cf_predictions.shape}")
    
    # 确保预测结果和测试数据长度一致
    assert len(user_cf_predictions) == len(test_data), \
        f"Predictions length ({len(user_cf_predictions)}) != Test data length ({len(test_data)})"
    
    # 评估 UserCF
    user_cf_metrics = evaluator.evaluate(test_data[:, 2], user_cf_predictions)
    print("\nUserCF Metrics:")
    for metric, value in user_cf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制 UserCF 评估图
    evaluator.plot_prediction_distribution(test_data[:, 2], user_cf_predictions)
    evaluator.plot_error_distribution(test_data[:, 2], user_cf_predictions)
    
    # 训练和评估 ItemCF
    print("\nTraining ItemCF model...")
    item_cf = ItemCF(n_neighbors=10)
    item_cf.fit(user_item_matrix)
    
    # 在测试集上进行预测
    print("Evaluating ItemCF model...")
    print(f"Test data shape: {test_data.shape}")
    
    item_cf_predictions = []
    for i, (user_id, item_id, rating) in enumerate(test_data):
        pred = item_cf.predict(user_id, item_id)
        item_cf_predictions.append(pred)
        if i % 100000 == 0:
            print(f"Processed {i}/{len(test_data)} predictions")
    
    item_cf_predictions = np.array(item_cf_predictions)
    print(f"Predictions shape: {item_cf_predictions.shape}")
    
    # 确保预测结果和测试数据长度一致
    assert len(item_cf_predictions) == len(test_data), \
        f"Predictions length ({len(item_cf_predictions)}) != Test data length ({len(test_data)})"
    
    # 评估 ItemCF
    item_cf_metrics = evaluator.evaluate(test_data[:, 2], item_cf_predictions)
    print("\nItemCF Metrics:")
    for metric, value in item_cf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 绘制 ItemCF 评估图
    evaluator.plot_prediction_distribution(test_data[:, 2], item_cf_predictions)
    evaluator.plot_error_distribution(test_data[:, 2], item_cf_predictions)

if __name__ == '__main__':
    main() 