import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from user_cf import UserCF
from item_cf import ItemCF
from data_processor import DataProcessor
import time
import psutil
import datetime

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB

def log_progress(message, start_time=None, memory_usage=True):
    """记录进度和资源使用情况"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory = f"内存使用: {get_memory_usage():.2f}MB" if memory_usage else ""
    elapsed = f"耗时: {(time.time() - start_time)/60:.2f}分钟" if start_time else ""
    print(f"[{current_time}] {message} {memory} {elapsed}")

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def prepare_data(ratings_path, data_dir='data', batch_size=500, sample_users=10000, min_ratings=5):
    """准备训练和���数据"""
    start_time = time.time()
    log_progress("开始数据准备...")
    
    ensure_dir(data_dir)
    train_path = os.path.join(data_dir, f'train_matrix_u{sample_users}_r{min_ratings}.npz')
    test_path = os.path.join(data_dir, f'test_matrix_u{sample_users}_r{min_ratings}.npz')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        log_progress("加载已处理的数据文件...")
        train_matrix = load_npz(train_path)
        test_matrix = load_npz(test_path)
        log_progress("数据加载完成", start_time)
        return train_matrix, test_matrix
    
    log_progress("处理新数据...")
    processor = DataProcessor(ratings_path)
    train_data, test_data = processor.process(
        sample_users=sample_users,
        min_ratings=min_ratings
    )
    
    log_progress("转换为稀疏矩阵格式...")
    n_users = len(processor.get_users())
    n_items = len(processor.get_items())
    
    train_matrix = csr_matrix(
        (train_data[:, 2], (train_data[:, 0], train_data[:, 1])),
        shape=(n_users, n_items)
    )
    test_matrix = csr_matrix(
        (test_data[:, 2], (test_data[:, 0], test_data[:, 1])),
        shape=(n_users, n_items)
    )
    
    log_progress("保存处理后的数据...")
    save_npz(train_path, train_matrix)
    save_npz(test_path, test_matrix)
    
    log_progress("数据准备完成", start_time)
    return train_matrix, test_matrix

def evaluate_model(model, test_matrix, n_users=1000):
    """评估模型性能"""
    start_time = time.time()
    log_progress(f"开始评估模型 (评估用户数: {n_users})...")
    
    mae_sum = 0.0
    rmse_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    diversity_sum = 0.0
    count = 0
    valid_users = 0
    
    test_users = np.random.choice(test_matrix.shape[0], size=n_users, replace=False)
    total_predictions = 0
    valid_predictions = 0
    
    # 用于计算覆盖率
    recommended_items = set()
    total_items = set(range(test_matrix.shape[1]))
    
    for i, user_id in enumerate(test_users):
        # 获取该用户的实际评分
        actual_ratings = test_matrix[user_id].toarray().ravel()
        rated_items = np.nonzero(actual_ratings)[0]
        
        if len(rated_items) == 0:
            continue
            
        user_predictions = []
        user_actuals = []
        
        # 预测评分
        for item_id in rated_items:
            try:
                pred = model.predict(user_id, item_id)
                if not np.isnan(pred):
                    user_predictions.append(pred)
                    user_actuals.append(actual_ratings[item_id])
                    valid_predictions += 1
                    recommended_items.add(item_id)
            except Exception as e:
                continue
        
        if len(user_predictions) > 0:
            # 计算基本误差指标
            user_predictions = np.array(user_predictions)
            user_actuals = np.array(user_actuals)
            errors = np.abs(user_predictions - user_actuals)
            
            mae_sum += np.sum(errors)
            rmse_sum += np.sum(errors ** 2)
            
            # 计算准确率和召回率
            threshold = 3.5  # 认为评分大于3.5的为"喜欢"
            actual_liked = user_actuals >= threshold
            pred_liked = user_predictions >= threshold
            
            if np.sum(pred_liked) > 0:
                precision = np.sum(actual_liked & pred_liked) / np.sum(pred_liked)
                precision_sum += precision
            
            if np.sum(actual_liked) > 0:
                recall = np.sum(actual_liked & pred_liked) / np.sum(actual_liked)
                recall_sum += recall
            
            # 计算NDCG@10
            k = min(10, len(user_predictions))
            sorted_indices = np.argsort(-user_predictions)[:k]
            dcg = np.sum(user_actuals[sorted_indices] / np.log2(np.arange(2, k + 2)))
            
            ideal_sorted_indices = np.argsort(-user_actuals)[:k]
            idcg = np.sum(user_actuals[ideal_sorted_indices] / np.log2(np.arange(2, k + 2)))
            
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_sum += ndcg
            
            # 计算推荐多样性
            unique_items = len(set(np.where(user_predictions >= threshold)[0]))
            diversity = unique_items / len(user_predictions)
            diversity_sum += diversity
            
            count += 1
            valid_users += 1
        
        total_predictions += len(rated_items)
        
        if (i + 1) % 10 == 0:
            log_progress(f"已评估 {i+1}/{n_users} 个用户, 有效预测: {valid_predictions}/{total_predictions}")
    
    if count > 0:
        mae = mae_sum / count
        rmse = np.sqrt(rmse_sum / count)
        precision = precision_sum / count
        recall = recall_sum / count
        ndcg = ndcg_sum / count
        diversity = diversity_sum / count
        coverage = len(recommended_items) / len(total_items)
        
        log_progress(f"评估完成 - 有效用户: {valid_users}/{n_users}, 有效预测: {valid_predictions}/{total_predictions}")
    else:
        mae = rmse = precision = recall = ndcg = diversity = coverage = np.nan
        log_progress("警告: 没有有效的预测结果!")
    
    log_progress("评估完成", start_time)
    return {
        'mae': mae,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'ndcg': ndcg,
        'diversity': diversity,
        'coverage': coverage,
        'valid_predictions': valid_predictions,
        'total_predictions': total_predictions
    }

def main():
    total_start_time = time.time()
    log_progress("开始推荐系统评估...")
    
    # 调整数据规模和参数
    sample_users = 1000    # 采样用户数
    min_ratings = 20       # 每个用户最少评分数
    batch_size = 100      # 批处理大小
    n_neighbors = 30      # 增加邻居数量以提高覆盖率
    n_eval_users = 50     # 增加评估用户数量
    
    # 修正数据文件路径
    ratings_path = '../ml-32m/ml-32m/ratings.csv'
    if not os.path.exists(ratings_path):
        log_progress(f"错误: 找不到数据文件 {ratings_path}")
        log_progress("请确保数据文件位于正确位置")
        return
    
    train_matrix, test_matrix = prepare_data(
        ratings_path,
        data_dir='../data',
        batch_size=batch_size,
        sample_users=sample_users,
        min_ratings=min_ratings
    )
    
    log_progress(f"数据集规模: {train_matrix.shape[0]}用户, {train_matrix.shape[1]}物品")
    log_progress(f"训练集非零元素: {train_matrix.nnz}, 稀疏度: {train_matrix.nnz/(train_matrix.shape[0]*train_matrix.shape[1])*100:.4f}%")
    log_progress(f"测试集非零元素: {test_matrix.nnz}, 稀疏度: {test_matrix.nnz/(test_matrix.shape[0]*test_matrix.shape[1])*100:.4f}%")
    
    # 评估基于用户的协同过滤
    log_progress("\n开始基于用户的协同过滤评估...")
    user_cf = UserCF(n_neighbors=n_neighbors)
    user_start_time = time.time()
    user_cf.fit(train_matrix)
    user_metrics = evaluate_model(user_cf, test_matrix, n_users=n_eval_users)
    user_time = time.time() - user_start_time
    
    log_progress("\n基于用户的协同过滤结果:")
    print(f"MAE: {user_metrics['mae']:.4f}")
    print(f"RMSE: {user_metrics['rmse']:.4f}")
    print(f"准确率: {user_metrics['precision']:.4f}")
    print(f"召回率: {user_metrics['recall']:.4f}")
    print(f"F1分数: {user_metrics['f1']:.4f}")
    print(f"NDCG@10: {user_metrics['ndcg']:.4f}")
    print(f"多样性: {user_metrics['diversity']:.4f}")
    print(f"覆盖率: {user_metrics['coverage']:.4f}")
    print(f"有效预测率: {user_metrics['valid_predictions']}/{user_metrics['total_predictions']} ({user_metrics['valid_predictions']/user_metrics['total_predictions']*100:.2f}%)")
    print(f"耗时: {user_time/60:.2f}分钟")
    
    # 评估基于物品的协同过滤
    log_progress("\n开始基于物品的协同过滤评估...")
    item_cf = ItemCF(n_neighbors=n_neighbors)
    item_start_time = time.time()
    item_cf.fit(train_matrix)
    item_metrics = evaluate_model(item_cf, test_matrix, n_users=n_eval_users)
    item_time = time.time() - item_start_time
    
    log_progress("\n基于物品的协同过滤结果:")
    print(f"MAE: {item_metrics['mae']:.4f}")
    print(f"RMSE: {item_metrics['rmse']:.4f}")
    print(f"准确率: {item_metrics['precision']:.4f}")
    print(f"召回率: {item_metrics['recall']:.4f}")
    print(f"F1分数: {item_metrics['f1']:.4f}")
    print(f"NDCG@10: {item_metrics['ndcg']:.4f}")
    print(f"多样性: {item_metrics['diversity']:.4f}")
    print(f"覆盖率: {item_metrics['coverage']:.4f}")
    print(f"有效预测率: {item_metrics['valid_predictions']}/{item_metrics['total_predictions']} ({item_metrics['valid_predictions']/item_metrics['total_predictions']*100:.2f}%)")
    print(f"耗时: {item_time/60:.2f}分钟")
    
    # 输出比较结果
    log_progress("\n算法比较:")
    metrics = ['mae', 'rmse', 'precision', 'recall', 'f1', 'ndcg', 'diversity', 'coverage']
    print("\n指标\t\tUserCF\t\tItemCF")
    print("-" * 50)
    for metric in metrics:
        print(f"{metric.upper()}\t\t{user_metrics[metric]:.4f}\t\t{item_metrics[metric]:.4f}")
    
    log_progress("\n评估完成", total_start_time)

if __name__ == '__main__':
    main() 