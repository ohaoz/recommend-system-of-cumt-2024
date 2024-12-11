import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from user_cf import UserCF
from item_cf import ItemCF
from user_profile_cf import UserProfileCF
from time_aware_cf import TimeAwareCF
from data_processor import DataProcessor
import time
import psutil
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    """准备训练和测试数据
    
    Args:
        ratings_path (str): 评分数据文件路径
        data_dir (str): 数据存储目录
        batch_size (int): 批处理大小
        sample_users (int): 采样用户数
        min_ratings (int): 最少评分数
        
    Returns:
        tuple: (训练集矩阵, 测试集矩阵)
    """
    start_time = time.time()
    log_progress("开始数据准备...")
    
    ensure_dir(data_dir)
    train_path = os.path.join(data_dir, f'train_matrix_u{sample_users}_r{min_ratings}.npz')
    test_path = os.path.join(data_dir, f'test_matrix_u{sample_users}_r{min_ratings}.npz')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        log_progress("加载已处理的数据文件...")
        train_matrix = load_npz(train_path)
        test_matrix = load_npz(test_path)
        log_progress(f"训练集规模: {train_matrix.shape}, 非零元素: {train_matrix.nnz}")
        log_progress(f"测试集规模: {test_matrix.shape}, 非零元素: {test_matrix.nnz}")
        log_progress("数据加载完成", start_time)
        return train_matrix, test_matrix
    
    log_progress("处理新数据...")
    processor = DataProcessor(ratings_path)
    train_data, test_data = processor.process(
        test_size=0.2,  # 20%作为测试集
        random_state=42,  # 固定随机种子以确保可重复性
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
    
    log_progress(f"训练集规模: {train_matrix.shape}, 非零元素: {train_matrix.nnz}")
    log_progress(f"测试集规模: {test_matrix.shape}, 非零元素: {test_matrix.nnz}")
    log_progress(f"训练集稀疏度: {train_matrix.nnz/(n_users*n_items)*100:.4f}%")
    log_progress(f"测试集稀疏度: {test_matrix.nnz/(n_users*n_items)*100:.4f}%")
    
    log_progress("保存处理后的数据...")
    save_npz(train_path, train_matrix)
    save_npz(test_path, test_matrix)
    
    log_progress("数据准备完成", start_time)
    return train_matrix, test_matrix

def evaluate_model(model, test_matrix, n_users=1000, n_rec=10):
    """评估模型性能
    
    Args:
        model: 推荐模型
        test_matrix: 测试集矩阵
        n_users: 评估用户数
        n_rec: 推荐列表长度
    """
    start_time = time.time()
    log_progress(f"开始评估模型 (评估用户数: {n_users})...")
    
    mae_sum = 0.0
    rmse_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0
    valid_users = 0
    total_error_count = 0
    
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
        user_items = []  # 存储物品ID
        
        # 预测评分
        for item_id in rated_items:
            try:
                pred = model.predict(user_id, item_id)
                if not np.isnan(pred):
                    user_predictions.append(pred)
                    user_actuals.append(actual_ratings[item_id])
                    user_items.append(item_id)
                    valid_predictions += 1
                    recommended_items.add(item_id)
            except Exception as e:
                continue
        
        if len(user_predictions) > 0:
            # 计算基本误差指标
            user_predictions = np.array(user_predictions)
            user_actuals = np.array(user_actuals)
            errors = np.abs(user_predictions - user_actuals)
            
            # 计算每个用户的平均误差
            user_mae = np.mean(errors)
            user_rmse = np.sqrt(np.mean(errors ** 2))
            
            mae_sum += user_mae
            rmse_sum += user_rmse
            total_error_count += len(errors)
            
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
            
            count += 1
            valid_users += 1
        
        total_predictions += len(rated_items)
        
        if (i + 1) % 10 == 0:
            log_progress(f"已评估 {i+1}/{n_users} 个用户, 有效预测: {valid_predictions}/{total_predictions}")
            if count > 0:
                current_mae = mae_sum / count
                current_rmse = rmse_sum / count
                log_progress(f"当前 MAE: {current_mae:.4f}, RMSE: {current_rmse:.4f}")
    
    if count > 0:
        # 使用用户平均误差的平均值
        mae = mae_sum / count
        rmse = rmse_sum / count
        precision = precision_sum / count
        recall = recall_sum / count
        ndcg = ndcg_sum / count
        coverage = len(recommended_items) / len(total_items)
        
        log_progress(f"评估完成 - 有效用户: {valid_users}/{n_users}, 有效预测: {valid_predictions}/{total_predictions}")
        log_progress(f"平均每用户预测数: {total_error_count/count:.2f}")
    else:
        mae = rmse = precision = recall = ndcg = coverage = np.nan
        log_progress("警告: 没有有效的预测结果!")
    
    log_progress("评估完成", start_time)
    return {
        'mae': mae,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'ndcg': ndcg,
        'coverage': coverage,
        'valid_predictions': valid_predictions,
        'total_predictions': total_predictions,
        'predictions_per_user': total_error_count/count if count > 0 else 0
    }

def save_results(metrics, model_name, output_dir='results'):
    """保存评估结果到文件
    
    Args:
        metrics (dict): 评估指标
        model_name (str): 模型名称
        output_dir (str): 输出目录
    """
    ensure_dir(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'{model_name}_results_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    log_progress(f"结果已保存到: {output_file}")

def plot_metrics(results, output_dir='plots'):
    """绘制评价指标图表
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
    """
    ensure_dir(output_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 1. 准确性指标对比 (MAE & RMSE)
    plt.figure(figsize=(10, 6))
    metrics = ['MAE', 'RMSE']
    x = np.arange(len(results.keys()))
    width = 0.35
    
    mae_values = [results[model]['mae'] for model in results]
    rmse_values = [results[model]['rmse'] for model in results]
    
    plt.bar(x - width/2, mae_values, width, label='MAE')
    plt.bar(x + width/2, rmse_values, width, label='RMSE')
    
    plt.xlabel('推荐算法')
    plt.ylabel('误差值')
    plt.title('各算法准确性指标对比')
    plt.xticks(x, results.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # 2. 推荐质量指标对比 (Precision, Recall, F1)
    plt.figure(figsize=(10, 6))
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(results.keys()))
    width = 0.25
    
    precision_values = [results[model]['precision'] for model in results]
    recall_values = [results[model]['recall'] for model in results]
    f1_values = [results[model]['f1'] for model in results]
    
    plt.bar(x - width, precision_values, width, label='准确率')
    plt.bar(x, recall_values, width, label='召回率')
    plt.bar(x + width, f1_values, width, label='F1分数')
    
    plt.xlabel('推荐算法')
    plt.ylabel('指标值')
    plt.title('各算法推荐质量指标对比')
    plt.xticks(x, results.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_metrics.png'), bbox_inches='tight')
    plt.close()
    
    # 3. NDCG和覆盖率对比
    plt.figure(figsize=(10, 6))
    metrics = ['NDCG@10', 'Coverage']
    x = np.arange(len(results.keys()))
    width = 0.35
    
    ndcg_values = [results[model]['ndcg'] for model in results]
    coverage_values = [results[model]['coverage'] for model in results]
    
    plt.bar(x - width/2, ndcg_values, width, label='NDCG@10')
    plt.bar(x + width/2, coverage_values, width, label='覆盖率')
    
    plt.xlabel('推荐算法')
    plt.ylabel('指标值')
    plt.title('各算法NDCG和覆盖率对比')
    plt.xticks(x, results.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ndcg_coverage.png'), bbox_inches='tight')
    plt.close()
    
    # 4. 训练时间对比
    plt.figure(figsize=(10, 6))
    times = [results[model]['training_time'] for model in results]
    plt.bar(results.keys(), times)
    plt.xlabel('推荐算法')
    plt.ylabel('训练时间 (秒)')
    plt.title('各算法训练时间对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time.png'), bbox_inches='tight')
    plt.close()
    
    # 5. 热力图展示所有指标
    plt.figure(figsize=(12, 8))
    metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F1', 'NDCG@10', 'Coverage', 'Training Time']
    data = []
    for model in results:
        data.append([
            results[model]['mae'],
            results[model]['rmse'],
            results[model]['precision'],
            results[model]['recall'],
            results[model]['f1'],
            results[model]['ndcg'],
            results[model]['coverage'],
            results[model]['training_time']
        ])
    
    # 数据标准化，使不同量级的指标可比
    data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    sns.heatmap(data_normalized, 
                xticklabels=metrics,
                yticklabels=results.keys(),
                annot=np.array(data).round(4),  # 显示原始值
                fmt='.4f',
                cmap='RdYlBu_r')
    plt.title('算法性能指标热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), bbox_inches='tight')
    plt.close()
    
    # 6. 雷达图
    plt.figure(figsize=(10, 10))
    metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F1', 'NDCG@10']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # 闭合雷达图
    angles = np.concatenate((angles, [angles[0]]))
    metrics = metrics + [metrics[0]]
    
    for model in results:
        values = [
            results[model]['mae'],
            results[model]['rmse'],
            results[model]['precision'],
            results[model]['recall'],
            results[model]['f1'],
            results[model]['ndcg']
        ]
        # 标准化值到0-1之间
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values = np.concatenate((values, [values[0]]))
        
        plt.polar(angles, values, label=model, marker='o')
    
    plt.xticks(angles[:-1], metrics[:-1])
    plt.title('算法性能雷达图')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), bbox_inches='tight')
    plt.close()

def main():
    total_start_time = time.time()
    log_progress("开始推荐系统评估...")
    
    # 调整数据规模和参数
    sample_users = 500
    min_ratings = 5
    batch_size = 100
    n_neighbors = 20
    n_eval_users = 100
    
    # 准备数据
    ratings_path = '../ml-latest-small/ratings.csv'
    train_matrix, test_matrix = prepare_data(
        ratings_path,
        data_dir='../data',
        batch_size=batch_size,
        sample_users=sample_users,
        min_ratings=min_ratings
    )
    
    results = {}
    
    # 1. 基于用户的协同过滤
    log_progress("\n1. 基于用户的协同过滤评估...")
    user_cf = UserCF(n_neighbors=n_neighbors)
    user_start_time = time.time()
    try:
        user_cf.fit(train_matrix)
        user_train_time = time.time() - user_start_time
        user_metrics = evaluate_model(user_cf, test_matrix, n_users=n_eval_users)
        user_metrics['training_time'] = user_train_time
        user_metrics['timestamp'] = datetime.datetime.now().isoformat()
        user_metrics['parameters'] = {
            'n_neighbors': n_neighbors,
            'sample_users': sample_users,
            'min_ratings': min_ratings,
            'n_eval_users': n_eval_users
        }
        save_results(user_metrics, 'user_cf')
        results['UserCF'] = user_metrics
        log_progress(f"UserCF 训练时间: {user_train_time:.2f}秒")
    except Exception as e:
        log_progress(f"UserCF 训练失败: {str(e)}")
    
    # 2. 基于物品的协同过滤
    log_progress("\n2. 基于物品的协同过滤评估...")
    item_cf = ItemCF(n_neighbors=n_neighbors)
    item_start_time = time.time()
    try:
        item_cf.fit(train_matrix)
        item_train_time = time.time() - item_start_time
        item_metrics = evaluate_model(item_cf, test_matrix, n_users=n_eval_users)
        item_metrics['training_time'] = item_train_time
        item_metrics['timestamp'] = datetime.datetime.now().isoformat()
        item_metrics['parameters'] = {
            'n_neighbors': n_neighbors,
            'sample_users': sample_users,
            'min_ratings': min_ratings,
            'n_eval_users': n_eval_users
        }
        save_results(item_metrics, 'item_cf')
        results['ItemCF'] = item_metrics
        log_progress(f"ItemCF 训练时间: {item_train_time:.2f}秒")
    except Exception as e:
        log_progress(f"ItemCF 训练失败: {str(e)}")
    
    # 3. 基于用户画像的协同过滤
    log_progress("\n3. 基于用户画像的协同过滤评估...")
    profile_cf = UserProfileCF()
    profile_start_time = time.time()
    try:
        profile_cf.fit(train_matrix)
        profile_train_time = time.time() - profile_start_time
        profile_metrics = evaluate_model(profile_cf, test_matrix, n_users=n_eval_users)
        profile_metrics['training_time'] = profile_train_time
        profile_metrics['timestamp'] = datetime.datetime.now().isoformat()
        profile_metrics['parameters'] = {
            'sample_users': sample_users,
            'min_ratings': min_ratings,
            'n_eval_users': n_eval_users
        }
        save_results(profile_metrics, 'user_profile_cf')
        results['UserProfileCF'] = profile_metrics
        log_progress(f"UserProfileCF 训练时间: {profile_train_time:.2f}秒")
    except Exception as e:
        log_progress(f"UserProfileCF 训练失败: {str(e)}")
    
    # 4. 基于时间的协同过滤
    log_progress("\n4. 基于时间的协同过滤评估...")
    time_cf = TimeAwareCF()
    time_start = time.time()
    try:
        time_cf.fit(train_matrix)
        time_train_time = time.time() - time_start
        time_metrics = evaluate_model(time_cf, test_matrix, n_users=n_eval_users)
        time_metrics['training_time'] = time_train_time
        time_metrics['timestamp'] = datetime.datetime.now().isoformat()
        time_metrics['parameters'] = {
            'sample_users': sample_users,
            'min_ratings': min_ratings,
            'n_eval_users': n_eval_users
        }
        save_results(time_metrics, 'time_aware_cf')
        results['TimeAwareCF'] = time_metrics
        log_progress(f"TimeAwareCF 训练时间: {time_train_time:.2f}秒")
    except Exception as e:
        log_progress(f"TimeAwareCF 训练失败: {str(e)}")
    
    # 打印总结果
    log_progress("\n评估结果汇总:")
    
    # 创建比较表格
    metrics = ['mae', 'rmse', 'precision', 'recall', 'f1', 'ndcg', 'coverage', 'training_time']
    index = ['MAE', 'RMSE', 'Precision', 'Recall', 'F1', 'NDCG@10', 'Coverage', 'Training Time(sec)']
    
    comparison_data = {}
    for model_name, model_metrics in results.items():
        comparison_data[model_name] = [model_metrics.get(m, np.nan) for m in metrics]
    
    comparison = pd.DataFrame(comparison_data, index=index)
    
    print("\n性能指标比较:")
    print(comparison.round(4).to_markdown())
    
    # 打印详细的训练时间信息
    print("\n训练时间详情:")
    for model_name, model_metrics in results.items():
        train_time = model_metrics.get('training_time', np.nan)
        if not np.isnan(train_time):
            print(f"{model_name}: {train_time:.2f}秒 ({train_time/60:.4f}分钟)")
    
    # 在打印结果后��加可视化
    log_progress("\n生成评估结果可视化...")
    plot_metrics(results)
    log_progress("可视化完成，图表已保存到plots目录")
    
    total_time = time.time() - total_start_time
    log_progress(f"\n总评估时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

if __name__ == '__main__':
    main() 