# 推荐系统算法实现

本项目实现了多种推荐系统算法，包括基于用户的协同过滤(UserCF)、基于物品的协同过滤(ItemCF)、基于用户画像的协同过滤(UserProfileCF)和基于时间的协同过滤(TimeAwareCF)。

## 项目结构

```
src/
├── main.py              # 主程序入口
├── data_processor.py    # 数据预处理
├── user_cf.py          # 基于用户的协同过滤
├── item_cf.py          # 基于物品的协同过滤
├── user_profile_cf.py  # 基于用户画像的协同过滤
└── time_aware_cf.py    # 基于时间的协同过滤
```

## 环境要求

- Python 3.7+
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集
   - 将MovieLens数据集放在`data/`目录下
   - 支持ml-100k、ml-1m等格式

2. 运行程序
   ```bash
   python src/main.py
   ```

3. 查看结果
   - 评估指标将打印在控制台
   - 可视化结果保存在`plots/`目录下

## 评估指标

- MAE (平均绝对误差)
- RMSE (均方根误差)
- Precision (准确率)
- Recall (召回率)
- F1 Score (F1分数)
- NDCG@10
- Coverage (覆盖率)
- Training Time (训练时间)

## 可视化结果

- accuracy_comparison.png: 准确性指标对比
- quality_metrics.png: 推荐质量指标对比
- ndcg_coverage.png: NDCG和覆盖率对比
- training_time.png: 训练时间对比
- metrics_heatmap.png: 性能指标热力图
- radar_chart.png: 算法性能雷达图

## 注意事项

1. 数据集文件未包含在代码仓库中，需要自行下载
2. 建议使用虚拟环境运行程序
3. 首次运行可能需要较长时间进行数据处理和模型训练