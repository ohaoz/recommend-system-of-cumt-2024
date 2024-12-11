# 协同过滤推荐系统

基于MovieLens数据集实现的协同过滤推荐系统，包含基于用户和基于物品的两种方法。

## 功能特点

1. **算法实现**
   - 基于用户的协同过滤(UserCF)
   - 基于物品的协同过滤(ItemCF)
   - IDF权重优化
   - 评分惩罚机制

2. **评估指标**
   - MAE (平均绝对误差)
   - RMSE (均方根误差)
   - 准确率(Precision)
   - 召回率(Recall)
   - F1分数
   - NDCG@10
   - 多样性(Diversity)
   - 覆盖率(Coverage)

3. **性能优化**
   - 批处理计算
   - 稀疏矩阵优化
   - 内存使用优化

## 项目结构

```
recommend/
├── data/                # 数据目录
├── src/                 # 源代码
│   ├── main.py         # 主程序
│   ├── user_cf.py      # 基于用户的协同过滤
│   ├── item_cf.py      # 基于物品的协同过滤
│   └── data_processor.py# 数据处理
└── README.md           # 项目说明
```

## 使用方法

1. **环境准备**
   ```bash
   pip install numpy pandas scipy scikit-learn
   ```

2. **运行程序**
   ```bash
   cd src
   python main.py
   ```

## 实验结果

### UserCF vs ItemCF

指标 | UserCF | ItemCF
--- | --- | ---
MAE | 22.44 | 24.43
RMSE | 5.35 | 5.67
Precision | 0.74 | 0.73
Recall | 0.76 | 0.67
F1 | 0.75 | 0.70
NDCG | 0.90 | 0.90
Diversity | 0.72 | 0.63
Coverage | 0.13 | 0.15

## 注意事项

1. 数据集较大，建议使用足够的内存
2. 可以通过调整参数优化性能
3. 推荐根据实际需求选择合适的算法

## 未来改进

1. 添加更多评估指标
2. 实现混合推荐策略
3. 优化计算效率
4. 添加更多特征工程