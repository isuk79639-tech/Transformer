# Transformer

```
src/
├── config.py              # 配置管理
├── components.py          # Transformer 核心组件
├── transformer.py         # 完整 Transformer 模型
├── data_utils.py         # 数据处理工具
├── training_utils.py     # 训练稳定性工具
├── model_utils.py        # 模型分析和检查点管理
├── trainer.py            # 训练器
├── experiments.py        # 消融实验和敏感性分析
├── train.py              # 主训练脚本
└── requirements.txt      # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

##### 基础训练

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 1e-4
```

##### 消融实验和学习率超参数分析

```bash
# 完整消融实验（每个配置训练10个epoch）
python train.py --run_ablation --ablation_epochs 10
# 完整敏感性分析（每个参数值训练10个epoch）
python train.py --run_sensitivity --sensitivity_epochs 10
```

## 实验配置

```
单卡24GB 3090运行若干小时
```

