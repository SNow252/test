# 实验报告模板：带噪声的约束优化问题与自适应算法设计

## 1. 问题描述
本实验研究第一象限单位圆盘内的二维带噪声约束优化问题。算法搜索时每次调用目标函数均重新采样高斯噪声，最终性能统计使用无噪声真实目标函数 f(x)。

## 2. 算法设计
### 2.1 SA
标准模拟退火，采用指数降温、高斯邻域扰动和投影约束处理。

### 2.2 GA
实数编码遗传算法，包含锦标赛选择、blend 交叉、高斯变异、精英保留和自适应变异步长。

### 2.3 NAMS
NAMS（Noise-tolerant Adaptive Memetic Search）为本文设计的噪声容忍自适应模因搜索算法。算法先进行全局随机筛选，再对精英候选进行重采样，随后围绕多个精英点进行自适应局部搜索，最后对候选库进行验证性重采样。该策略避免在早期把预算浪费在单个点的重复评估上，同时在最终选择时降低噪声误导风险。

## 3. 实验设置
每种算法独立运行 50 次，每次预算 2000 次目标函数评估。统计中位数、IQR、可行解比例，并用 Wilcoxon 符号秩检验比较算法差异。

## 4. 实验结果
插入 summary.csv、wilcoxon_tests.csv、boxplot_final_true_f.png、convergence_curves.png 和 best_solution_scatter.png。

## 5. 参数敏感性分析
对 SA 的 T0、alpha 和 GA 的 crossover_prob、mutation_step0 做参数扫描，插入 sensitivity_SA_heatmap.png 和 sensitivity_GA_heatmap.png。

## 6. 开放性探究
本文选择在线估计噪声水平并动态调整算法参数。NAMS 使用候选点重复评估得到的均值和标准误构造保守评分，并根据候选点间差距与不确定性的关系自适应分配重采样预算。

## 7. 结论
根据 summary.csv 和 Wilcoxon 检验结果总结三种算法的收敛速度、稳定性和显著性差异。
