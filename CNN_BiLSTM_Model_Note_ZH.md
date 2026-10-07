# CNN-BiLSTM 模型说明 — BS6200 EEG 癫痫发作识别项目

> 用途：这份文档用于解释 CNN-BiLSTM 分支中模型的逻辑、数据流程、训练方式、评价指标以及当前二分类结果。以后可以直接拿它与 MultiRocket、InceptionTime、XGBoost、TCN 或其他模型进行统一比较。

## 1. 这个模型在项目中的作用

**模型：** 1D CNN + Bidirectional LSTM，也就是 CNN-BiLSTM。

**输入：** 每个样本是一段由 178 个连续测量点组成的 EEG 短片段。

**代码支持的任务：**
- 二分类：癫痫发作 vs. 非癫痫发作。
- 五分类：原始五种 EEG 状态。

**这份文档目前已经填入的正式结果：** 二分类。五分类部分等正式输出后再补充。

这个模型的核心思想是：

```text
原始 EEG
    ↓
CNN 学习局部波形
    ↓
BiLSTM 学习这些局部特征之间的前后关系
    ↓
分类
```

## 2. 当前代码遵循的统一实验协议

`cnn_bilstm_sharedsplit_v3.py` 读取：

```text
splits/shared_split_with_cv.csv
```

当前统一划分为：

- 总样本数：11,500
- Development set：9,200
- Final test set：2,300
- 固定 CV folds：0、1、2、3、4
- 随机种子：6200

代码不会重新随机生成 train/test split，也不会重新生成 CV folds。

这件事非常重要，因为最终六个模型只有使用完全一样的数据划分，结果才可以公平比较。

每一个 CV fold 中：

- 7,360 个样本作为训练集。
- 1,840 个样本作为验证集。
- 2,300 个 final test 样本完全不参与这一过程。

## 3. 数据处理流程

### 3.1 标签定义

二分类：

```text
原始 y = 1   -> seizure -> 内部标签 1
原始 y = 2–5 -> non-seizure -> 内部标签 0
```

五分类：

```text
原始 y = 1–5 -> 内部标签 0–4
```

内部标签变化只是为了方便 PyTorch 训练，不改变原来的类别含义。

### 3.2 标准化以及为什么不会数据泄漏

每一个 CV fold 都只用 training fold 拟合 `StandardScaler`。

流程是：

```text
training fold
    ↓
拟合 scaler
    ↓
标准化 training fold
    ↓
使用同一个 scaler 标准化 validation fold
```

validation fold 不参与均值和标准差的计算。

最终模型阶段：

```text
全部 9,200 development 样本
    ↓
拟合 final scaler
    ↓
标准化 development
    ↓
用同一个 scaler 标准化 final test
```

因此 final test 不会提前把信息泄漏给模型。

## 4. CNN-BiLSTM 网络结构

当前网络为：

```text
178 点 EEG
    ↓
Conv1D：1 -> 32 channels，kernel size = 5
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Max Pooling
    ↓
Conv1D：32 -> 64 channels，kernel size = 5
    ↓
Batch Normalization
    ↓
ReLU
    ↓
Max Pooling
    ↓
Conv1D：64 -> 128 channels，kernel size = 3
    ↓
Batch Normalization
    ↓
ReLU
    ↓
双向 LSTM，每个方向 hidden size = 64
    ↓
全连接层：128 -> 64
    ↓
ReLU
    ↓
Dropout = 0.30
    ↓
输出层
```

### 4.1 CNN 到底在做什么

1D CNN 会让一个小的卷积核沿着 EEG 序列移动。

它主要学习的是**局部波形模式**，例如：

- 突然的信号变化；
- 局部振荡形状；
- 短时间内较大的振幅结构；
- 相邻采样点之间的组合规律。

这些模式不是我们人工规定的，而是训练过程中由网络自己学习。

第一层 CNN 倾向于学习较简单的局部结构，后面的卷积层可以把这些简单结构组合成更抽象的特征。

### 4.2 Batch Normalization 有什么作用

Batch Normalization 用于稳定网络中间层激活值的分布。

它通常可以让优化过程更加稳定，也有利于训练过程更容易收敛。

### 4.3 ReLU 有什么作用

ReLU：

```text
ReLU(x) = max(0, x)
```

它给模型增加非线性表达能力。

如果没有非线性激活函数，即使叠加很多卷积层，本质上仍很难学习复杂 EEG 模式。

### 4.4 Max Pooling 有什么作用

Max Pooling 会缩短序列长度，同时保留局部最强的响应。

本代码大致是：

```text
178 点 -> 约 89 点 -> 约 44 点
```

好处是：

- 降低计算量；
- 让模型对局部模式具体出现在哪一个采样点稍微不那么敏感；
- 将后续 BiLSTM 需要处理的序列缩短。

### 4.5 BiLSTM 到底在做什么

经过 CNN 后，输入已经不再只是 178 个原始数字，而是一段由 CNN 学出来的特征序列。

BiLSTM 会同时从两个方向读取：

```text
Forward：开头 -> 结尾
Backward：结尾 -> 开头
```

每个方向 hidden size = 64，最后合并成 128 维表示。

它的意义是学习：

> CNN 发现的多个局部波形模式，在整段 EEG 中前后如何组合。

因为我们做的是离线的完整 EEG 片段分类，所以整个 178 点在预测前都已经存在，因此从后向前读取不属于信息泄漏。

### 4.6 CNN-LSTM 和 CNN-BiLSTM 的区别

CNN-LSTM 只有一个方向。

CNN-BiLSTM 有两个方向。

BiLSTM 因此可以同时利用某个局部模式之前和之后的信息。

但是：

> BiLSTM 并不一定比 LSTM 更好。

因为我们每个样本只有 178 点，是比较短的序列。BiLSTM 参数更多，也可能带来额外复杂度。因此是否真正提升表现应该通过实验比较，而不是预先假设。

### 4.7 全连接层和 Dropout 的意义

全连接层把 BiLSTM 最后得到的表示转换成类别分数。

Dropout 在训练时随机关闭一部分隐藏单元，其主要目的是减少模型过度依赖某些特征，从而降低过拟合风险。

## 5. 特征工程和特征选择

这个 CNN-BiLSTM 使用：

```text
Raw EEG：X1–X178
```

它没有做传统意义上的人工特征工程，也没有显式执行：

- SelectKBest；
- L1 特征选择；
- PCA；
- SHAP-based selection；
- permutation-importance selection。

这里最容易混淆的一点是：

> CNN 的自动特征学习不等于传统意义上的 feature selection。

传统模型可能是：

```text
Raw EEG
    ↓
人工提取时域/频域/小波特征
    ↓
特征选择
    ↓
分类器
```

CNN-BiLSTM 是：

```text
Raw EEG
    ↓
CNN 自动学习表示
    ↓
BiLSTM
    ↓
分类
```

因此以后组内比较时，这个模型应该被描述为：

**raw-signal representation learning model**

而不是 handcrafted-feature model。

## 6. 类别不平衡如何处理

二分类里面非发作样本明显更多。

当前代码默认使用：

```text
class-weighted cross-entropy
```

类别权重只根据 training fold 计算。

它的作用是让较少的 seizure 类别在 loss 里面拥有更大的影响，从而避免模型只因为 non-seizure 数量多就偏向预测 non-seizure。

这里没有生成假的 EEG 样本，因此和 SMOTE 的思路不同。

## 7. 5-fold CV 到底在做什么

9200 个 development 样本被固定分成五份。

每次：

```text
4 folds -> training
1 fold  -> validation
```

循环五次以后，每一个 development 样本都恰好当过一次 validation。

这样我们能得到五个性能结果。

最终报告：

```text
mean ± SD
```

Mean 表示平均性能。

SD 表示模型在不同数据子集上的稳定程度。

因此比较模型时不能只看 mean，也应该看 SD。

## 8. Early stopping 和 best epoch

这次正式命令允许：

```text
每个 fold 最多 25 epochs
```

但 early stopping 会一直观察 validation loss。

如果 validation loss 连续若干 epoch 没有改善，就可以提前停止。

本次五折得到的 best epochs 是：

```text
[9, 19, 25, 10, 16]
```

不同 fold 的最佳 epoch 不一样完全正常。

程序随后取五个 best epoch 的**中位数**：

```text
median = 16
```

然后用这个值训练 final model。

为什么不用 final test 来选 epoch？

因为这样会泄漏 test 信息。

为什么用中位数？

- 完全由 development CV 决定；
- 不容易被某一个特别早或特别晚的 fold 影响；
- 比直接拍脑袋选一个 epoch 更有依据。

## 9. Final model 是如何得到的

五折 CV 完成后：

```text
五个 best epochs
    ↓
取中位数 = 16
    ↓
把全部 9200 development 样本放在一起
    ↓
重新训练一个新的 CNN-BiLSTM，训练 16 epochs
    ↓
最终只在 2300 个 final test 样本上测一次
```

所以你看到的 final model 不是某一个 CV fold 的模型，而是 CV 完成之后重新训练出来的模型。

最终训练 loss 从第 1 epoch 的大约 0.2767 降到了第 16 epoch 的大约 0.0331。

中间个别 epoch 有小幅升高并不奇怪，因为神经网络采用 mini-batch 和随机优化。

注意：

> final training curve 只有 train loss，不代表我们没有做验证。

验证已经发生在前面的五折 CV 阶段。final model 的 epoch 数也是由 CV 决定的。

## 10. 每一个评价指标到底是什么意思

### Accuracy — 准确率

所有样本里面预测正确的比例。

问题是，如果类别非常不平衡，只预测多数类也可能得到比较高的 Accuracy，因此不能单独使用。

### Balanced Accuracy — 平衡准确率

它综合考虑 seizure 和 non-seizure 两类的识别能力。

比普通 Accuracy 更适合类别不平衡任务。

### Precision — 精确率

模型预测为 seizure 的样本里面，有多少真的 seizure。

Precision 高意味着假警报较少。

### Recall / Sensitivity — 召回率 / 敏感度

真正 seizure 的样本里面，有多少被模型成功找到。

对于 seizure detection 特别重要，因为 FN 就意味着漏掉真正的发作片段。

### Specificity — 特异度

真正 non-seizure 里面，有多少正确预测为 non-seizure。

Specificity 高意味着误报较少。

### F1-score

Precision 和 Recall 的调和平均。

你们 protocol 将 seizure-class F1 设为二分类主要指标，因为既不希望漏诊太多，也不希望误报太多。

### ROC-AUC

衡量模型在各种阈值下区分两类的总体能力。

### PR-AUC

衡量 Precision–Recall 之间的整体表现。

当 positive class 较少时，PR-AUC 通常比单纯 Accuracy 更值得关注。

### Confusion Matrix — 混淆矩阵

二分类中：

```text
TN：真正 non-seizure，预测也是 non-seizure
FP：真正 non-seizure，却预测成 seizure
FN：真正 seizure，却预测成 non-seizure
TP：真正 seizure，预测也是 seizure
```

## 11. 当前二分类正式结果

### 11.1 5-fold CV

| 指标 | 5折交叉验证 均值 ± 标准差 | 最终测试集 |
| --- | ---: | ---: |
| 准确率 | 0.9884 ± 0.0049 | 0.9857 |
| Balanced 准确率 | 0.9870 ± 0.0051 | 0.9878 |
| 精确率 | 0.9584 ± 0.0182 | 0.9402 |
| 召回率 / 敏感度 | 0.9848 ± 0.0056 | 0.9913 |
| 特异度 | 0.9893 ± 0.0048 | 0.9842 |
| F1 分数 | 0.9714 ± 0.0118 | 0.9651 |
| ROC-AUC | 0.9992 ± 0.0004 | 0.9993 |
| PR-AUC | 0.9969 ± 0.0016 | 0.9974 |

每折详细结果：

| fold | accuracy | balanced_accuracy | precision | recall_sensitivity | specificity | f1 | roc_auc | pr_auc | best_epoch | best_val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9826 | 0.9810 | 0.9375 | 0.9783 | 0.9837 | 0.9574 | 0.9988 | 0.9956 | 9 | 0.0473 |
| 1 | 0.9940 | 0.9922 | 0.9811 | 0.9891 | 0.9952 | 0.9851 | 0.9997 | 0.9987 | 19 | 0.0215 |
| 2 | 0.9918 | 0.9918 | 0.9682 | 0.9918 | 0.9918 | 0.9799 | 0.9996 | 0.9982 | 25 | 0.0221 |
| 3 | 0.9842 | 0.9830 | 0.9426 | 0.9810 | 0.9851 | 0.9614 | 0.9987 | 0.9950 | 10 | 0.0463 |
| 4 | 0.9891 | 0.9871 | 0.9628 | 0.9837 | 0.9905 | 0.9731 | 0.9990 | 0.9971 | 16 | 0.0357 |

其中最重要的 F1：

```text
F1 = 0.9714 ± 0.0118
```

平均值很高，而且 fold 间 SD 相对较小，说明在当前这套 segment-level split 下表现比较稳定。

ROC-AUC 和 PR-AUC 也都非常高，说明模型对 seizure 与 non-seizure 的概率区分能力较强。

### 11.2 Final test

最终混淆矩阵：

```text
                      Predicted
                  Non-seizure   Seizure
True Non-seizure      1811          29
True Seizure              4         456
```

也就是说：

- 1811 个非发作正确识别；
- 29 个非发作被误报成 seizure；
- 只有 4 个真正 seizure 被漏掉；
- 456 个 seizure 正确识别。

对应：

- Sensitivity = 99.13%
- Specificity = 98.42%
- Precision = 94.02%
- F1 = 0.9651
- ROC-AUC = 0.9993
- PR-AUC = 0.9974

从错误类型看：

> 当前模型宁愿多产生一些 false positive，也很少漏掉 seizure。

这是一个需要和其他模型比较的重要特征，而不能只看 Accuracy。

## 12. 怎么看 final training loss 曲线

总体趋势明显下降，说明模型还在持续降低 training error。

中间小幅上涨并不自动代表异常。

但需要注意：

> 仅凭 final training loss 不能证明完全没有 overfitting。

真正控制 overfitting 的证据来自前面的 validation loss、early stopping 和 5-fold CV。

## 13. 当前结果不能证明什么

现在的高分不能直接证明：

- 模型已经理解全部五种 EEG 状态；
- BiLSTM 一定有必要；
- CNN-BiLSTM 一定优于 MultiRocket；
- 模型一定可以泛化到全新的患者；
- 模型学习到的一定都是临床上的 seizure physiology。

尤其需要注意：

数据是从较长 EEG 记录切出来的短片段。

如果同一个 original recording 产生的不同 segment 同时进入 train 和 test，那么 segment-level split 有可能让成绩偏乐观。

只要我们目前无法可靠恢复 original recording group ID，这一点就应该作为 limitation 明确写出。

## 14. 目前 protocol 里还没有完成的部分

当前 V3 已经完成最核心的：

- shared split；
- fixed five-fold CV；
- leakage-free scaling；
- class weighting；
- early stopping；
- common metrics；
- final-test isolation。

但最终项目完整 protocol 还有：

- 多随机种子重复；
- 如果资源允许，正式 hyperparameter tuning / nested CV；
- 利用 development OOF prediction 做 threshold analysis；
- false positive / false negative 的样本级 error analysis；
- 五分类结果解释；
- 与其他模型的统一比较。

这些属于下一阶段实验，不能假装当前代码已经完成。

## 15. 以后每一个模型都建议记录的统一比较字段

| 比较项目 | 当前 CNN-BiLSTM |
| --- | --- |
| 输入表示 | Raw EEG，178 点 |
| 人工特征 | 无 |
| 自动特征学习 | 有，1D CNN |
| 显式 feature selection | 无 |
| 时间建模 | Bidirectional LSTM |
| 类别不平衡处理 | Class-weighted loss |
| 是否统一 train/test split | 是 |
| 是否统一 CV folds | 是 |
| 是否无泄漏 preprocessing | 是 |
| Binary 主指标 | Seizure-class F1 |
| Five-class 主指标 | Macro F1 |
| 可解释性 | 较弱，主要属于 learned representation |
| 核心优势 | 同时学习局部 EEG 波形和双向时间上下文 |
| 核心局限 | 复杂度高、可解释性弱，而且 segment-level split 可能使结果偏乐观 |

以后 MultiRocket、InceptionTime、XGBoost、DWT-CNN-LSTM 等模型也按照这个表写，就能很直观比较模型差异。

## 16. 当前模型输出文件分别有什么意义

```text
results_cnn_bilstm_sharedsplit/
└── full_25epochs_<timestamp>/
    ├── run_config.json
    ├── all_results.json
    ├── binary/
    │   ├── cv_fold_metrics.csv
    │   ├── results_summary.json
    │   ├── final_test_predictions.csv
    │   ├── final_test_confusion_matrix.png
    │   ├── final_training_history.csv
    │   ├── final_training_history.png
    │   ├── final_model.pt
    │   └── fold_0 ... fold_4/
    └── five/
        └── 五分类对应输出
```

具体理解：

**cv_fold_metrics.csv**  
保存每个 fold 的性能，是以后计算 mean ± SD 和模型配对比较的核心文件。

**results_summary.json**  
保存本模型最重要的 summary，包括 CV mean/SD、best epochs 和 final test metrics。

**final_test_predictions.csv**  
保存每一个 test 样本的真实标签、预测标签和概率。以后做 threshold analysis、error analysis、ensemble 都非常重要。

**final_test_confusion_matrix.png**  
直观看 FN 和 FP。

**final_training_history.csv / png**  
查看最终模型的训练 loss。

**final_model.pt**  
保存最终 PyTorch 模型权重和 scaler 参数，可以以后重新载入。

## 17. 汇报时一句话怎么介绍这个模型

**CNN-BiLSTM 首先利用 1D CNN 从原始 EEG 中自动学习局部波形特征，再通过双向 LSTM 学习这些局部模式的前后关系，并使用和全组其他模型完全一致的固定数据划分和五折交叉验证进行评价。**

## 18. 当前进度

**Binary：** 正式结果已完成并写入本文件。

**Five-class：** 等正式结果完成以后补充。

**下一步模型比较：** 不只比较 Accuracy，还需要同时比较 F1、Recall、Specificity、PR-AUC、fold 间波动、输入特征表示、可解释性、计算复杂度和错误模式。
