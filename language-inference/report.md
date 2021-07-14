# 基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM](https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

2. 数据集：https://nlp.stanford.edu/projects/snli/
4. 知识点：
   1. 注意力机制
   2. token2token attetnion

## 自然语言推理问题（Natural Language Inference）

### NLI概述

假设hypothesis是否能从前提premise推断得出。如以下例子，两句话之间的关系将被标记为`entailment`。

> p: *Several airlines polled saw costs grow more than expected, even after adjusting for inflation.*
>
> h: *Some of the companies in the poll reported cost increases.*

### 数据集

用于NLI任务的数据集：包含570,000个英文句子对，由多人标记的SNLI

## ESIM

ESIM的特点 ：精细的设计序列式的推断结构，考虑局部推断和全局推断。作者主要是用句子间的注意力机制(intra-sentence attention)，来实现局部的推断，进一步实现全局的推断。

ESIM由三部分组成：input encoding, local inference modeling, inference composition。比如有两个句子，$\textbf a$和$\textbf b$，其中$\textbf a$代表hypothesis，$\textbf b$代表premise。$\textbf a=(a_1, ..., a_{l_a})$，$\textbf b=(b_1, ..., b_{l_b})$，这里$\textbf a_i$  和$\textbf b_j$都是一个l维的向量，即$\textbf a_i, \textbf b_j \in\mathbb R^l$可以用预训练词嵌入初始化。目标是预测$a,b$两句话之间的逻辑关系，即标签$y$。

### Input Encoding

首先使用BiLSTM编码输入两句话a和b，使用 BiLSTM 可以学习如何表示一句话中的 word 和它上下文的关系，即在 word embedding 之后，在当前的语境下重新编码，得到新的 embeding 向量。

![截屏2021-06-03 14.08.12](/Users/guoqingyan/Library/Application Support/typora-user-images/截屏2021-06-03 14.08.12.png)

这部分的代码如下

```python
embedding1 = self.embedding(x1)  # (batch, seq1_len, embedding_dim)
embedding1 = self.dropout(embedding1)
embedding1 = self.bilstm1(embedding1, x1_len) # (batch, seq1_len, hidden_size)
embedding2 = self.dropout(embedding2)
embedding2 = self.embedding(x2)  #(batch, seq2_len, embedding_dim)
embedding2 = self.bilstm1(embedding2, x2_len) # (batch, seq2_len, hidden_size)
```

### Local Inference Modeling

这一步有助于收集单词及其上下文的局部推理。

`x1_align, x2_align = self.soft_align_attentioin(embedding1, x1_len, embedding2, x2_len)`

#### Alignment

这里使用 soft_align_attention，具体步骤如下。

1. 计算两个句子 word 之间的相似度，得到2维的相似度矩阵

   ```python
   attention = torch.matmul(x1, x2.transpose(1, 2))
   ```

$$
e_{ij}=\overline a_i^T\overline b_j
$$

2. 用得到的相似度矩阵（attention），结合 a，b 两句话，互相生成彼此相似性加权后的句子，维度保持不变。

   ![截屏2021-06-05 16.10.31](/Users/guoqingyan/Library/Application Support/typora-user-images/截屏2021-06-05 16.10.31.png)

   ```python
   weight1 = F.softmax(attention.transpose(1,2) + mask1.unsqueeze(1), 
                       dim = -1)  # (batch, seq2_len, seq1_len)
   x1_align = torch.matmul(weight2, x2)  # (batch, seq1_len, hidden_size)
   
   weight2 = F.softmax(attention + mask2.unsqueeze(1), 
                       dim=-1)   # (batch, seq1_len, seq2_len)
   x2_align = torch.matmul(weight1, x1)  # (batch, seq2_len, hidden_size)
   ```

####Enhancement of Local Inference Information

这一步是计算 a 和 align 之后的 a 的差、点积，增强序列模型，来帮助锐化元素之间的局部推理信息并且获得推理关系。

![截屏2021-06-05 16.19.47](/Users/guoqingyan/Library/Application Support/typora-user-images/截屏2021-06-05 16.19.47.png)

```python
# (batch, seq1_len, 4*hidden_size)
x1_combined = torch.cat([embedding1, x1_align, embedding1 - x1_align, 
                         embedding1 * x1_align], dim=-1)
# (batch, seq2_len, 4*hidden_size)
x2_combined = torch.cat([embedding2, x2_align, embedding2 - x2_align, 
                         embedding2 * x2_align], dim=-1)
```

### Inference Composition

再一次用 BiLSTM 提前上下文信息，同时使用 MaxPooling 和 AvgPooling 进行池化操作, 最后接一个全连接层。

```python
# (batch, seq_len, hidden_size)
x_compose = self.bilstm2(self.dropout(x), lens)
# (batch, hidden_size)
p1 = F.avg_pool1d(x_compose.transpose(1,2), x.size(1)).squeeze(-1) 
# (batch, hidden_size)
p2 = F.max_pool1d(x_compose.transpose(1,2), x.size(1)).squeeze(-1)  
```

## 数据预处理

使用`torchtext.data`来构建`Dataloader`。主要对句子进行填充（`<pad>`）处理，使得每句话长度一样，并且由于数据集中有些句子对的标签是'-'，即未被标记，这里需要去掉这些example，同时分割训练集与验证集，标记标签label与hypothesis和premise。

需要导入glove预训练词向量，对于oov的单词，从正态分布中抽取随机数。

代码文件见`dataloader.py`.

## 实验

由于训练较慢，只训练了10个epoch，超参数设置如下：

```python
BATCH_SIZE = 32
HIDDEN_SIZE = 100
EPOCHS = 20
DROPOUT_RATE = 0.5
NUM_LAYERS = 1
LEARNING_RATE = 4e-4
CLIP = 10
EMBEDDING_DIM = 50
```

训练过程准确率、loss的变化曲线如下，最终在测试集准确率达到0.812。

<table>
	<tr>         
		<td>     
			<img src=./images/train_loss.svg border=0,width="600px" height="120px">
      <center>图a(a) train_loss</center>
		</td>
		<td>
			<img src=./images/val_loss.svg border=0,width="600px" height="120px">
      <center>图1(b) val_loss</center>
		</td>
        <td>
            <img src=./images/val_acc.svg border=0,width="600px" height="120px">
        	<center>图1(c) val_acc</center>
        </td>
	</tr>
</table>


## 参考文献

[pytorch官方文档](https://pytorch-cn.readthedocs.io/zh/latest/search.html?q=)

[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038v3.pdf)

[Reasoning about Entailment with Neural Attention](https://arxiv.org/pdf/1509.06664v1.pdf)