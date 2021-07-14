# 基于深度学习的文本分类

实现CNN、RNN的文本分类；

1. word embedding、随机embedding 的方式初始化

2. 用glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/

3. 知识点：CNN/RNN的特征抽取，词嵌入，Dropout

## 数据处理

### 数据集划分

首先对数据集进行划分，并存储，避免每一次都要重新划分。

```python
Size of trainset: 124848
Size of testset: 15606
Size of devset: 15606
```

### Dataloader构建

函数：

```python
def Dataloader(batch_size):
    ...
return train_iter, dev_iter, test_iter, TEXT.vocab.vectors, LABEL.vocab
```

* 读取数据，不需要的field设置为None

  ```python
  fields = [("PhraseId", None),("SentenceId", None), ('Phrase', TEXT),
            ('Sentiment', LABEL)]
  train_data = TabularDataset(path='./data/split_train.csv', 
                              format='csv',fields=fields, skip_header=True)
  dev_data = TabularDataset(path='./data/split_dev.csv', 
                            format='csv', fields=fields,skip_header=True)
  test_data = TabularDataset(path='./data/split_test.csv', 
                             format='csv',fields=fields,skip_header=True,)
  ```

* 构建字典，将字符映射到embedding

  ```python
  TEXT.build_vocab(train_data, vectors='glove.6B.50d')
  TEXT.vocab.vectors.unk_init = init.xavier_uniform
  LABEL.build_vocab(train_data)
  ```

* 根据batch_size构建数据迭代器

  ```python
  train_iter = BucketIterator(train_data, batch_size=batch_size,
                              train=True, shuffle=True, 
                              device=torch.device('cpu'))
  ```



## 初始化方式

### 随机初始化

* 代码实现：这里`embedding_dim`表示每个词要用多少维的向量去表示

  ```python
  if self.embedding_type == 'random':
      self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                    embedding_dim=embedding_dim)
  ```

### GloVe embedding初始化

GloVe模型使用了语料库的全局统计（overall statistics）特征，也使用了局部的上下文特征（即滑动窗口）。GloVe模型引入了共现概率矩阵，有以下定义，这里首先介绍共现矩阵$X$。

1. $X$的元素$X_{ij}$是语料库中出现在word $i$ 上下文中的word $j$ 的次数

2. $X_i=\Sigma_k X_{ik}$表示出现在单词 $i$ 上下文中的所有word的总次数
3. $P_{ij}=P(j|i)=\frac{X_{ij}}{X_i}$ 表示word $j$ 出现在word $i$ 上下文的概率
4. $Ratio_{i,j,k}=\frac {P_{i,k}}{P_{j,k}}$表示两个条件概率的比率

很容易得到$Ratio_{i,j,k}$的值有以下规律：

| $Ratio_{i,j,k}$的值 | 单词j,k相关 | 单词j,k不相关 |
| ------------------- | ----------- | ------------- |
| 单词i,k相关         | 趋近1       | 很大          |
| 单词i,k不相关       | 很小        | 趋近1         |

接下来我们想获取每一个单词的向量表示v，并且用词向量$v_i, v_j, v_k$通过某种函数可以计算得出Ratio值，这就说明词向量中包含了共现矩阵所蕴含的信息。假设用词向量$v_i, v_j, v_k$计算Ratio值的函数为$g(v_i, v_j, v_k)$，那么
$$
Ratio_{i,j,k}=\frac {P_{i,k}}{P_{j,k}}=g(v_i, v_j, v_k)
$$
考虑单词i与单词j之间的关系，函数中应该有一项$v_i-v_j$，又因为结果是标量，所以应该使用内积的形式，得到以下公式：
$$
\frac {P_{i,k}}{P_{j,k}}=exp((v_i-v_j)^Tv_k)=\frac {exp(v_i^Tv_k)}{exp(v_j^Tv_k)}
$$
使分子分母分别相等，就可以统一形式，现在只需要使得以下式子成立：
$$
P_{i,j}=exp(v_i^Tv_j)\\
log(P_{i,j})=v_i^Tv_j\\
log(P_{j,i})=v_j^Tv_i
$$
现在出现了一个问题：$log(P_{i,j})$不等于$log(P_{j,i})$，但是$v_i^Tv_j$却等于$v_j^Tv_i$，即不满足对称性，现在将条件概率展开，其中$b_i$中包含了$log(X_i)$
$$
log(P_{i,j})=log(X_{i,j})-log(X_i)=v_i^Tv_j\\
log(X_{i,j})=v_i^Tv_j+b_i+b_j
$$
于是就产生了代价函数：
$$
J=\sum_{i,j}^N(v_i^Tv_j+b_i+b_j-log(X_{i,j}))^2
$$
出现频率越高的词对权重应该越大，需要在代价函数中添加权重项，于是代价函数进一步完善： 
$$
J=\sum_{i,j}^Nf(X_{i,j})(v_i^Tv_j+b_i+b_j-log(X_{i,j}))^2
$$

* 代码实现，这里给`weight`预训练的glove embedding

  ```python
  self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=embedding_dim,
                                _weight=pretrained_vectors)
  ```

## 网络结构

### CNN

CNN的核心思想是**捕捉局部特征**，对于文本来说，局部特征就是**由若干单词组成的滑动窗口**，类似于N-gram。CNN的优势在于能够**自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息**。

（1）第一层为输入层。输入层是一个 ![[公式]](https://www.zhihu.com/equation?tex=n+%5Ctimes+k) 的矩阵，其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 为一个句子中的单词数， ![[公式]](https://www.zhihu.com/equation?tex=k) 是每个词对应的词向量的维度。输入层的每一行就是一个单词所对应的 ![[公式]](https://www.zhihu.com/equation?tex=k) 维的词向量。另外，这里为了使向量长度一致对原句子进行了padding操作。比如图中的输入是$7*5$的矩阵，一共有7个单词，每个单词用5维的向量表示。

（2）第二层为卷积层，第三层为池化层。

这里卷积操作与CV有所不同，在CV中，卷积核往往都是正方形的，比如 ![[公式]](https://www.zhihu.com/equation?tex=3+%5Ctimes+3) 的卷积核，然后卷积核在整张图像上沿高和宽按步长移动进行卷积操作。与而这里**输入层是一个由词向量拼成的词矩阵，且卷积核的宽和该词矩阵的宽相同，该宽度即为词向量大小，即卷积核只会在高度方向移动**。因此，每次卷积核滑动过的位置都是完整的单词，不会将几个单词的一部分"vector"进行卷积，这就保证了单词作为语言中最小粒度的合理性。

图中一共有三个不同尺度的卷积核，每个卷积核有两个通道，将卷积层的输出经过激活层然后输入到池化层进行拼接，最后经过softmax分类。

![截屏2021-05-20 21.17.49](/Users/guoqingyan/Library/Application Support/typora-user-images/截屏2021-05-20 21.17.49.png)

我在实验中使用的三个卷积核大小分别是3\*dim，4\*dim，5\*dim

```python
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, embedding_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(1, 100, kernel_size=(4, embedding_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(1, 100, kernel_size=(5, embedding_dim), padding=(4,0))
```

### RNN

#### 标准RNN

如图所示为RNN的网络结构，其中$U,V,W$都为权重矩阵，$s_t$表示t时刻隐藏层的输出，$x_t$表示t时刻的输入，$o_t$表示t时刻网络的输出，这个网络在t时刻接收到输入$x_t$之后，隐藏层的值是 $s_t$，输出值是$o_t$。关键一点是，$s_t$的值不仅仅取决于$x_t$，还取决于$s_{t-1}$。

![截屏2021-05-13 20.11.58](/Users/guoqingyan/Library/Application Support/typora-user-images/截屏2021-05-13 20.11.58.png)

计算公式：其中$\sigma()$表示激活函数，通常RNN用于分类，所以一般为softmax函数
$$
S_t=f(U·X_t+W·S_{t-1})\\
O_t=g(V·S_t) \\
\hat y^{(t)}=\sigma (O_t)
$$

* 代码实现：

```python
self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, n_class)
```

#### LSTM

随着距离的增加，标准RNN无法有效的利用历史信息。长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。LSTM的网络结构如图，LSTM由遗忘门、输入门和输出门这三个门来控制细胞状态。本质上是细胞状态是经输入、遗忘门的产物，经过输出门得到h，就是想输出什么内容给下一个单元。

![](https://img-blog.csdnimg.cn/20200711145439953.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_1111,size_16,color_FFFFFF,t_70)

(1) 决定细胞状态需要丢弃哪些信息，通过遗忘门的sigmoid单元处理，它通过查看$h_{t-1}$和$x_t$的信息来输出一个0-1之间的向量，向量中的0-1值决定细胞状态$C_{t-1}$中 哪些信息保留或丢弃
$$
f_t=\sigma(W_f·[h_{t-1}, x_t]+b_f)
$$
(2)决定给细胞状态添加哪些新信息

* 利用$h_{t-1}$和$x_t$通过输入门的操作决定更新哪些信息；

* 利用$h_{t-1}$和$x_t$通过一个$tanh$层得到新的候选细胞信息$\tilde C_t$，这些信息可能会被更新到细胞信息中。
  $$
  i_t=\sigma (W_i·[h_{t-1}, x_t]+b_i)\\
  \tilde C_t=tanh(W_C·[h_{t-1}, x_t]+b_C)
  $$

(3) 将旧的细胞信息$C_{t-1}$更新为新的细胞信息$C_t$，更新规则为：通过遗忘门选择忘记旧细胞信息的一部分，然后通过输入门添加候选细胞信息$\tilde C_t$的一部分得到新的细胞信息$C_t$
$$
C_t=f_t*C_{t-1}+i_t*\tilde C_t
$$
(4) 更新完细胞状态后需要根据输入的$h_{t-1}$和$x_t$来判断输出细胞的哪些状态特征。经过被称为输出门的sigmoid层得到判断条件，然后将细胞状态经过tanh层得到一个[−1,1]间的向量，该向量与输入门得到的判断条件相乘即得到最终RNN单元的输出
$$
o_t=\sigma(W_o[h_{t-1}, x_t]+b_o)\\
h_t=o_t*tanh(C_t)
$$
代码实现：

```python
self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                   batch_first=True, bidirectional=True)
self.fc = nn.Linear(hidden_size*2, n_class)
```

#### GRU

GRU是LSTM网络的一种变体，它较LSTM结构更加简单，而且效果也很好。在LSTM中引入了三个门函数：输入门、遗忘门和输出门来控制输入值、记忆值和输出值。而在GRU模型中只有两个门：**更新门**和**重置门**。

具体结构如下图所示，图中的$z_t$和$r_t$分别表示更新门和重置门。

* 更新门：控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。

* 重置门：控制前一状态有多少信息被写入到当前的候选集 ℎ̃𝑡上，重置门越小，前一状态的信息被写入的越少。

* 前向传播公式：

  * 得到门控信号：

  $$
  r_t=\sigma(W_r·[h_{t-1},w_t])\\
  z_t=\sigma(W_z·[h_{t-1},w_t])\\
  $$

  * 首先使用重置门控得到重置之后的数据$h_{t-1}'=h_{t-1}\odot r_t$，再将$h_{t-1}'$经过与输入$x_t$进行拼接，经过一个tanh函数将数据放缩到[-1,1]范围内，这里的$\tilde h_t$主要是包含了当前输入的$x_t$数据。对$\tilde h_t$添加到当前的隐藏状态，相当于记忆了当前时刻的状态。类似于LSTM的选择记忆阶段。

  $$
  \tilde h_t=tanh(W_{\tilde h_t}·[r_t\odot h_{t-1},x_t])
  $$

  * 更新步骤，门控信号（这里的z )的范围为0~1。门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。

  $$
  h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde h_t
  $$

* 几个公式详解：
  * $h_t=(1-z_t)\odot h_{t-1}$：表示对原本隐藏状态的选择性“遗忘”。这里的 1-z可以想象成遗忘门，忘记 $h_{t-1}$维度中一些不重要的信息。
  * $z_t\odot \tilde h_t$：表示对包含当前节点信息的$\tilde h_t$进行选择性"记忆"。对$\tilde h_t$维度中的某些信息进行选择。
  * $h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde h_t$: 忘记传递下来的$h_{t-1}$中的某些维度信息，并加入当前节点输入的某些维度信息。

![](https://images2018.cnblogs.com/blog/1335117/201807/1335117-20180727095108158-462781335.png)

代码实现：

```python
self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, 
                  num_layers=2, bidirectional=True)
self.fc = nn.Linear(hidden_size, n_class)
```

## 实验结果

默认超参数设置：

| Batch size | Learning rate | Epoch |
| ---------- | ------------- | ----- |
| 128        | 0.001         | 50    |

### CNN

(1) 如图所示为random嵌入初始化，Embedding_dim=50，Dropout=0.3，训练过程中loss与训练集、测试集准确率的变化，可以看到测试集准确率在到达66%之后就很难有提升，而训练集准确率达到了90%左右，过拟合较严重。

<table>
	<tr>         
		<td>     
			<img src=./code/images/train_acc2.svg border=0,width="600px" height="120px">
      <center>图1(a) train_acc</center>
		</td>
		<td>
			<img src=./code/images/val_acc2.svg border=0,width="600px" height="120px">
      <center>图1(b) val_acc</center>
		</td>
        <td>
            <img src=./code/images/train_acc2.svg border=0,width="600px" height="120px">
        	<center>图1(c) loss</center>
        </td>
	</tr>
</table>


(2) 如图所示为glove嵌入初始化，Embedding_dim=100，Dropout=0.5，训练过程中loss与训练集、测试集准确率的变化，最终训练集的准确率达到95%，但是测试集准确率却和random嵌入初始化差不多，但是在第5轮的时候测试集准确率就达到了最大值，而random初始化在第15轮的时候达到最大值。

<table>
	<tr>         
		<td>     
			<img src=./code/images/train_acc.svg border=0,width="600px" height="120px">
      <center>图2(a) train_acc</center>
		</td>
		<td>
			<img src=./code/images/val_acc.svg border=0,width="600px" height="120px">
      <center>图2(b) val_acc</center>
		</td>
        <td>
            <img src=./code/images/train_loss.svg border=0,width="600px" height="120px">
        	<center>图2(c) loss</center>
        </td>
	</tr>
</table>


(3) 其他参数设置

| Embedding | Embedding_dim | Dropout | Epoch | Max train_acc | Max val_acc |
| --------- | ------------- | ------- | ----- | ------------- | ----------- |
| random    | 50            | 0.3     | 50    | 89.77%        | 66.70%      |
| random    | 50            | 0.5     | 50    | 85.51%        | 66.80%      |
| random    | 100           | 0.5     | 50    | 90.05%        | 66.88%      |
| glove     | 50            | 0.3     | 30    | 90.66%        | 64.56%      |
| glove     | 50            | 0.5     | 100   | 95.48%        | 67.06%      |
| glove     | 100           | 0.5     | 100   | 97.73%        | 67.03%      |
| glove     | 200           | 0.5     | 50    | 96.56%        | **67.47%**  |

从训练情况来看，random初始化虽然在刚开始的时候准确率低于glove初始化的结果，但是随着训练轮次的增加，准确率可以达到和glove差不多，

### RNN

(1) 如图所示为glove嵌入初始化，Embedding_dim=100，Dropout=0.5，learning_rate=0.0005，RNN类型为lstm的训练过程中loss与训练集、测试集准确率的变化情况。

<table>
	<tr>         
		<td>     
			<img src= ./code/images/train_acc3.svg border=0,width="600px" height="120px">
      <center>图3(a) train_acc</center>
		</td>
		<td>
			<img src=./code/images/val_acc3.svg border=0,width="600px" height="120px">
      <center>图3(b) val_acc</center>
		</td>
        <td>
            <img src=./code/images/train_loss3.svg border=0,width="600px" height="120px">
        	<center>图3(c) loss</center>
        </td>
	</tr>
</table>


(2) 其他参数设置

| Embedding | RNN type | embedding_dim | Dropout | Max train_acc | Max val_acc |
| --------- | -------- | ------------- | ------- | ------------- | ----------- |
| glove     | rnn      | 50            | 0.5     | 54.57%        | 52.39%      |
| glove     | lstm     | 50            | 0.5     | 53.57%        | 51.20%      |
| glove     | gru      | 50            | 0.5     | 51.35%        | 50.15%      |
| glove     | lstm     | 100           | 0.5     | **76.43%**    | **65.35%**  |
| glove     | gru      | 100           | 0.5     | 51.88%        | 50.65%      |
| random    | rnn      | 50            | 0.5     | 52.17%        | 50.28%      |
| random    | lstm     | 50            | 0.5     | 51.98%        | 50.31%      |
| random    | gru      | 50            | 0.5     | 51.37%        | 50.16%      |
| random    | lstm     | 100           | 0.5     | 52.75%        | 50.39%      |

### 实验分析

1. 在CNN中，过拟合比较严重，训练集的准确率一直在增加，而测试集准确率一开始增加，后来却逐渐下降。整体情况CNN的效果好于RNN。

2. 在RNN当中，当嵌入维度是100的时候，lstm表现显著好于其他的情况。

3. 对于dropout对实验结果的影响，经过查阅资料，dropout取值一般不大于0.6，从实验结果来看，dropout取0.5的时候效果比较好。

## 参考

[一文搞懂RNN（循环神经网络）基础篇](https://zhuanlan.zhihu.com/p/30844905)

[Tensorboard Usage](https://www.tensorflow.org/tensorboard/get_started)

## 训练环境

```python
Single Tesla-V100
cuda==10.1
tensorboard==2.4.1
tensorboard-plugin-cit==1.7.0
cudatoolkit==10.1.243
torch==1.7.1
```

