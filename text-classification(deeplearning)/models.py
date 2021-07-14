import torch.nn.functional as F
import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, embedding_type, vocab_size, n_class, pretrained_vectors, dropout=0.5,embedding_dim=50):
        super(CNN, self).__init__()
        self.embedding_type = embedding_type
        # ���embedding��ʼ��
        if self.embedding_type == 'random':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)  # vocab_size * embedding_dim ÿһ�б�ʾһ���ʵ�������ʾ
            # ÿ����Ҫ�ü�ά����������ʾ
        # ����gloveԤѵ����embedding���г�ʼ��
        if self.embedding_type == 'glove':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,_weight=pretrained_vectors)

        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, embedding_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(1, 100, kernel_size=(4, embedding_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(1, 100, kernel_size=(5, embedding_dim), padding=(4,0))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3*100, n_class)

    def forward(self, x):  # x.shape: (batch_size, length)  e.g.(16, 27)
        x = self.embedding(x)  # (batch_size, length, embedding_dim)   e.g.(16, 27, 50)  27���� ÿ������һ��50ά��������ʾ
        x = x.unsqueeze(1)  # (batch_size, 1,  length, embedding_dim)  unsqueeze ��Ϊ������һ��ͨ�� ���ھ��   e.g.(16, 1, 27, 50)

        x1 = self.conv1(x)  # (batch_size, 100, length-kernal_size+1, 1)  e.g.(16, 100, 25, 1)
        x1 = F.relu(x1)  # (batch_size, 100, length-kernal_size+1, 1)  e.g.(16, 100, 25, 1)
        x1 = x1.squeeze(3)  # ȥ�����һ��ͨ�� (batch_size, 100, length-kernal_size+1)  e.g. (16, 100, 25)
        # print(x1.size(2))  # length-kernal_size+1, e.g.25
        x1 = F.max_pool1d(x1, x1.size(2))  # �൱��ÿ��ͨ��ֻȡһ�����ֵ  (batch_size, 100, 1)  e.g. (16, 100, 1)
        x1 = x1.squeeze(2)  # (batch_size, 100)  e.g. (16, 100)

        # ͬ���Ĳ���
        x2 = F.relu(self.conv2(x)).squeeze(3)  # (16, 100, 24)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # (16, 100)

        x3 = F.relu(self.conv3(x)).squeeze(3)  # (16, 100, 23)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # (16, 100)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 100*3)  e.g.(16, 300)
        x = self.dropout(x)  # (batch_size, 100*3)  e.g.(16, 300)
        out = self.fc(x)  # (batch_size, n_class)  e.g. ([16, 5])
        return out

class RNN(nn.Module):
    def __init__(self, vocab_size,embedding_type,   n_class, pretrained_vectors, embedding_dim,rnn_type='rnn', USE_CUDA=False):
        super(RNN, self).__init__()
        self.embedding_type = embedding_type
        self.rnn_type = rnn_type
        self.n_class = n_class
        self.USE_CUDA = USE_CUDA

        # ���embedding��ʼ��
        if self.embedding_type == 'random':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)  # vocab_size * embedding_dim ÿһ�б�ʾһ���ʵ�������ʾ
            # ÿ����Ҫ�ü�ά����������ʾ
        # ����gloveԤѵ����embedding���г�ʼ��
        if self.embedding_type == 'glove':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,_weight=pretrained_vectors)

        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=128, batch_first=True)
            self.fc = nn.Linear(128, n_class)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=128, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(128*2, n_class)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=128, num_layers=2, bidirectional=True)
            self.fc = nn.Linear(128, n_class)
    def forward(self, x):
        # x (batch_size, length)
        batch_size, length = x.shape
        # (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)

        if self.rnn_type == "rnn":
            h0 = torch.randn(1, batch_size, 128)
            if self.USE_CUDA:
                h0 = h0.cuda()
            _, hn = self.rnn(x, h0)
        elif self.rnn_type == "lstm":
            h0 = torch.randn(2, batch_size, 128)
            c0 = torch.randn(2, batch_size,128)
            if self.USE_CUDA:
                c0 = c0.cuda()
            output, (hn, _) = self.rnn(x, (h0, c0))
        else:  # gru
            h0 = torch.randn(2, batch_size, 128)
            if self.USE_CUDA:
                h0 = h0.cuda()
            output, (hn, _)  = self.rnn(x, h0)

        out = self.fc(hn).squeeze(0)


        return out

