#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BiLSTM(nn.Module):
    def __init__(self, input_size,dropout_rate=0.5, hidden_size=128, num_layers=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size // 2, num_layers, batch_first=True, bidirectional=True)
        self.init_weights()
    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)  # 内积
            else:
                p.data.zero_()
                p.data[self.hidden_size//2:self.hidden_size] = 1
    def forward(self,x,lens):
        '''

        :param x: (batch, seq_len, input_size)
        :param len: (batch,)
        :return: (batch, seq_len, hidden_size)
        '''
        ordered_lens, index = lens.sort(descending=True)  # 句子按照长度降序排列
        ordered_x = x[index]  # (batch_size,seq_lens,embedding_dim)

        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x.to(DEVICE), ordered_lens.to(DEVICE), batch_first=True)
        packed_output,_ = self.bilstm(packed_x)
        output,_=nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]
        return recover_output

class ESIM(nn.Module):
    def __init__(self, vocab_size, num_class, embedding_dim, hidden_size, dropout_rate=0.5, num_layers=1,
                 pretrained_weights=None, freeze=False):
        super(ESIM, self).__init__()
        self.pretrained_weights = pretrained_weights
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm1 = BiLSTM(input_size=embedding_dim, hidden_size=hidden_size,dropout_rate=dropout_rate, num_layers=num_layers)
        self.bilstm2 = BiLSTM(input_size=hidden_size,hidden_size=hidden_size,dropout_rate=dropout_rate, num_layers=num_layers)
        self.fc1 = nn.Linear(4*hidden_size, hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        # 初始化embedding层参数
        if self.pretrained_weights is None:
            nn.init.normal_(self.embedding.weight)
            self.embed.weight.data.mul_(0.01)
        # 初始化三层全连接层的参数
        nn.init.normal_(self.fc1.weight)
        self.fc1.weight.data.mul_(0.01)
        nn.init.normal_(self.fc2.weight)
        self.fc2.weight.data.mul_(0.01)
        nn.init.normal_(self.fc3.weight)
        self.fc3.weight.data.mul_(0.01)

    def soft_align_attentioin(self, x1, x1_lens, x2, x2_lens):
        '''
        :param x1: (batch, seq1_len, hidden_size)
        :param x1_len: (batch,)  每个句子1的长度
        :param x2: (batch, seq2_len, hidden_size)
        :param x2_len: (batch,)  每个句子2的长度
        :return: x1_align (batch, seq1_len, hidden_size), x2_align (batch, seq2_len, hidden_Size)
        '''

        seq1_len = x1.size(1) # (batch, )
        seq2_len = x2.size(1)  # (batch, )
        batch_size = x1.size(0)
        # (batch, seq1_len, seq2_len)
        attention = torch.matmul(x1, x2.transpose(1, 2))
        # (batch,  seq_len1)
        mask1 = torch.arange(seq1_len).expand(batch_size, seq1_len).to(x1.device) >= x1_lens.unsqueeze(1)
        # (batch, seq_len2)
        mask2 = torch.arange(seq2_len).expand(batch_size, seq2_len).to(x2.device) >= x2_lens.unsqueeze(1)
        mask1 = mask1.float().masked_fill_(mask1, float('inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('inf'))
        # dim=-1: 对某一维度的行进行softmax运算
        weight1 = F.softmax(attention.transpose(1,2) + mask1.unsqueeze(1), dim = -1) # (batch, seq2_len, seq1_len)
        weight2 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # (batch, seq1_len, seq2_len)
        x1_align = torch.matmul(weight2, x2)  # (batch, seq1_len, hidden_size)
        x2_align = torch.matmul(weight1, x1)  # (batch, seq2_len, hidden_size)

        return x1_align, x2_align

    def composition(self, x, lens):
        x = F.relu(self.fc1(x))
        x_compose = self.bilstm2(self.dropout(x), lens)  # (batch, seq_len, hidden_size)
        p1 = F.avg_pool1d(x_compose.transpose(1,2), x.size(1)).squeeze(-1)  # (batch, hidden_size)
        p2 = F.max_pool1d(x_compose.transpose(1,2), x.size(1)).squeeze(-1)  # (batch, hidden_size)

        return torch.cat([p1, p2], 1)  # (batch, hidden_size*2)

    def forward(self, x1, x1_len, x2, x2_len):
        '''
        :param x1: (batch, seq1_len)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len)
        :param x2_lens: (batch,)
        :return:(batch, num_class)
        '''

        # input encoding
        embedding1 = self.embedding(x1)  # (batch, seq1_len, embedding_dim)
        embedding2 = self.embedding(x2)  #(batch, seq2_len, embedding_dim)
        embedding1 = self.dropout(embedding1)
        embedding2 = self.dropout(embedding2)
        embedding1 = self.bilstm1(embedding1, x1_len)  # (batch, seq1_len, hidden_size)
        embedding2 = self.bilstm1(embedding2, x2_len)  # (batch, seq2_len, hidden_size)

        # local inderence collocted over sequence
        x1_align, x2_align = self.soft_align_attentioin(embedding1, x1_len, embedding2, x2_len)

        # enhancement of local inference information
        # (batch, seq1_len, 4*hidden_size)
        x1_combined = torch.cat([embedding1, x1_align, embedding1 - x1_align, embedding1 * x1_align], dim=-1)
        # (batch, seq2_len, 4*hidden_size)
        x2_combined = torch.cat([embedding2, x2_align, embedding2 - x2_align, embedding2 * x2_align], dim=-1)

        # inference composition
        x1_composed = self.composition(x1_combined, x1_len)  # (batch, 2*hidden_size)
        x2_composed = self.composition(x2_combined, x2_len)  # (batch, 2*hidden_size)
        composed = torch.cat([x1_composed, x2_composed], -1)  # (batch, 4*hidden_size)
        out = self.fc3(self.dropout(torch.tanh(self.fc2(self.dropout(composed)))))
        return out