#coding=utf8
from torchtext.vocab import Vectors
import dataloader
import torch.optim as optim
from model.esim import  ESIM
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf


BATCH_SIZE = 32
HIDDEN_SIZE = 100
EPOCHS = 20
DROPOUT_RATE = 0.5
NUM_LAYERS = 1
LEARNING_RATE = 4e-4
CLIP = 10
EMBEDDING_DIM = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VECTORS = Vectors('glove.6B.50d.txt', './weight/glove.6B')
DATA_DIR = './snli_1.0'
tf.gfile.DeleteRecursively('./logs')  # 删除之前的记录

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(data_iter, epoch, use_cache=False):
    if use_cache:
        model.load_state_dict(torch.load('best_model.ckpt'))
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label

            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != batch.label).sum().item()

    acc = correct_num / (correct_num + err_num)
    tqdm.write(
            "Epoch: %d, Test Acc: %.3f, Loss %.3f" % (epoch + 1, acc, total_loss))
    return acc

def train(train_iter, dev_iter, loss_func, optimizer, epochs, clip=5):

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        writer = SummaryWriter('logs', comment='ESIM')
        total_loss = 0
        i = 0
        for batch in tqdm(train_iter):
            premise, premise_lens = batch.premise
            # print(premise)
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label

            model.zero_grad()
            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            batches_done = epoch * len(train_iter) + i  # len(train_iter): 所有的batch个数
            writer.add_scalar('train_loss', loss, batches_done)
        tqdm.write("Epoch: %d, Train Loss: %.3f" % (epoch + 1, total_loss))

        acc = eval(dev_iter, epoch)
        writer.add_scalar('val_acc', acc, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.ckpt')


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT, LABEL = dataloader.Dataloader(batch_size=BATCH_SIZE, device=DEVICE, data_path=DATA_DIR, vectors=VECTORS)
    model = ESIM(vocab_size=len(TEXT.vocab), num_class=len(LABEL.vocab.stoi),
                 embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, dropout_rate=DROPOUT_RATE,
                 num_layers=NUM_LAYERS,pretrained_weights=TEXT.vocab.vectors).to(DEVICE)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()


    train(train_iter, dev_iter, loss_func, optimizer, EPOCHS,CLIP)
    # eval(test_iter, use_cache=True)