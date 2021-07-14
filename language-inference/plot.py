from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf

file = open('train_log', 'r')
tf.gfile.DeleteRecursively('./logs')
writer = SummaryWriter('logs', comment='ESIM')
i = 0
j=0
for line in file.readlines():
    line = line.strip().split(' ')
    if(line[2] == 'Train'):
        writer.add_scalar('train_loss', float(line[-1]), i)
        i+=1
    else:
        writer.add_scalar('val_loss', float(line[-1]), j)
        writer.add_scalar('val_acc', float(line[-3][:-1]), j)
        j+=1