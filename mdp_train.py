
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,5,6,7"  # 必须在`import torch`语句之前设置才能生效
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net
from utils import train_dataset, val_dataset, train_one_epoch, val_one_epoch, set_random_seed

device = torch.device('cuda')
batch_size = 64
epochs = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Net().to(device)
model = nn.DataParallel(model)  # 就在这里wrap一下，模型就会使用所有的GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

tb_writer = SummaryWriter(comment='data-parallel-training')
for epoch in range(epochs):
    avg_train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
    avg_val_loss, val_accuracy = val_one_epoch(model, criterion, val_loader, device)
    print(f'Epoch {epoch} finished with average training loss: {avg_train_loss}, val loss: {avg_val_loss}, val accuracy: {val_accuracy}')
    tb_writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
    tb_writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
    tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)
tb_writer.close()