import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5,6,7"
import torch
import torch.optim as optim
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model import Net
from utils import train_dataset, val_dataset, train_one_epoch, val_one_epoch, set_random_seed


def main():
    epochs = 20
    batch_size = 64
    set_random_seed(42)

    use_ddp = 'LOCAL_RANK' in os.environ

    if use_ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if local_rank == 0:
        tb_writer = SummaryWriter(comment='ddp-3' if use_ddp else 'single-gpu')

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        avg_train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        
        if local_rank == 0:
            avg_val_loss, val_accuracy = val_one_epoch(model, criterion, val_loader, device)
            print(f'Epoch {epoch} finished with average training loss: {avg_train_loss}, val loss: {avg_val_loss}, val accuracy: {val_accuracy}')
            tb_writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
            tb_writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
            tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)

    if local_rank == 0:
        tb_writer.close()

if __name__ == "__main__":
    main()
