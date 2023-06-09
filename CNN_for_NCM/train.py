import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from config import get_config
from utils import load_datasets
from utils import loss_fn
from utils import optimize_fn

dataset_name = 'example'
version = 'v1'
config = get_config(dataset_name, version)
save_name = dataset_name + '_' + version

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

root_dir = config['data_dir']
print("The data lies here =>", root_dir)

train_set, validate_set = load_datasets(config)

train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=config['batch_size'])
val_loader = DataLoader(dataset=validate_set, shuffle=False, batch_size=config['batch_size'])


if config['pretrained']:
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=12)  # Pre-trained
else:
    model = EfficientNet.from_name('efficientnet-b7', num_classes=12)  # from model

model.to(device)

criterion = loss_fn(config)
optimizer = optimize_fn(config, model.parameters())

accuracy_stats = {
    "train": [],
    "val": []
}
loss_stats = {
    "train": [],
    "val": []
}

print("Begin training.")
acc_train = 0

for e in range(1, config['epoch'] + 1):
    num_cnt = 0
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch, _ in tqdm(train_loader):

        inputs, labels = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

        num_cnt = len(labels)

        train_epoch_loss += float(train_loss.item() * inputs.size(0) / num_cnt)
        train_corr = torch.sum(preds == labels.data)

        train_epoch_acc += float((train_corr.double() / num_cnt).cpu() * 100)
    
    # VALIDATION
    with torch.no_grad():
        model.eval()
        val_epoch_loss = 0
        val_epoch_acc = 0
        for X_val_batch, y_val_batch, _ in tqdm(val_loader):
            inputs, labels = X_val_batch.to(device), y_val_batch.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  
            val_loss = criterion(outputs, labels)

            num_cnt = len(labels)

            val_epoch_loss += float(val_loss.item() * inputs.size(0) / num_cnt)
            val_corr = torch.sum(preds == labels.data)

            val_epoch_acc += float((val_corr.double() / num_cnt).cpu() * 100)
            
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
    print(f'\n Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    
    # save the model
    if acc_train < train_epoch_acc/len(train_loader):
        print('Accuracy improved from {} to {}. Model saved.'.format(
            acc_train, train_epoch_acc/len(train_loader)))
        acc_train = train_epoch_acc/len(train_loader)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
        }, f'./{root_dir}/{save_name}.pth')


train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(
    id_vars=['index']).rename(columns={"index": "epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(
    id_vars=['index']).rename(columns={"index": "epochs"})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

sns.lineplot(data=train_val_acc_df, x="epochs", y="value",
             hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x="epochs", y="value",
             hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

plt.savefig(f'./{root_dir}/{save_name}.png', dpi=300)