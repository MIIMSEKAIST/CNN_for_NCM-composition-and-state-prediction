import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from efficientnet_pytorch import EfficientNet

from config import get_config
from utils import load_test_dataset
from utils import loss_fn
from utils import optimize_fn

dataset_name = 'example'
version = 'v1'
config = get_config(dataset_name, version)
save_name = dataset_name + '_' + version

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
seed=123

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We're using =>", device)

root_dir = config['data_dir']
print("The data lies here =>", root_dir)

test_set = load_test_dataset(config)

test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=1)

if config['pretrained']:
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=12)  # Pre-trained
else:
    model = EfficientNet.from_name('efficientnet-b7', num_classes=12)  # from model

model.to(device)
model.load_state_dict(torch.load(f'./{root_dir}/{save_name}.pth', map_location=device)['model_state_dict'])

criterion = loss_fn(config)
optimizer = optimize_fn(config, model.parameters())

# Test
with torch.no_grad():
    model.eval()
    result = list()

    for X_val_batch, y_val_batch, img_name in tqdm(test_loader):
        inputs, labels = X_val_batch.to(device), y_val_batch.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = test_set.idx2class[int(preds.detach().cpu())]
        labels = test_set.idx2class[int(labels.detach().cpu())]
        result.append([img_name[0], preds, labels, preds==labels])
        
result = pd.DataFrame(result, columns=['Image Name', 'Prediction', 'Label', 'Result'])
result.index += 1
result.to_csv(f'./{root_dir}/{save_name}.csv')
print(result)
print('Result has been saved.')