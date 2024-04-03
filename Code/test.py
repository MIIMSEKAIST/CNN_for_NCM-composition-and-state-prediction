import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from efficientnet_pytorch import EfficientNet

from config import get_config
from utils import load_test_dataset
from utils import loss_fn
from utils import optimize_fn

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',  # font family
    'font.size': 10,  # font size
    'font.weight': 'normal',  # font weight
    'axes.labelsize': 'large',  # label size
    'axes.titlesize': 'x-large',  # title size
    'xtick.labelsize': 'medium',  # x-axis tick label size
    'ytick.labelsize': 'medium',  # y-axis tick label size
    'legend.fontsize': 'medium',  # legend font size
    'lines.linewidth': 2.0,  # line width
    'figure.figsize': (4, 2),  # figure size in inches
    'figure.dpi': 100,  # figure resolution
    'savefig.dpi': 200,  # save figure resolution
    'axes.grid': False,# grid on or off
    'axes.linewidth':2.0,
    'grid.alpha': 0.5,  # grid transparency
    'grid.linestyle': '--',  # grid line style
    'text.color': 'black',  # text color
    'axes.facecolor': 'white',  # plot background color
    'figure.facecolor': 'white',  # figure background color
})



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
    y_true = []
    y_pred = []
    class_names = ['333', '333_cycled', '333_formation',' 523','523_cycled','523_formation ',
                   '622','622_cycled','622_formation','811','811_cycled',
                   '811_formation'] 

    for X_val_batch, y_val_batch, img_name in tqdm(test_loader):
        inputs, labels = X_val_batch.to(device), y_val_batch.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = test_set.idx2class[int(preds.detach().cpu())]
        labels = test_set.idx2class[int(labels.detach().cpu())]
        result.append([img_name[0], preds, labels, preds==labels])

        y_true.append(labels)
        y_pred.append(preds)

    # Convert the lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45)
    plt.show()

    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)
        
result = pd.DataFrame(result, columns=['Image Name', 'Prediction', 'Label', 'Result'])
result.index += 1
result.to_csv(f'./{root_dir}/{save_name}.csv')
print(result)
print('Result has been saved.')

