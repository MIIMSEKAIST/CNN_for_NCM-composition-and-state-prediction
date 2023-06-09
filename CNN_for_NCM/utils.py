import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from statistics import mean
from tqdm import tqdm
from PIL import Image
from glob import glob

from datasets import SEMDataset

# Calculate mean and std for normalization
def calculate_statistics(folder):
    avg, std = list(), list()
    for img in tqdm(folder):
        img = Image.open(img)
        tensor = TF.to_tensor(img)
        tensor = tensor.float()
        avg.append(torch.mean(tensor).tolist())
        std.append(torch.std(tensor).tolist())
    avg = [round(mean(avg), 3)] * 3
    std = [round(mean(std), 3)] * 3
    return avg, std

# load data to Dataset class
def load_datasets(config):
    data_dir = config['data_dir']
    if config['normalize']:
        train_avg, train_std = calculate_statistics(glob(f'./{data_dir}/train/*/*.jpg'))
        train_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=train_avg, std=train_std)
        ])
        
        validate_avg, validate_std = calculate_statistics(glob(f'./{data_dir}/validate/*/*.jpg'))
        validate_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=validate_avg, std=validate_std)
        ])
        
        train_set = SEMDataset(data_dir+'/train', train_img_transforms, seed = config['seed'])
        val_set = SEMDataset(data_dir+'/validate', validate_img_transforms, seed = config['seed'])
        
    else:
        train_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        train_set = SEMDataset(data_dir+'/train', train_img_transforms, seed = config['seed'])
        validate_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        val_set = SEMDataset(data_dir+'/validate', validate_img_transforms, seed = config['seed'])
        
    return train_set, val_set

def load_test_dataset(config):
    data_dir = config['data_dir']
    if config['normalize']:
        test_avg, test_std = calculate_statistics(glob(f'./{data_dir}/test/*/*.jpg'))
        test_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=test_avg, std=test_std)
        ])
        
        test_set = SEMDataset(data_dir+'/train', test_img_transforms, seed = config['seed'])
        
    else:
        test_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        test_set = SEMDataset(data_dir+'/train', test_img_transforms, seed = config['seed'])
 
    return test_set

def loss_fn(config):
    # you can add different loss function
    if config['criterion'] == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    
def optimize_fn(config, parameters):
    # you can add different optimizer
    if config['optimizer'] == 'Adam':
        return torch.optim.Adam(parameters, config['lr'])
    else:
        raise NotImplementedError
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def class_acc(y_pred, y_test):
    y_pred_tag = torch.softmax(y_pred, dim=1)
    y_pred_tags = torch.argmax(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc