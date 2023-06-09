"""
Determines Network Training Configurations. The training process uses this
module to obtain hyperparameters from versions defined for each dataset.
This version must be specified upon training, evaluating, and applying the
network. This module must be updated each time the user would like to train
on a new dataset, or with new hyperparameters.
(This is most easily accomplished in a text or code editor, and not from
within the command line.)
"""
import sys


# Generates the default configuration for a given dataset with the following
# hyperparameters. These default values will later be updated by the version
# specified in the get_config function below.
def gen_default(dataset, size, batch_size=4, lr=1e-4, epoch=40):
    default = {
        'data_dir': './data/' + dataset,
        'size': size,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'lr': lr,
        'epoch': epoch,
        'normalize': False,
        'pretrained': True,
        'criterion': 'CrossEntropyLoss',
        'seed': 42,
    }
    return default

config = {
    'example': {
        'default': gen_default('example',size=10),
        'v1': {'lr': 0.00035446559318532957}
    }
}

# The get_config function takes the default set of hyperparameters given by the gen_default
# function, and updates, or creates new, entries given the version specified.
def get_config(dataset, version):
    try:
        args = config[dataset]['default'].copy()
    except KeyError:
        print(f'dataset {dataset} does not exist')
        sys.exit(1)
    try:
        args.update(config[dataset][version])
    except KeyError:
        print(f'version {version} is not defined')
    args['name'] = dataset + '_' + version
    return args
