###  CNN_for_NCM-composition-and-state-prediction

### Title:
Composition and state prediction of lithium-ion cathode via convolutional neural network trained on scanning electron microscopy images
### DOI : https://doi.org/10.1038/s41524-024-01279-6


### Abstract:
High-throughput materials research is strongly required to accelerate the development of safe and high energy density lithium-ion battery (LIB) applicable to electric vehicle and energy storage system. The artificial intelligence, including machine learning with neural networks such as Boltzmann neural networks and convolutional neural networks (CNN), is a powerful tool to explore next-generation electrode materials and functional additives. In this paper, we develop a prediction model that classifies the major composition (e.g. 333, 523, 622 and 811) and different states (e.g. pristine, pre-cycled, and 100 times cycled) of various Li(Ni, Co, Mn)O2 (NCM) cathodes via CNN trained on scanning electron microscopy (SEM) images. Based on those results, our trained CNN model shows a high accuracy of 99.6% where the number of test set is 3840. In addition, the model can be applied to the case of untrained SEM data of NCM cathodes with functional electrolyte additives.

###  The dataset  used in this work will be available upon reasonable request.




### Procedure
1. Donwload the dataset, then unzipped, rename the parent folder with "source" , and put the folder in the same directory of the python codes in this repository
2. Image Augmentation: aug_img.py
    * Description:
        - parser = argparse.ArgumentParser(description='Image augmentation with desginated size and ratio')
        - parser.add_argument('path', help='folder path', type=str, default='source')
        - parser.add_argument('size', help='Size of the generated dataset', type=int)
        - parser.add_argument('-p', '--pixel', help='Size of the generated image', default=True)
        - parser.add_argument('-t', '--test', help='Ratio of test images', type=int, default=10)
        - parser.add_argument('-r', '--ratio', help='Ratio of validation images', type=int, default=20)
        - parser.add_argument('-n', '--name', help='Dataset name', default=None)
        - parser.add_argument('--seed', default=1345879)
        - parser.add_argument('-v', '--verbose', default=True)
        -  args = parser.parse_args()
    * Example line:
        - python aug_img.py source 1000 –p 224 –t 10 –r 20 –n example
          
3. Set configurations: config.py
   * You can add new configuration for training/test environment
   * If you want to add different optimizer / loss function, you can modify loss_fn / optimize_fn in utils.py
4. Train: train.py
   * Change dataset_name and version (designated version should be in config.py)
   * Run the file (python train.py)
5. Test: test.py
   * Change dataset_name and version (designated version should be in config.py)
   * Run the file (python test.py)
   * This will generate prediction results with each single entity as a csv file

### Customized dataset
1. This file includes dataset classes for train, validation, and testing.
2. If you want to have idea, you can refer to the PyTorch Dataset class documentations.












 

