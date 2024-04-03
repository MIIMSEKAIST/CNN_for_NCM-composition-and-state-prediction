"""
This script file generates cropped images from source SEM images.
"""

import os
import sys
import random
import argparse
from PIL import Image

# Directory function
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# generate images
def generate_imgs(label, imgs, size, pixel=True, path=None, verbose=False):

    if pixel == True and verbose:
        print('Generated image size will be half of source image size.')

    save_path = make_dir(os.path.join('./', path))

    for _ in range(size):
        file_num = random.randint(0, len(imgs)- 1)
        img_name, img_format = os.path.splitext(imgs[file_num])

        image = Image.open(os.path.join('./source', label, imgs[file_num]))
        
        if pixel == 'full':
            generated_img_name = f'{label}_full_{img_name}{img_format}'
            
            ## y-axis > 895 pixel for removing information of SEM images
            image.crop((0, 0, image.size[0], image.size[1] - 65)).save(os.path.join(save_path, generated_img_name))
        
        else:
            # Set the size of generated images
            pixels = [int(image.size[0]/2), int(image.size[1]/2)] if pixel == True else [int(pixel), int(pixel)] # [size, size] / square shape

            if image.size[0]-pixels[0] < 0 and image.size[1]-pixels[1] < 0:
                print('Cropped images must be smaller than full-sized images.')
                sys.exit(1)
            
            # Generate random upper-left coordinate
            x_coord, y_coord = random.randint(0, image.size[0]-pixels[0]), random.randint(0, image.size[1]-pixels[1]-65)
            
            # Cropping
            # crop coordinates: (Left, Upper, Right, Lower)
            generated_img_name = f'{label}_x{x_coord}y{y_coord}_{img_name}{img_format}'
            image.crop((x_coord, y_coord, x_coord+pixels[0], y_coord+pixels[1])).save(os.path.join(save_path, generated_img_name))
            
    if verbose:
        print(f'saved path: {path}')


def main(args):
    args.path = os.path.join('./', args.path)
    
    # Get NCM labels from folder name with order of Ni, Co, and Mn
    labels = [labels for labels in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, labels))]

    if args.name:
        save_path = os.path.join('./data', args.name)
    else:
        p = args.pixel if args.pixel != True else 'Half'
        save_path = os.path.join(f'./data/size{args.size}_pixel{p}')
    
    for label in labels:
        # you can change image extension from jpg to anything
        imgs = [f for f in os.listdir(os.path.join(args.path, label)) if f[-3:] == 'jpg' and f[0] != '.']
        num_imgs = len(imgs)
        
        if args.verbose:
            print(f'Number of images in {label} folder: {num_imgs}')
        
        num_test = int(round(args.test / 100 * num_imgs, 0))
        num_imgs = num_imgs - num_test
        num_validate = int(round(args.ratio / 100 * num_imgs, 0 ))
        num_train = int(num_imgs - num_validate)
        
        if args.verbose:
            print(f'Number of training {label} images: {num_train}')
            print(f'Number of validation {label} images: {num_validate}')
            print(f'Number of test {label} images: {num_test}')

        # Full images without informatic area of SEM images
        if args.size == 0:
            generate_imgs(label, imgs, len(imgs), pixel='full', path=os.path.join(args.path, 'test_full'))
        else:
            # seed for the random getting same result at everytime
            seed = args.seed
            random.seed(seed)
            
            # Randomly select test images
            test_imgs = []
            for _ in range(num_test):
                test_imgs.append(imgs.pop(random.randint(0, len(imgs)-1)))

            # Randomly select validate images
            validate_imgs = []
            for _ in range(num_validate):
                validate_imgs.append(imgs.pop(random.randint(0, len(imgs)-1)))
            
            training_imgs = imgs


            # generate test images
            generate_imgs(label, test_imgs, size=int(args.size * args.ratio/100),
                        pixel=args.pixel, path=os.path.join(save_path,'test', label), verbose=args.verbose)
            
            # generate training images
            generate_imgs(label, training_imgs, int(args.size * (1-args.ratio/100)),
                        pixel=args.pixel, path=os.path.join(save_path,'train', label), verbose=args.verbose)
            
            # generate validate images
            generate_imgs(label, validate_imgs, int(args.size * args.ratio/100),
                        pixel=args.pixel, path=os.path.join(save_path, 'validate', label), verbose=args.verbose)
    if args.verbose:
        print('Done!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image augmentation with desginated size and ratio')
    parser.add_argument('path', help='folder path', type=str, default='source')
    parser.add_argument('size', help='Size of the generated dataset', type=int)
    parser.add_argument('-p', '--pixel', help='Size of the generated image', default=True)
    parser.add_argument('-t', '--test', help='Ratio of test images', type=int, default=10)
    parser.add_argument('-r', '--ratio', help='Ratio of validation images', type=int, default=20)
    parser.add_argument('-n', '--name', help='Dataset name', default=None)
    parser.add_argument('--seed', default=1345879)
    parser.add_argument('-v', '--verbose', default=True)
    args = parser.parse_args()
    
    main(args)
