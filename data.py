import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.2):
    """ Load the images and masks """
    images = sorted(glob(os.path.join(path, "*/image/*.png")))
    masks = sorted(glob(os.path.join(path, "*/mask/*.png")))

    """ Split the data """
    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the dir name and image name """
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(x)))
        name = f"{dir_name}_{os.path.splitext(os.path.basename(x))[0]}"

        """ Read the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1, y1 = augmented["image"], augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2, y2 = augmented["image"], augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3, y3 = augmented["image"], augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))
            m = (m > 127).astype(np.uint8) * 255  # Чёткое бинарное разделение

            tmp_image_name = f"{name}_{idx}.jpg"
            tmp_mask_name = f"{name}_{idx}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

def main():
    dataset_path = "D:/datasets/data_train/train"
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.2)

    print("Train:", len(train_x))
    print("Valid:", len(valid_x))

    output_train = "D:/datasets/data_train/processed/train"
    output_valid = "D:/datasets/data_train/processed/valid"

    create_dir(os.path.join(output_train, "image"))
    create_dir(os.path.join(output_train, "mask"))
    create_dir(os.path.join(output_valid, "image"))
    create_dir(os.path.join(output_valid, "mask"))

    augment_data(train_x, train_y, output_train, augment=True)
    augment_data(valid_x, valid_y, output_valid, augment=False)

if __name__ == "__main__":
    main()
