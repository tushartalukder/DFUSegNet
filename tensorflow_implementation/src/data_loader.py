# src/data_loader.py

import albumentations as A
import numpy as np
import tensorflow as tf
import pandas as pd
import natsort
import glob
import os
from src.utils import hed

# --- Exact Augmentations from Notebook ---
aug1 = A.HorizontalFlip(p=1.0)
aug2 = A.VerticalFlip(p=1.0)
# ... (all other augmentations aug3 to aug10 remain the same) ...
aug3 = A.OpticalDistortion(distort_limit=1.0, p=1.0)
aug4 = A.Blur(blur_limit=11, p=1.0)
aug6 = A.Rotate(limit=90, interpolation=1, border_mode=2, rotate_method='largest_box', p=1.0)
aug7 = A.Downscale(scale_min=0.25, scale_max=0.75, p=1.0)
aug8 = A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1.0)
aug9 = A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0)
aug10 = A.Affine(scale=(0.2, 0.3), translate_percent=0.2, rotate=(-30, 30), shear=(-45, 45), p=1.0)

def augment(image, mask1, n):
    """Exact augmentation selection function from notebook."""
    if n == 0: augmented = aug10(image=image, mask=mask1)
    elif n == 1: augmented = aug1(image=image, mask=mask1)
    elif n == 2: augmented = aug2(image=image, mask=mask1)
    elif n == 3: augmented = aug3(image=image, mask=mask1)
    elif n == 4: augmented = aug4(image=image, mask=mask1)
    elif n in [6, 7]: augmented = aug6(image=image, mask=mask1)
    elif n == 5: return image, mask1
    elif n == 8: augmented = aug7(image=image, mask=mask1)
    elif n == 9: augmented = aug8(image=image, mask=mask1)
    elif n == 10: augmented = aug9(image=image, mask=mask1)
    else: augmented = aug6(image=image, mask=mask1)
    
    return augmented['image'], augmented['mask']

def get_data_generators(img_dir, lbl_dir, target_size=(512, 512)):
    """Creates Keras ImageDataGenerators with batch_size=1 and shuffle=False to replicate notebook behavior."""
    imggen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    img_paths = natsort.natsorted(glob.glob(os.path.join(img_dir, '*.png')))
    lbl_paths = natsort.natsorted(glob.glob(os.path.join(lbl_dir, '*.png')))
    
    df_img = pd.DataFrame({'path': [os.path.basename(p) for p in img_paths]})
    df_lbl = pd.DataFrame({'path': [os.path.basename(p) for p in lbl_paths]})
    
    print(f"Found {len(df_img)} validated image filenames in {os.path.basename(img_dir)}.")
    
    img_gen = imggen.flow_from_dataframe(
        dataframe=df_img, directory=img_dir, x_col='path', class_mode=None,
        target_size=target_size, color_mode='rgb', shuffle=False, batch_size=1,
        interpolation='nearest'
    )
    lbl_gen = imggen.flow_from_dataframe(
        dataframe=df_lbl, directory=lbl_dir, x_col='path', class_mode=None,
        target_size=target_size, color_mode='grayscale', shuffle=False, batch_size=1,
        interpolation='nearest'
    )
    
    return img_gen, lbl_gen

def generate_real_samples(image_generator, mask_generator, n_samples, batch_idx):
    """
    Exact batch generation function using generators.
    Selects random samples and applies a deterministic augmentation based on batch index.
    """
    dataset_size = len(image_generator)
    ix = np.random.randint(0, dataset_size, n_samples)
    n_aug = batch_idx % 12  # Deterministic augmentation choice

    X1, X2, X3, X4, X5, X6 = [], [], [], [], [], []

    for i in ix:
        # Accessing generator[i] fetches the i-th batch (which is one image)
        original_image = image_generator[i][0]
        original_mask = mask_generator[i][0]

        augmented_image, augmented_mask = augment(
            original_image,
            np.around(original_mask),
            n_aug
        )
        
        X1.append(augmented_image)
        X2.append(augmented_mask)
        
        # Create deep supervision targets
        mask_512 = tf.expand_dims(np.around(augmented_mask), 0)
        mask_256 = tf.image.resize(mask_512, (256, 256))
        mask_128 = tf.image.resize(mask_512, (128, 128))
        mask_64 = tf.image.resize(mask_512, (64, 64))

        X3.append(np.reshape(hed(mask_512), (512, 512, 1)))
        X4.append(np.reshape(hed(mask_256), (256, 256, 1)))
        X5.append(np.reshape(hed(mask_128), (128, 128, 1)))
        X6.append(np.reshape(hed(mask_64), (64, 64, 1)))

    y_targets = [
        np.array(X2), np.array(X2), np.array(X2), np.array(X2), np.array(X2),
        np.array(X3), np.array(X4), np.array(X5), np.array(X6)
    ]
    
    return [np.array(X1), y_targets]