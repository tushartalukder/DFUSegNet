# src/train.py

import numpy as np
import tensorflow as tf
from datetime import datetime

import src.config as config
from src.model import define_generator
from src.data_loader import get_data_generators, generate_real_samples
from src.utils import summarize_performance, remove_prev_performance

def train(g_model, train_img_gen, train_mask_gen, valid_img_gen, valid_mask_gen, test_img_gen, test_mask_gen):
    """Exact training and evaluation loop using generators."""
    n_epochs = config.EPOCHS
    n_batch = config.BATCH_SIZE
    
    # Number of training batches per epoch
    bat_per_epo = len(train_img_gen) // n_batch
    
    print('--- Starting Training ---')
    score_max = 0
    score_epoch = 0

    for j in range(n_epochs):
        start_time = datetime.now()
        gloss = []
        for i in range(bat_per_epo):
            # Generate a batch of real samples with cycling augmentations
            [X_realA, y_targets] = generate_real_samples(train_img_gen, train_mask_gen, n_batch, i)
            
            # Update generator model
            g_loss_list = g_model.train_on_batch(X_realA, y_targets)
            gloss.append(g_loss_list[0]) # Total loss

        # --- Performance Summary at End of Epoch ---
        glossm = np.mean(gloss)

        # Evaluate on validation set
        iou_scorev1, dice1 = [], []
        for i in range(len(valid_img_gen)):
            output = valid_img_gen[i] # This is already shape (1, 512, 512, 3)
            gen_image = np.around(g_model.predict(output)[0])
            tar_image = np.around(valid_mask_gen[i])
            
            intersection = np.logical_and(gen_image, tar_image)
            union = np.logical_or(gen_image, tar_image)
            iou = (np.sum(intersection) + 1e-6) / (np.sum(union) + 1e-6)
            iou_scorev1.append(iou)
            dice1.append((2. * iou) / (1. + iou))
        
        iou_scorev = np.mean(iou_scorev1)
        dice_score_val = np.mean(dice1)

        print(f'Epoch {j+1}> g[{glossm:.3f}] valid_iou:[{iou_scorev:.5f}] valid_dice:[{np.mean(dice1):.5f}]')
        
        # Checkpoint the best model
        if dice_score_val >= score_max:
            print(f'Best model found! Val Dice improved from {score_max:.5f} to {dice_score_val:.5f}')
            remove_prev_performance(score_epoch, config.MODEL_SAVE_DIR)
            score_max = dice_score_val
            score_epoch = j + 1
            summarize_performance(j, g_model, config.MODEL_SAVE_DIR)
        
        print(f"Time for epoch: {datetime.now() - start_time}")

if __name__ == '__main__':
    print("--- Setting up Data Generators ---")
    train_img_gen, train_mask_gen = get_data_generators(config.TRAIN_IMG_DIR, config.TRAIN_LBL_DIR)
    valid_img_gen, valid_mask_gen = get_data_generators(config.VALID_IMG_DIR, config.VALID_LBL_DIR)
    test_img_gen, test_mask_gen = get_data_generators(config.TEST_IMG_DIR, config.TEST_LBL_DIR)

    g_model = define_generator(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH, n_channels=config.CHANNELS)
    # g_model.summary()

    print("TensorFlow version:", tf.__version__)
    print("Is GPU available?", tf.config.list_physical_devices('GPU'))

    train(g_model, train_img_gen, train_mask_gen, valid_img_gen, valid_mask_gen, test_img_gen, test_mask_gen)