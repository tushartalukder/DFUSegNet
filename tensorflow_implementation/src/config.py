# src/config.py

import os

# --- Training Parameters ---
EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 0.00005

# --- Image Dimensions ---
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3

# --- Data Paths ---
# Make sure these paths are correct for your system
BASE_PATH = "D:/tushar/WOUNDSEG/azh/split_1/"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train/images")
TRAIN_LBL_DIR = os.path.join(BASE_PATH, "train/labels")
VALID_IMG_DIR = os.path.join(BASE_PATH, "test/images")
VALID_LBL_DIR = os.path.join(BASE_PATH, "test/labels")

TEST_BASE_PATH = "D:/tushar/WOUNDSEG/azh/test/"
TEST_IMG_DIR = os.path.join(TEST_BASE_PATH, "images")
TEST_LBL_DIR = os.path.join(TEST_BASE_PATH, "labels")

# --- Model Saving & Evaluation ---
MODEL_SAVE_DIR = "D:/tushar/dfuc/models/best/"
# Path to the specific model you want to evaluate
EVAL_MODEL_PATH = "D:/tushar/dfuc/models/best/gmodel_000180.h5"