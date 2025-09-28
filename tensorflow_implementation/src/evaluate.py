# src/evaluate.py

import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.ndimage import distance_transform_edt

import src.config as config
from src.data_loader import get_data_generators
from src.model_parts import PAM_Module, TokenizedMLPBlock, CombinedImageProcessingLayer
from src.losses import structure_loss, focal_loss, bce_dice_loss, dice_loss
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = tf.dtypes.cast(result, tf.bool)
    reference = tf.dtypes.cast(reference, tf.bool)

    if voxelspacing is not None:
        if not isinstance(voxelspacing, list):
            voxelspacing = [voxelspacing] * result.shape.rank
        voxelspacing = tf.constant(voxelspacing, dtype=tf.float64)

    # Binary structure
    footprint = generate_binary_structure(result.shape.rank, connectivity)

    # Test for emptiness
    if tf.math.count_nonzero(result) == 0:
        return tf.constant([-1, -1], dtype=tf.float64)
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if tf.math.count_nonzero(reference) == 0:
        return tf.constant([-1, -1], dtype=tf.float64)
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # Extract only 1-pixel border line of objects
    result_border = tf.math.logical_xor(result, binary_erosion(result, structure=footprint, iterations=1))
    reference_border = tf.math.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=1))

    # Compute average surface distance
    # Note: TensorFlow's distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(tf.math.logical_not(reference_border), sampling=voxelspacing)
    sds = tf.boolean_mask(dt, result_border)

    return sds

def hd_calc(result, reference, voxelspacing=None, connectivity=1):
    hd1 = tf.reduce_max(__surface_distances(result, reference, voxelspacing, connectivity))
    if hd1 < 0:
        return -1
    hd2 = tf.reduce_max(__surface_distances(reference, result, voxelspacing, connectivity))
    if hd2 < 0:
        return -1
    hd = tf.maximum(hd1, hd2)
    return hd

def asd(result, reference, voxelspacing=None, connectivity=1):
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = tf.reduce_mean(sds)
    return asd

def assd_calc(result, reference, voxelspacing=None, connectivity=1):
    assd = tf.reduce_mean([asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)])
    return assd

def check_accuracy_new(model, image_generator, mask_generator):
    """Exact evaluation function adapted for generators."""
    iou_list, f1_list, precision_list, recall_list, acc_list, hd_list, assd_list = [], [], [], [], [], [], []
    epsilon = 1e-7

    num_images = len(image_generator)
    for i in range(num_images):
        x = image_generator[i]  # Fetches batch of 1
        y = np.around(mask_generator[i][0]) # Get the mask array
        
        preds = np.around(model.predict(x)[0])
        
        # Reshape for metrics if necessary
        preds = np.squeeze(preds)
        y = np.squeeze(y)

        hd_tmp = hd_calc(preds, y)
        if hd_tmp > 0:
            hd_list.append(hd_tmp)
            assd_list.append(assd_calc(preds, y))

        tp = tf.reduce_sum(y * preds)
        tn = tf.reduce_sum((1 - y) * (1 - preds))
        fp = tf.reduce_sum((1 - y) * preds)
        fn = tf.reduce_sum(y * (1 - preds))

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_tmp = 2 * (precision * recall) / (precision + recall + epsilon)
        acc_tmp = (tp + tn) / (tp + fp + tn + fn)
        
        intersection = np.logical_and(preds, y)
        union = np.logical_or(preds, y)
        iou_tmp = (np.sum(intersection) + epsilon) / (np.sum(union) + epsilon)
        
        iou_list.append(iou_tmp)
        f1_list.append(f1_tmp.numpy())
        precision_list.append(precision.numpy())
        recall_list.append(recall.numpy())
        acc_list.append(acc_tmp.numpy())

    return iou_list, f1_list, precision_list, recall_list, acc_list, hd_list, assd_list

if __name__ == '__main__':
    print("--- Loading Test Data Generator ---")
    test_img_gen, test_mask_gen = get_data_generators(config.TEST_IMG_DIR, config.TEST_LBL_DIR)

    print(f"--- Loading Model: {config.EVAL_MODEL_PATH} ---")
    custom_objects = {
        'PAM_Module': PAM_Module, 'TokenizedMLPBlock': TokenizedMLPBlock,
        'CombinedImageProcessingLayer': CombinedImageProcessingLayer, 'structure_loss': structure_loss,
        'focal_loss': focal_loss, 'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss
    }
    model = tf.keras.models.load_model(config.EVAL_MODEL_PATH, custom_objects=custom_objects)
    
    print("--- Starting Evaluation ---")
    iou, f1, prec, rec, acc, hd, assd = check_accuracy_new(model, test_img_gen, test_mask_gen)

    print("\n--- Evaluation Results ---")
    print(f"IOU: {np.mean(iou):.5f} (± {np.std(iou):.5f})")
    print(f"F1 Score (Dice): {np.mean(f1):.5f} (± {np.std(f1):.5f})")
    print(f"Precision: {np.mean(prec):.5f} (± {np.std(prec):.5f})")
    print(f"Recall: {np.mean(rec):.5f} (± {np.std(rec):.5f})")
    print(f"Accuracy: {np.mean(acc):.5f} (± {np.std(acc):.5f})")
    print(f"Hausdorff Distance: {np.mean(hd):.5f} (± {np.std(hd):.5f})")
    print(f"Average Surface Distance: {np.mean(assd):.5f} (± {np.std(assd):.5f})")