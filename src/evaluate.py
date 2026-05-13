import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def calculate_angle_error(pred_angle, gt_angle):
    diff = abs(pred_angle - gt_angle) % 360
    return min(diff, 360 - diff)

def evaluate_image_predictions(gt_df, pred_df, distance_threshold=20.0):
    if len(pred_df) == 0:
        return 0, 0, len(gt_df), [] 
    if len(gt_df) == 0:
        return 0, len(pred_df), 0, []
        
    gt_coords = gt_df[['center_x', 'center_y']].values
    pred_coords = pred_df[['center_x', 'center_y']].values
    
    cost_matrix = np.linalg.norm(gt_coords[:, np.newaxis] - pred_coords, axis=2)
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    tp, fp, fn = 0, len(pred_df), len(gt_df)
    angle_errors = []
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if cost_matrix[gt_idx, pred_idx] <= distance_threshold:
            tp += 1
            fp -= 1
            fn -= 1
            gt_angle = gt_df.iloc[gt_idx]['angle_deg']
            pred_angle = pred_df.iloc[pred_idx]['angle_deg']
            angle_errors.append(calculate_angle_error(pred_angle, gt_angle))
            
    return tp, fp, fn, angle_errors

def main(gt_path, pred_path):
    print("Loading data...")
    print("--- Evaluation Results ---")
    print("Precision:  [Awaiting Model Predictions]")
    print("Recall:     [Awaiting Model Predictions]")
    print("F1 Score:   [Awaiting Model Predictions]")
    print("MAE Angle:  [Awaiting Model Predictions]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tube Detection Model")
    parser.add_argument("--data", type=str, required=True, help="Path to ground truth annotations.csv")
    args = parser.parse_args()
    main(args.data, None)
