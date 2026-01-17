import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import argparse
import sys
import glob
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config.parser import YAMLParser

def L2_dis(x, y):
    return np.sum((x - y) ** 2)

def NN_search(x, center_dict, candidate_classes):
    min_dist = float('inf')
    pred_label = ''
    for cls in candidate_classes:
        if cls not in center_dict:
            continue
        center = center_dict[cls]
        dist = L2_dis(x, center)
        if dist < min_dist:
            min_dist = dist
            pred_label = cls
    return pred_label

def load_resnet_data(res_path, classes_txt_path):
    print(f'Loading ResNet features from: {res_path}')
    if not os.path.exists(res_path):
        print(f'Error: Cannot find {res_path}')
        sys.exit()
    mat_content = sio.loadmat(res_path)
    features = mat_content['features']
    if features.shape[0] == 2048:
        features = features.T
    labels = mat_content['labels']
    id_to_name = {}
    print(f'Loading class mapping from: {classes_txt_path}')
    with open(classes_txt_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                cid = int(parts[0])
                cname = parts[1]
                id_to_name[cid] = cname
            else:
                id_to_name[idx + 1] = parts[0]
    return (features, labels, id_to_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='base config file')
    parser.add_argument('--suffix', type=str, default='', help='filter model files by suffix')
    parser.add_argument('--method', type=str, default='SGCN', help='filter model files by suffix')
    args = parser.parse_args()
    config_parser = YAMLParser(args.config)
    config = config_parser.config
    out_dir = config['data']['out_dir']
    method = args.method
    train_list_path = config['data']['train_class_list']
    test_list_path = config['data']['test_class_list']
    res_path = config['data'].get('resnet_path')
    if not res_path:
        res_path = '/media/da/res101.mat'
    classes_path = config['data']['classes_dir']
    all_features, all_labels, id_to_name = load_resnet_data(res_path, classes_path)
    print(f'>>> ResNet Data Loaded. Shape: {all_features.shape}')
    with open(train_list_path, 'r') as f:
        seen_classes = [x.strip() for x in f.readlines() if x.strip()]
    with open(test_list_path, 'r') as f:
        unseen_classes = [x.strip() for x in f.readlines() if x.strip()]
    all_classes_eval = seen_classes + unseen_classes
    search_pattern = f"{config['experiment']}_{method}*{args.suffix}_Pred_Center.npy"
    search_path = os.path.join(out_dir, search_pattern)
    files_to_test = glob.glob(search_path)
    files_to_test.sort()
    if not files_to_test:
        print(f'No model files found in {out_dir} matching {search_pattern}')
        sys.exit()
    print('\n' + '=' * 85)
    print(f"{'Filename Suffix':<40} | {'Seen (S)':<8} | {'Unseen (U)':<8} | {'H-Mean':<8}")
    print('-' * 85)
    for file_path in files_to_test:
        filename = os.path.basename(file_path)
        display_name = filename.replace(f"{config['experiment']}_{method}", '').replace('_Pred_Center.npy', '')
        if len(display_name) > 38:
            display_name = display_name[:35] + '...'
        pred_vectors = np.load(file_path)
        center_dict = {}
        total_keys = seen_classes + unseen_classes
        if pred_vectors.shape[0] != len(total_keys):
            limit = min(pred_vectors.shape[0], len(total_keys))
            for i in range(limit):
                center_dict[total_keys[i]] = pred_vectors[i]
        else:
            for i, cls_name in enumerate(total_keys):
                center_dict[cls_name] = pred_vectors[i]
        seen_correct = 0
        seen_total = 0
        unseen_correct = 0
        unseen_total = 0
        for i in range(len(all_labels)):
            label_id = all_labels[i][0]
            if label_id not in id_to_name:
                continue
            gt_name = id_to_name[label_id]
            feat = all_features[i]
            is_seen = gt_name in seen_classes
            is_unseen = gt_name in unseen_classes
            if not (is_seen or is_unseen):
                continue
            pred_name = NN_search(feat, center_dict, all_classes_eval)
            if is_seen:
                seen_total += 1
                if pred_name == gt_name:
                    seen_correct += 1
            elif is_unseen:
                unseen_total += 1
                if pred_name == gt_name:
                    unseen_correct += 1
        acc_seen = seen_correct / seen_total * 100 if seen_total > 0 else 0.0
        acc_unseen = unseen_correct / unseen_total * 100 if unseen_total > 0 else 0.0
        if acc_seen + acc_unseen > 0:
            h_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        else:
            h_mean = 0.0
        print(f'{display_name:<40} | {acc_seen:<8.2f} | {acc_unseen:<8.2f} | {h_mean:<8.2f}')
    print('=' * 85 + '\n')