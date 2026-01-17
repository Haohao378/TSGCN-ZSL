import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
BASE_DIR = '/media/data/T-ZSL/SGCN/EDO/outputs'
methods_config = {'VCL': {'pred': os.path.join(BASE_DIR, 'SWIN_DAILY_output_predVCL.txt'), 'true': os.path.join(BASE_DIR, 'SWIN_DAILY_output_trueVCL.txt')}, 'CDVSc': {'pred': os.path.join(BASE_DIR, 'SWIN_DAILY_output_predCDVSc.txt'), 'true': os.path.join(BASE_DIR, 'SWIN_DAILY_output_trueCDVSc.txt')}, 'BMVSc': {'pred': os.path.join(BASE_DIR, 'SWIN_DAILY_output_predBMVSc.txt'), 'true': os.path.join(BASE_DIR, 'SWIN_DAILY_output_trueBMVSc.txt')}, 'WDVSc': {'pred': os.path.join(BASE_DIR, 'SWIN_DAILY_output_predEMDVSc.txt'), 'true': os.path.join(BASE_DIR, 'SWIN_DAILY_output_trueEMDVSc.txt')}, 'DTVSc': {'pred': os.path.join(BASE_DIR, 'SWIN_DAILY_output_predDTVSc.txt'), 'true': os.path.join(BASE_DIR, 'SWIN_DAILY_output_trueDTVSc.txt')}}
class_file = '/media/data/T-ZSL/plot/test_ID.txt'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def load_data_and_plot():
    target_class = {}
    i = 0
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            for lines in f:
                line = lines.strip()
                target_class[line] = i
                i += 1
    else:
        print('Error: Class file not found!')
        return
    num_classes = i
    for method_name, paths in methods_config.items():
        print(f'Processing {method_name}...')
        if not os.path.exists(paths['pred']) or not os.path.exists(paths['true']):
            print(f'  Skipping {method_name}: Files not found.')
            continue
        with open(paths['pred'], 'r') as f:
            pred_raw = eval(f.read())
        with open(paths['true'], 'r') as f:
            true_raw = eval(f.read())
        pred_labels = [target_class[p] for p in pred_raw]
        true_labels = [target_class[t] for t in true_raw]
        cm = confusion_matrix(true_labels, pred_labels)
        cm_norm = np.round(cm / np.sum(cm, axis=1)[:, np.newaxis], 2)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm_norm, ax=ax, cmap='Blues', cbar=True, cbar_kws={'label': '', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0], 'shrink': 0.8}, xticklabels=False, yticklabels=False, square=True, linewidths=0.5, linecolor='white')
        tick_step = 5
        ticks = np.arange(0, num_classes, tick_step)
        tick_labels = [str(t) for t in ticks]
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticks(ticks + 0.5)
        ax.set_yticklabels(tick_labels, rotation=0)
        save_name = f'cm_{method_name}.pdf'
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f'  Saved {save_name}')
        plt.close()
if __name__ == '__main__':
    load_data_and_plot()