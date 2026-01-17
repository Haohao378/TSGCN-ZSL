import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib.font_manager as font_manager
from matplotlib import gridspec
font_path = './times.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.2

def plot_refined():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='')
    parser.add_argument('--save_name', default='final_smoothing_analysis')
    args = parser.parse_args()
    if not os.path.exists(args.json_path):
        print(f'Error: JSON file not found at {args.json_path}')
        data = {'coupled': {d: np.random.rand(30, 30) for d in ['2', '4', '8', '16', '32']}, 'decoupled': {d: np.random.rand(30, 30) for d in ['2', '4', '8', '16', '32']}}
    else:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
    depths = ['2', '4', '8', '16', '32']
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08], wspace=0.05, hspace=0.1)
    cmap = 'Reds'
    vmin, vmax = (0.0, 1.0)
    rows = [('coupled', 'Coupled GCN'), ('decoupled', 'Decoupled SGCN')]
    axes_list = []
    for row_idx, (key_name, row_label) in enumerate(rows):
        for col_idx, depth in enumerate(depths):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes_list.append(ax)
            matrix = np.array(data[key_name][depth])
            if matrix.shape[0] > 100:
                matrix = matrix[:25, :25]
            sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False, square=True, linewidths=0, rasterized=True)
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            if row_idx == 0:
                ax.set_title(f'Layer = {depth}', fontsize=18, pad=10)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=18, labelpad=10)
    cbar_ax = fig.add_subplot(gs[:, 5])
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 + 0.02, pos.y0 + 0.1, pos.width * 0.4, pos.height * 0.8])
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Cosine Similarity', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_visible(False)
    pdf_path = f'{args.save_name}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f'Done! Saved to {pdf_path}')
if __name__ == '__main__':
    plot_refined()