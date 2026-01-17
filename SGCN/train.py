import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
import time
from GCN import GCN
from torch.optim import lr_scheduler
from Sinkhorn import SinkhornDistance
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import nltk
from nltk.corpus import wordnet as wn
from config.parser import YAMLParser

def CDVSc(a, b, device, n, m, lamda):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    mask = list(range(n - m))
    L2_loss = ((a[mask] - b[mask]) ** 2).sum() / ((n - m) * 2)
    A = a[n - m:]
    B = b[n - m:]
    dist_matrix = torch.cdist(A, B, p=2).pow(2)
    row_min, _ = dist_matrix.min(dim=1)
    col_min, _ = dist_matrix.min(dim=0)
    CD_loss = row_min.sum() + col_min.sum()
    tot_loss = L2_loss + CD_loss * lamda
    return tot_loss

def BMVSc(a, b, device, n, m, lamda):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    mask = list(range(n - m))
    L2_loss = ((a[mask] - b[mask]) ** 2).sum() / ((n - m) * 2)
    A = a[n - m:]
    B = b[n - m:]
    dist_matrix = torch.cdist(A, B, p=2).pow(2)
    cost = dist_matrix.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    matching_loss = 0
    for r, c in zip(row_ind, col_ind):
        matching_loss += dist_matrix[r, c]
    tot_loss = L2_loss + matching_loss * lamda
    return tot_loss

def EMDVSc(a, b, device, n, m, lamda, no_use_VSC=False):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    WD = SinkhornDistance(0.01, 1000, device, 'mean')
    mask = list(range(n - m))
    L2_loss = ((a[mask] - b[mask]) ** 2).sum() / ((n - m) * 2)
    if no_use_VSC:
        return (L2_loss, None, None)
    A = a[n - m:]
    B = b[n - m:]
    WD_loss, P, C = WD(A, B)
    WD_loss = WD_loss.to(device)
    tot_loss = L2_loss + WD_loss * lamda
    return (tot_loss, P, C)

def DTVSc(a, b, device, n, m, lamda, max_epoch, epoch):
    split_epoch = max_epoch // 4
    if epoch < split_epoch:
        loss, _, _ = EMDVSc(a, b, device, n, m, lamda)
    else:
        loss = BMVSc(a, b, device, n, m, lamda)
    return loss

class GloVe:

    def __init__(self, file_path):
        self.dimension = None
        self.embedding = dict()
        print(f'Loading GloVe embeddings from {file_path} ...')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                strs = line.strip().split()
                word = strs[0]
                vector = torch.FloatTensor(list(map(float, strs[1:])))
                self.embedding[word] = vector
                if self.dimension is None:
                    self.dimension = len(vector)

    def zeros(self):
        return torch.zeros(self.dimension)

    def __getitem__(self, words):
        if type(words) is str:
            words = [words]
        ret = self.zeros()
        cnt = 0
        for word in words:
            v = self.embedding.get(word)
            if v is not None:
                ret += v
                cnt += 1
        return ret / cnt if cnt > 0 else self.zeros()

def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges

def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:
            continue
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)
if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='base config file')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--wd', type=float, default=None, help='weight decay')
    parser.add_argument('--lamda', type=float, default=None, help='lambda for unseen loss')
    parser.add_argument('--max_epoch', type=int, default=None, help='max epochs')
    parser.add_argument('--hidden_layers', type=str, default=None, help="e.g. '1024' or '1024,1024'")
    parser.add_argument('--suffix', type=str, default='', help="suffix for saved file name (e.g. '_lr0.0001')")
    parser.add_argument('--method', type=str, default=None, help='override method: DTVSc, etc.')
    args = parser.parse_args()
    config_parser = YAMLParser(args.config)
    config = config_parser.config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'>>> Training on {device}')
    method = args.method if args.method is not None else config['data']['method']
    classNum = config['data']['classNum']
    unseenclassnum = config['data']['unseenclassnum']
    input_dim = config['data']['input_dim']
    train_class_list = config['data']['train_class_list']
    test_class_list = config['data']['test_class_list']
    out_dir = config['data']['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if method == 'VCL':
        no_use_VSC = True
    else:
        no_use_VSC = False
    xml_set = []
    train_wnids = [line.strip() for line in open(train_class_list, 'r')]
    test_wnids = [line.strip() for line in open(test_class_list, 'r')]
    key_wnids = train_wnids + test_wnids
    if 'CUB' in config['experiment'] or 'SUN' in config['experiment'] or 'FLO' in config['experiment']:
        print(f">>> Processing {config['experiment']}: KNN Graph Construction...")
        n = classNum
        att_path = config['data']['attribute_dir']
        raw_attributes = np.loadtxt(att_path)
        classes_path = config['data']['classes_dir']
        name_to_index = {}
        with open(classes_path, 'r') as f:
            for idx, line in enumerate(f):
                content = line.strip()
                if not content:
                    continue
                parts = content.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    name_to_index[parts[1]] = int(parts[0]) - 1
                else:
                    class_name = parts[0]
                    name_to_index[class_name] = idx
        ordered_attributes = []
        for name in key_wnids:
            ordered_attributes.append(raw_attributes[name_to_index[name]])
        word_vectors = torch.tensor(np.array(ordered_attributes)).float().to(device)
        word_vectors = F.normalize(word_vectors, dim=1)
        sim_matrix = torch.mm(word_vectors, word_vectors.t())
        edges = []
        k = 10
        if 'SUN' in config['experiment']:
            k = 20
        _, indices = torch.topk(sim_matrix, k=k + 1, dim=1)
        indices = indices.cpu().numpy()
        for i in range(n):
            for neighbor in indices[i]:
                if i != neighbor:
                    edges.append((i, neighbor))
        print(f'>>> KNN Graph Built! Nodes: {n}, Edges: {len(edges)}')
    else:
        print('>>> Processing EDO/AWA2: WordNet Graph...')
        s = list(map(getnode, key_wnids))
        induce_parents(s, xml_set)
        wnids = list(map(getwnid, s))
        edges = getedges(s)
        n = len(wnids)
        glove_path = os.path.join(parent_dir, 'glove.6B.300d.txt')
        glove = GloVe(glove_path)
        word_vectors = torch.stack([glove[getnode(w).lemmas()[0].name()] for w in wnids]).to(device)
    edges = list(set(edges + [(v, u) for u, v in edges]))
    edges = list(set(edges + [(u, u) for u in range(n)]))
    hidden_str = args.hidden_layers if args.hidden_layers is not None else config[method]['hidden_layers']
    print(f'>>> Hidden Layers: {hidden_str}')
    Net_s = GCN(n, edges, input_dim, input_dim, hidden_str, device).to(device)
    vcdir = os.path.join(out_dir, 'VC_feature', 'ave_VC.json')
    obj = json.load(open(vcdir, 'r'))
    VC = obj['train']
    vcdir = os.path.join(out_dir, 'VC_feature', 'cluster_VC.json')
    obj = json.load(open(vcdir, 'r'))
    test_center = obj['VC']
    VC = VC + test_center
    VC = torch.tensor(VC).float().to(device)
    VC = F.normalize(VC, dim=1)
    output_dim = VC.shape[1]
    class_edges = edges + [(u, u) for u in range(classNum)]
    Net = GCN(classNum, class_edges, input_dim, output_dim, hidden_str, device).to(device)
    lr = args.lr if args.lr is not None else 0.0001
    wd = args.wd if args.wd is not None else 0.0005
    max_epoch = args.max_epoch if args.max_epoch is not None else config[method]['max_epoch']
    lamda = args.lamda if args.lamda is not None else config[method]['lamda']
    print(f'>>> Config: LR={lr}, WD={wd}, Lamda={lamda}, Epoch={max_epoch}, Method={method}')
    optimizer = torch.optim.Adam(list(Net.parameters()) + list(Net_s.parameters()), lr=lr, weight_decay=wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    if 'SUN' in config['experiment']:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=max_epoch // 2, gamma=0.1)
    print(f'\n>>> Start Training {method} (Max Epoch: {max_epoch})...')
    mini_loss = float('inf')
    start_time = time.time()
    for epoch in range(max_epoch + 1):
        Net.train()
        Net_s.train()
        scheduler.step()
        refined_attrs = Net_s(word_vectors)
        syn_vc = Net(refined_attrs[0:classNum])
        if method == 'VCL':
            loss, _, _ = EMDVSc(syn_vc, VC, device, classNum, unseenclassnum, lamda, no_use_VSC=True)
        elif method == 'CDVSc':
            loss = CDVSc(syn_vc, VC, device, classNum, unseenclassnum, 0.0005)
        elif method == 'BMVSc':
            loss = BMVSc(syn_vc, VC, device, classNum, unseenclassnum, lamda)
        elif method == 'EMDVSc':
            loss, _, _ = EMDVSc(syn_vc, VC, device, classNum, unseenclassnum, lamda)
        elif method == 'DTVSc':
            split_epoch = max_epoch // 4
            if epoch == split_epoch:
                mini_loss = float('inf')
            loss = DTVSc(syn_vc, VC, device, classNum, unseenclassnum, lamda, max_epoch, epoch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(Net.parameters(), max_norm=10.0)
        nn.utils.clip_grad_norm_(Net_s.parameters(), max_norm=10.0)
        optimizer.step()
        if loss.item() < mini_loss:
            mini_loss = loss.item()
            parameters_s = Net_s.state_dict()
            parameters = Net.state_dict()
        if epoch % 500 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f'Epoch {epoch}/{max_epoch} | Loss: {loss.item():.6f} | Time: {elapsed:.2f} m')
    if method in ['BMVSc', 'EMDVSc', 'DTVSc', 'CDVSc', 'VCL']:
        Net.load_state_dict(parameters)
        Net_s.load_state_dict(parameters_s)
    Net.eval()
    Net_s.eval()
    with torch.no_grad():
        output_vectors = Net(Net_s(word_vectors)[0:classNum])
    output_vectors = output_vectors.detach().cpu().numpy()
    suffix = args.suffix if args.suffix else str(args.lr) + '_' + str(args.wd) + '_' + str(args.lamda) + '_' + str(args.max_epoch) + '_' + args.hidden_layers
    save_name = config['experiment'] + '_' + method + suffix + '_Pred_Center.npy'
    save_path = os.path.join(out_dir, save_name)
    np.save(save_path, output_vectors)
    print(f'>>> Result saved to: {save_path}')