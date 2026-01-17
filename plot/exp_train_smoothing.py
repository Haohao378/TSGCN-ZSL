import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from config.parser import YAMLParser

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

class GraphConv(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        return torch.mm(adj, support)

class ResMLPBlock(nn.Module):

    def __init__(self, hidden_dim):
        super(ResMLPBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.LeakyReLU(0.2)
        nn.init.eye_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.act(out)
        return out + residual

class ComparisonModel(nn.Module):

    def __init__(self, nfeat, nhid, nclass, total_layers, model_type, device):
        super(ComparisonModel, self).__init__()
        self.model_type = model_type
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(0.2)
        if model_type == 'Coupled':
            self.layers.append(GraphConv(nfeat, nhid))
            for _ in range(total_layers - 2):
                self.layers.append(GraphConv(nhid, nhid))
            self.layers.append(GraphConv(nhid, nclass))
        elif model_type == 'Decoupled':
            self.layers.append(GraphConv(nfeat, nhid))
            self.layers.append(GraphConv(nhid, nhid))
            remain_layers = max(1, total_layers - 2)
            for i in range(remain_layers - 1):
                self.layers.append(ResMLPBlock(nhid))
            self.layers.append(nn.Linear(nhid, nclass))

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GraphConv):
                x = layer(x, adj)
                if i < len(self.layers) - 1:
                    x = self.act(x)
            elif isinstance(layer, ResMLPBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_SGCN_EDO.yml', help='Path to config file')
    parser.add_argument('--subset_num', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f'Error: Config file not found at {args.config}')
        return
    config_parser = YAMLParser(args.config)
    config = config_parser.config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f">>> Experiment: {config['experiment']} | Subset={args.subset_num} | Device={device}")
    train_class_list = config['data']['train_class_list']
    test_class_list = config['data']['test_class_list']
    train_wnids = [line.strip() for line in open(train_class_list, 'r')]
    test_wnids = [line.strip() for line in open(test_class_list, 'r')]
    key_wnids = train_wnids + test_wnids
    n_nodes = 0
    word_vectors = None
    edges = []
    if 'CUB' in config['experiment'] or 'SUN' in config['experiment'] or 'FLO' in config['experiment']:
        print('>>> Mode: Attribute-based KNN Graph (CUB/SUN/FLO)')
        n_nodes = len(key_wnids)
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
            if name in name_to_index:
                ordered_attributes.append(raw_attributes[name_to_index[name]])
            else:
                print(f"Warning: Class '{name}' not found in classes.txt mapping!")
                ordered_attributes.append(np.zeros(raw_attributes.shape[1]))
        word_vectors = torch.tensor(np.array(ordered_attributes)).float().to(device)
        norm_wv = F.normalize(word_vectors, dim=1)
        sim_matrix = torch.mm(norm_wv, norm_wv.t())
        k = 20 if 'SUN' in config['experiment'] else 10
        _, indices = torch.topk(sim_matrix, k=k + 1, dim=1)
        indices = indices.cpu().numpy()
        edges = []
        for i in range(n_nodes):
            for neighbor in indices[i]:
                if i != neighbor:
                    edges.append((i, neighbor))
        print(f'   Nodes: {n_nodes}, Edges: {len(edges)}, Input Dim: {word_vectors.shape[1]}')
    else:
        print('>>> Mode: Hierarchy-based WordNet Graph (EDO/AWA2)')
        xml_set = []
        s = list(map(getnode, key_wnids))
        induce_parents(s, xml_set)
        wnids = list(map(getwnid, s))
        edges = getedges(s)
        n_nodes = len(wnids)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        glove_path = os.path.join(parent_dir, 'glove.6B.300d.txt')
        if os.path.exists(glove_path):
            glove = GloVe(glove_path)
            word_vectors = torch.stack([glove[getnode(w).lemmas()[0].name()] for w in wnids]).to(device)
        else:
            print(f'Warning: GloVe not found at {glove_path}, using random init.')
            word_vectors = torch.randn(n_nodes, 300).to(device)
    edges = list(set(edges + [(v, u) for u, v in edges]))
    edges = list(set(edges + [(u, u) for u in range(n_nodes)]))
    adj = torch.eye(n_nodes).to(device)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    degree = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    out_dir = config['data']['out_dir']
    try:
        vcdir_train = os.path.join(out_dir, 'VC_feature', 'ave_VC.json')
        VC_train = json.load(open(vcdir_train, 'r'))['train']
        vcdir_test = os.path.join(out_dir, 'VC_feature', 'cluster_VC.json')
        VC_test = json.load(open(vcdir_test, 'r'))['VC']
    except FileNotFoundError:
        print('Error: VC features not found. Please run Feature_extractor.py first.')
        return
    VC_all = VC_train + VC_test
    VC_target = torch.tensor(VC_all).float().to(device)
    VC_target = F.normalize(VC_target, dim=1)
    visual_dim = VC_target.shape[1]
    print(f'>>> Detected Visual Feature Dimension: {visual_dim}')
    target_len = len(key_wnids)
    input_dim = word_vectors.shape[1]
    depths = [2, 4, 8, 16, 32]
    results_json = {'baseline': {}, 'ours': {}}
    print('\n>>> Start Training Comparison Models...')
    for model_type in ['Coupled', 'Decoupled']:
        results_json[model_type.lower()] = {}
        for depth in depths:
            hidden_dim = 1024
            model = ComparisonModel(input_dim, hidden_dim, visual_dim, depth, model_type, device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
            model.train()
            pbar = tqdm(range(args.epochs), desc=f'{model_type}-L{depth}', leave=False)
            for epoch in pbar:
                optimizer.zero_grad()
                output = model(word_vectors, norm_adj)
                pred_visual = output[0:target_len]
                loss = torch.sum((pred_visual - VC_target) ** 2) / target_len
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                final_feat = model(word_vectors, norm_adj)
                subset_feat = final_feat[0:args.subset_num]
                subset_feat = F.normalize(subset_feat, p=2, dim=1)
                sim_matrix = torch.mm(subset_feat, subset_feat.t())
                results_json[model_type.lower()][str(depth)] = sim_matrix.cpu().numpy().tolist()
            print(f'   Finished {model_type} Depth {depth}')
    save_path = os.path.join(out_dir, 'smoothing_experiment_data.json')
    with open(save_path, 'w') as f:
        json.dump(results_json, f)
    print(f'\n>>> Data saved to {save_path}')
    print(">>> Now run 'python plot_smoothing_json.py' to generate the figure.")
if __name__ == '__main__':
    run_experiment()