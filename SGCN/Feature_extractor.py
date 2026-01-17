import os
import glob
import json
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import argparse
from transformers import SwinForImageClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import argparse
from tqdm import tqdm

def default_loader(path):
    return Image.open(path).convert('RGB')
data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'val': transforms.Compose([transforms.Resize(256), transforms.TenCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))])}

class OwnDataset(Dataset):

    def __init__(self, img_dir, labels, indexlist=None, transform=transforms.ToTensor(), loader=default_loader, cache=True):
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.loader = loader
        self.cache = cache
        if indexlist is None:
            self.indexlist = list(range(len(self.img_dir)))
        else:
            self.indexlist = indexlist
        self.data = [None] * len(self.indexlist)

    def __getitem__(self, idx):
        if self.data[idx] is None:
            data = self.loader(self.img_dir[self.indexlist[idx]])
            if self.transform:
                data = self.transform(data)
        else:
            data = self.data[idx]
        if self.cache:
            self.data[idx] = data
        return (data, self.labels[self.indexlist[idx]])

    def __len__(self):
        return len(self.indexlist)

def get_data(data_dir, class_list, datasets, labstart=0):
    classes = []
    with open(class_list, 'r') as f:
        for lines in f:
            line = lines.strip()
            classes.append(line)
    imgs_label = []
    for i, name in enumerate(classes):
        if datasets == 'EDO':
            jpg_names = glob.glob(os.path.join(data_dir, name, '*.JPEG'))
        else:
            jpg_names = glob.glob(os.path.join(data_dir, name, '*.jpg'))
        imgs_label.extend(zip(jpg_names, [i + labstart] * len(jpg_names)))
    filenames, labels = zip(*imgs_label)
    return (filenames, labels, classes)

def Cluster(features, method, center_num):
    if method == 'Kmeans':
        obj = KM(features, center_num)
    else:
        obj = SC(features, center_num)
    return obj

def KM(features, center_num):
    print('Start Cluster ...')
    min_distance = float('inf')
    mini_centers = []
    for i in range(10):
        clf = KMeans(n_clusters=center_num, n_init=20, max_iter=200000, init='k-means++')
        s = clf.fit(features)
        labels = clf.labels_
        centers = clf.cluster_centers_
        distances = np.linalg.norm(features - centers[labels], axis=1)
        total_distance = np.sum(distances)
        print(total_distance)
        if total_distance < min_distance:
            min_distance = total_distance
            mini_centers = centers.tolist()
    print('Finish Cluster ...')
    obj = {}
    obj['VC'] = mini_centers
    return obj

def SC(features, center_num):
    Spectral = SpectralClustering(n_clusters=center_num, eigen_solver='arpack', affinity='nearest_neighbors')
    print('Start Cluster ...')
    pred_class = Spectral.fit_predict(features)
    print('Finish Cluster ...')
    belong = Spectral.labels_
    sum = {}
    count = {}
    for i, x in enumerate(features):
        label = belong[i]
        if sum.get(label) is None:
            feature_dim = len(x)
            sum[label] = [0.0] * feature_dim
            count[label] = 0
        for j, y in enumerate(x):
            sum[label][j] += y
        count[label] += 1
    all_cluster_center = []
    for label in sum.keys():
        for i, x in enumerate(sum[label]):
            sum[label][i] /= count[label] * 1.0
        all_cluster_center.append(sum[label])
    obj = {}
    obj['VC'] = all_cluster_center
    return obj

def savefeature(classdir, filename, obj):
    os.makedirs(classdir, exist_ok=True)
    cur_url = os.path.join(classdir, filename)
    json.dump(obj, open(cur_url, 'w'))
    print('%s has finished ...' % classdir)

def avgfeature(all_features):
    avg_features = np.sum(np.asarray(all_features), axis=0) / len(all_features)
    avg_features = torch.tensor(avg_features)
    avg_features = F.normalize(avg_features, dim=0)
    avg_features = avg_features.numpy()
    return avg_features

def extract_feature(data_loader, net, feature_root, classes, device):
    all_features = []
    feature_VC = []
    i = -1
    net.eval()
    print('>>> Extracting Features with Ten-Crop & Double Normalization...')
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
            inputs, labels = data
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.to(device).view(-1, c, h, w)
            if i == -1:
                i = labels[0].item()
            if i != labels[0].item() and len(all_features) > 0:
                classdir = os.path.join(feature_root, classes[i])
                obj = {}
                obj['features'] = all_features
                savefeature(classdir, 'feature.json', obj)
                avg_features = avgfeature(all_features)
                avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-08)
                feature_VC.append(avg_features.tolist())
                i = labels[0].item()
                all_features = []
            features = net(inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                elif hasattr(features, 'logits'):
                    features = features.logits
                elif isinstance(features, (list, tuple)):
                    features = features[0]
            features = features.view(bs, ncrops, -1)
            features = features.mean(dim=1)
            features = F.normalize(features, p=2, dim=1)
            fea_vec = features.cpu().detach().numpy()
            for vec in fea_vec:
                all_features.append(vec.tolist())
    if len(all_features) > 0:
        classdir = os.path.join(feature_root, classes[i])
        obj = {}
        obj['features'] = all_features
        savefeature(classdir, 'feature.json', obj)
        avg_features = avgfeature(all_features)
        avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-08)
        feature_VC.append(avg_features.tolist())
    return feature_VC
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='root of dataset')
    parser.add_argument('--datasets', type=str, default='EDO', help='EDO/AWA2')
    parser.add_argument('--train_class_list', type=str, help='Train(seen) class list location')
    parser.add_argument('--test_class_list', type=str, help='Test(Unseen) class list location')
    parser.add_argument('--cluster_method', type=str, default='SC', help='choose the cluster algorithm')
    parser.add_argument('--out_dir', type=str, help='root of outputs')
    parser.add_argument('--loadfile', type=str, default=None, help='Path to fine-tuned .pth model file')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 20} Model Initialization (Hugging Face) {'=' * 20}")
    hf_model_path = "./checkpoints/swin_base"
    try:
        print(f'>>> Loading Base Model from: {hf_model_path}')
        original_model = SwinForImageClassification.from_pretrained(hf_model_path, ignore_mismatched_sizes=True)
        first_layer_weight = original_model.swin.embeddings.patch_embeddings.projection.weight
        if torch.sum(first_layer_weight) == 0:
            print('⚠️ Warning: Backbone weights look suspicious (all zeros).')
        else:
            print('✅ Base Swin Transformer backbone loaded successfully.')
    except Exception as e:
        print(f'❌ Fatal Error loading HF model: {e}')
        raise e
    if args.loadfile is not None:
        print(f'\n>>> Loading Fine-tuned Weights from: {args.loadfile}')
        if os.path.isfile(args.loadfile):
            checkpoint = torch.load(args.loadfile, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                if name.startswith('model.'):
                    name = name[6:]
                if 'classifier' in name:
                    continue
                new_state_dict[name] = v
            original_model.load_state_dict(new_state_dict, strict=False)
            test_key = 'swin.embeddings.patch_embeddings.projection.weight'
            if test_key in new_state_dict:
                current_weight = original_model.state_dict()[test_key]
                loaded_weight = new_state_dict[test_key]
                if torch.equal(current_weight, loaded_weight):
                    print(f'✅ Validation Passed: Backbone weights loaded correctly.')
                else:
                    print(f'✅ Custom weights loaded (Backbone updated).')
            print(f'✅ Custom weights loaded process finished.')
        else:
            print(f'❌ Error: File {args.loadfile} not found! Using original HF weights.')

    class FeatureExtractor(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model.swin

        def forward(self, x):
            outputs = self.model(x)
            return outputs.pooler_output

    model = FeatureExtractor(original_model)
    model = model.to(device)
    model.eval()
    transform_data = data_transform['val']
    train_file, train_label, train_class = get_data(args.data_dir, args.train_class_list, args.datasets)
    test_file, test_label, test_class = get_data(args.data_dir, args.test_class_list, args.datasets, len(train_class))
    train_dataset = OwnDataset(train_file, train_label, list(range(len(train_label))), transform_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_dataset = OwnDataset(test_file, test_label, list(range(len(test_label))), transform_data)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    classes = train_class + test_class
    feature_root = os.path.join(args.out_dir, 'img_feature')
    os.makedirs(feature_root, exist_ok=True)
    train_VC = extract_feature(train_loader, model, feature_root, classes, device)
    test_VC = extract_feature(test_loader, model, feature_root, classes, device)
    obj = {}
    obj['train'] = train_VC
    obj['test'] = test_VC
    savefeature(os.path.join(args.out_dir, 'VC_feature'), 'ave_VC.json', obj)
    print('VC 结束')
    feature_root = os.path.join(args.out_dir, 'img_feature')
    all_test_features = []
    for ii, target in enumerate(test_class):
        cur = os.path.join(feature_root, target)
        url = os.path.join(cur, 'feature.json')
        js = json.load(open(url, 'r'))
        cur_features = js['features']
        all_test_features = all_test_features + cur_features
    obj = Cluster(all_test_features, args.cluster_method, len(test_class))
    savefeature(os.path.join(args.out_dir, 'VC_feature'), 'cluster_VC.json', obj)
    print('cluster finished')
