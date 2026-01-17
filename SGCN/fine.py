import os
import glob
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import SwinForImageClassification, SwinConfig

def default_loader(path):
    return Image.open(path).convert('RGB')

def Reduction_img(tensor, mean, std):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
data_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

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

def get_data(data_dir, class_list, labstart=0):
    classes = []
    with open(class_list, 'r') as f:
        for lines in f:
            line = lines.strip()
            classes.append(line)
    imgs_label = []
    for i, name in enumerate(classes):
        jpg_names = glob.glob(os.path.join(data_dir, name, '*.jpg'))
        imgs_label.extend(zip(jpg_names, [i + labstart] * len(jpg_names)))
    filenames, labels = zip(*imgs_label)
    return (filenames, labels, classes)

def get_Swin(length, base_model_path, loadfile=None):
    print(f'Loading Swin Transformer from: {base_model_path}')
    original_model = SwinForImageClassification.from_pretrained(base_model_path, num_labels=length, ignore_mismatched_sizes=True)
    if loadfile is not None:
        print(f'Loading fine-tuned weights from: {loadfile}')
        if os.path.exists(loadfile):
            state_dict = torch.load(loadfile)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            original_model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f'Warning: {loadfile} not found.')
    for param in original_model.parameters():
        param.requires_grad = False
    for param in original_model.classifier.parameters():
        param.requires_grad = True

    class SwinWrapper(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            if hasattr(outputs, 'logits'):
                return outputs.logits
            return outputs

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)
    return SwinWrapper(original_model)

def train(model, device, train_loader, epoch, optimizer):
    model.train()
    allloss = []
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        allloss.append(loss.item())
        optimizer.step()
    print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, np.mean(allloss)))

def test(model, device, val_loader):
    model.eval()
    test_loss = []
    correct = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            test_loss.append(criterion(y_hat, y).item())
            pred = y_hat.max(1, keepdim=True)[1]
            correct.append(pred.eq(y.view_as(pred)).sum().item() / pred.shape[0])
    print('\nTest set——{}: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(len(correct), np.mean(test_loss), np.mean(correct) * 100))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_class_list', type=str, help='Train Class')
    parser.add_argument('--out_dir', type=str, help='root of outputs')
    parser.add_argument('--swin_path', type=str, default='')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    filenames, labels, classes = get_data(args.data_dir, args.train_class_list)
    np.random.seed(0)
    label_shuffle_index = np.random.permutation(len(labels))
    label_train_num = len(labels) // 10 * 8
    train_list = label_shuffle_index[0:label_train_num]
    test_list = label_shuffle_index[label_train_num:]
    print('label_train_num________________', label_train_num, len(labels), len(classes))
    train_dataset = OwnDataset(filenames, labels, train_list, data_transform['train'])
    val_dataset = OwnDataset(filenames, labels, test_list, data_transform['val'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    ResNet = get_Swin(len(classes), base_model_path=args.swin_path)
    ResNet = ResNet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([{'params': ResNet.classifier.parameters()}], lr=0.0001, weight_decay=0.05)
    firstmodepth = os.path.join(args.out_dir, 'modelswin_1.pth')
    if os.path.exists(firstmodepth) == False:
        print('_____Stage 1: Training Classifier Only________')
        for epoch in range(1, 6):
            train(ResNet, device, train_loader, epoch, optimizer)
            test(ResNet, device, val_loader)
        torch.save(ResNet.state_dict(), firstmodepth)
    secondmodepth = os.path.join(args.out_dir, 'modelswin_2.pth')
    optimizer2 = optim.AdamW(ResNet.parameters(), lr=5e-05, weight_decay=0.05)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)
    for param in ResNet.parameters():
        param.requires_grad = True
    if os.path.exists(secondmodepth):
        ResNet.load_state_dict(torch.load(secondmodepth))
        test(ResNet, device, val_loader)
    else:
        ResNet.load_state_dict(torch.load(firstmodepth))
        print('_____Stage 2: Training Backbone________')
        for epoch in range(0, 1):
            train(ResNet, device, train_loader, epoch, optimizer2)
            if optimizer2.state_dict()['param_groups'][0]['lr'] > 1e-05:
                exp_lr_scheduler.step()
                print('___lr:', optimizer2.state_dict()['param_groups'][0]['lr'])
            test(ResNet, device, val_loader)
    torch.save(ResNet.state_dict(), secondmodepth)