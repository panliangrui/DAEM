import enum
import re
# from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import os
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset
# import redis
import pickle
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
import random
import torch.backends.cudnn as cudnn
import json
import joblib
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from torch.cuda.amp import GradScaler, autocast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import h5py
from sklearn.metrics import confusion_matrix, precision_score, f1_score,average_precision_score,precision_recall_curve
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Args:
            alpha (float): 平衡因子
            gamma (float): 聚焦参数
            reduction (str): 'none' | 'mean' | 'sum'
            ignore_index (int): 忽略某些类别的标签
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出的 logits，形状 (batch_size, num_classes)
            targets: 真实标签，形状 (batch_size,)
        """
        # 计算交叉熵损失（未进行 reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        # 将 logits 经过 softmax 得到预测概率
        pt = torch.exp(-ce_loss)  # pt = exp(-loss) = probability of true class

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query, positive):
        """
        query: Tensor, shape: [batch_size, feature_dim]
        positive: Tensor, shape: [batch_size, feature_dim]
        负样本采用批内其他样本
        """
        batch_size = query.shape[0]
        # 归一化
        query = F.normalize(query, dim=1)
        positive = F.normalize(positive, dim=1)
        # 计算相似度矩阵，shape: [batch_size, batch_size]
        logits = torch.matmul(query, positive.T) / self.temperature
        # 对角线为正样本对，标签为 0,1,...,batch_size-1
        labels = torch.arange(batch_size, device=query.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    实现 Supervised Contrastive Loss (Khosla et al., 2020)
    Args:
        temperature: 温度参数，控制 logits 缩放，默认 0.07
    """
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: 张量，形状 [batch_size, feature_dim]，已归一化的特征向量
            labels: 张量，形状 [batch_size]，类别标签（整数形式）
        Returns:
            loss: 标量损失
        """
        device = features.device
        batch_size = features.shape[0]

        # 对特征进行归一化（若模型输出未归一化）
        features = F.normalize(features, dim=1)

        # 构造 mask：同一类别的样本对应位置为1，否则为0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # shape: [batch_size, batch_size]

        # 计算相似度矩阵 (点积相似度)，并除以温度参数
        logits = torch.matmul(features, features.T) / self.temperature  # shape: [batch_size, batch_size]

        # 为数值稳定性减去每行的最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 将自身对比（对角线）置为0，避免影响 loss 计算
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 计算每个样本的分母部分
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算每个样本正样本对的平均 log probability
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 最终损失为所有样本的均值
        loss = - mean_log_prob_pos.mean()

        return loss

class BagDataset(Dataset):
    def __init__(self, train_path, args) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args

    def get_bag_feats(self, csv_file_df, args):
        # if args.dataset.startswith('tcga'):
        #     feats_csv_path = os.path.join('datasets', args.dataset, 'data_tcga_lung_tree',
        #                                   csv_file_df.iloc[0].split('/')[-1] + '.csv')
        # else:
        feats_csv_path = csv_file_df.iloc[0]


        feats_TME = os.path.splitext(feats_csv_path)[0] + '.h5'
        # feats_TME = feats_TME.replace('features/256/FEATURES_DIRECTORY/pt_files', 'TME')
        feats_TME = feats_TME.replace('WSI_all', 'all_graph')
        # feats_TME = joblib.load(feats_TME)
        with h5py.File(feats_TME, 'r') as hf:
            # 读取 node_features
            node_features = hf['node_features'][:]

            # 读取 edges
            edges = hf['edges'][:]

            # # 读取 labels
            # labels = hf['labels'][:]

            # 读取 x_img_256 和对应的 edge
            x_img_256 = hf['x_img_256'][:]
            x_img_256_edge = hf['x_img_256_edge'][:]

            # 读取 x_img_512 和对应的 edge
            x_img_512 = hf['x_img_512'][:]
            x_img_512_edge = hf['x_img_512_edge'][:]

            # # 读取 x_img_1024 和对应的 edge
            # x_img_1024 = hf['x_img_1024'][:]
            # x_img_1024_edge = hf['x_img_1024_edge'][:]

        label = np.zeros(args.num_classes)
        if args.num_classes == 1:
            label[0] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1]) <= (len(label) - 1):
                label[int(csv_file_df.iloc[1])] = 1
        label = torch.tensor(np.array(label))
        # feats = torch.tensor(np.array(feats)).float()
        # return label, feats, feats_TME
        return label, x_img_256, x_img_512,x_img_256_edge,x_img_512_edge,node_features, edges

    def dropout_patches(self, feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats

    def __getitem__(self, idx):
        # label, feats, feats_TME = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        label, x_img_256, x_img_512, x_img_256_edge, x_img_512_edge, node_features, edges = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        # return label, feats, feats_TME
        return label, x_img_256, x_img_512, x_img_256_edge, x_img_512_edge, node_features, edges

    def __len__(self):
        return len(self.train_path)



def test(x_img_256, x_img_512,x_img_256_edge,x_img_512_edge,node_features, edges, milnet, criterion, focal_loss_fn,cross_emb, alignment_loss_fn, optimizer, args, log_path):

    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    file_name_dict=[]
    noe_hot_labels = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        # for i, (bag_label, x_img_256, x_img_512, x_img_256_edge, x_img_512_edge,  node_features, edges) in enumerate(test_df):
            torch.cuda.empty_cache()
            # label = bag_label.cpu().numpy()
            # bag_label = bag_label.to(device)#.cuda()
            # bag_label = (bag_label.squeeze(0)).to(device)
            # label = (bag_label.unsqueeze(0)).cpu().numpy()

            node_image_path_256_fea =  torch.Tensor(x_img_256).to(device)
            node_image_path_512_fea =  torch.Tensor(x_img_512).to(device)

            edge_index_image_256 =  torch.from_numpy(x_img_256_edge).to(device)
            edge_index_image_512 =  torch.from_numpy(x_img_512_edge).to(device)
            node_features =  torch.Tensor(node_features).to(device)
            edges =  torch.from_numpy(edges).to(device)
            # bag_label = (bag_label.squeeze(0)).to(device)  # .cuda()

            torch.cuda.empty_cache()

            if args.model == 'our':
                # with autocast():
                results_dict = milnet(node_image_path_256_fea, node_image_path_512_fea, edge_index_image_256,
                                      edge_index_image_512, node_features.to(torch.float32), edges)  # bag_feats_TME)
                # logits, logits1, logits2, features1, features2 = results_dict['logits'], results_dict['logits1'], \
                # results_dict['logits2'], results_dict['expert1'], results_dict['expert2']
                logits, features1, features2 = results_dict['logits'], results_dict['expert1'], results_dict['expert2']


                predicted_class = torch.argmax(logits, dim=1)  # 返回索引 0 或 1

    return predicted_class



def main():
    parser = argparse.ArgumentParser(description='Train our model')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='cptac', type=str, help='Dataset folder name[jingkai, nanhua, pingjiang, tcga, xiangya, xiangya3]')
    # parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='our', type=str, help='model our')
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', type=str, help='which agg')
    parser.add_argument('--c_path', nargs='+',
                        default=None, type=str,
                        help='directory to confounders') #'./datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
    # parser.add_argument('--dir', type=str,help='directory to save logs')mpnn, pool=args.pool
    parser.add_argument('--mpnn', default= 'GCN', type=str, help='networks')
    parser.add_argument('--pool', default= 'mean', type=str, help='methods')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--devices', type=int, default=0,
                       help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='input batch size for training (default: 128)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                       help='input batch size for training (default: train batch size)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='number of epochs to train (default: 100)')
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='number of workers (default: 0)')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--pct_start', type=float, default=0.3)
    # parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=None)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--start-eval', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None)
    # parser.add_argument('--seed', type=int, default=12344)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--n_hop', type=int, default=5)
    parser.add_argument('--slope', type=float, default=0.0)
    parser.add_argument('--pe_norm', type=bool, default=False)
    parser.add_argument('--pe_origin_dim', type=int, default=20)
    parser.add_argument('--pe_dim', type=int, default=20)
    ################################
    parser.add_argument('--model_type', type=str, default='gnn', help='gnn|pna|gnn-transformer')
    # parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--learnable', type=bool, default=True)
    # parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--task', type=str, default='multi_class')
    parser.add_argument('--criterion', type=str, default='accuracy')
    # group = parser.add_argument_group('gnn')
    parser.add_argument('--node_method', type=str, default='linear')
    parser.add_argument('--edge_method', type=str, default='None')
    # parser.add_argument('--mpnn', type=str, default='gcn')
    parser.add_argument('--projection', type=str, default='mlp')
    parser.add_argument('--gnn_virtual_node', action='store_true')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--attn_dropout', type=float, default=0.5)
    parser.add_argument('--gnn_num_layer', type=int, default=5,
                       help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--channels', type=int, default=64,
                       help='dimensionality of hidden units in GNNs (default: 64)')
    parser.add_argument('--gnn_JK', type=str, default='last')
    parser.add_argument('--gnn_residual', action='store_true', default=False)
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--nhead', type=int, default=4,
                       help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--folder_path', default='./multi_graph_1',
                        help='number of GNN message passing layers (default: 5)')

    args = parser.parse_args()
    # assert args.model == 'transmil'
    args.q_lora_rank= 0
    args.kv_lora_rank = 512
    args.qk_nope_head_dim = 128
    args.qk_rope_head_dim = 64
    args.v_head_dim = 128
    args.max_batch_size = 8
    args.max_seq_len = 4096 * 4

    args.vocab_size = 102400
    args.dim = 1
    args.inter_dim = 10944
    args.moe_inter_dim = 1408
    args.n_layers = 27
    args.n_dense_layers = 1
    args.n_heads = 16
    args.original_seq_len = 4096
    args.rope_theta = 10000.0
    args.rope_factor = 40
    args.beta_fast = 32
    args.beta_slow = 1
    args.mscale = 1.
    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_c_path')
    else:
        save_path = os.path.join('baseline', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_fulltune')
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file, 'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    



    # 遍历每个折数进行训练
    for fold in range(5):
        if fold != 4:
            continue
        with open(log_path, 'a+') as log_txt:
            info = '\n' + 'Fold at: ' + str(fold) + '\n'
            log_txt.write(info)
        print('Fold at: ' + str(fold))

    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''

    if args.model == 'our':
        import Models.our as mil
        milnet = mil.fusion_model_graph(args = args, in_channels=args.feats_size, hidden_channels=args.hidden_channels, out_channels =args.num_classes).to(device)#   input_size=args.feats_size, n_classes=args.num_classes,
        # milnet = mil.MultiScaleGraphClassifier().to(device)
    for name, _ in milnet.named_parameters():
        print('Training {}'.format(name))
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n Training {}'.format(name))

        # sanity check begins here
        print('*******sanity check *********')
        for k, v in milnet.named_parameters():
            if v.requires_grad == True:
                print(k)

        # loss, optim, schduler
        if args.num_classes == 2:
            alignment_loss_fn = nn.MSELoss()
            focal_loss_fn = SupervisedContrastiveLoss(
                temperature=0.07)  # FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')#SupervisedContrastiveLossOneHot(temperature=0.5)#nn.BCEWithLogitsLoss()focal_loss_fn =
            criterion = nn.BCEWithLogitsLoss()
            cross_emb = nn.CosineEmbeddingLoss()
            # info_nce_loss_fn = InfoNCELoss(temperature=0.2)
        else:
            criterion = nn.CrossEntropyLoss()
        original_params = []
        confounder_parms = []
        for pname, p in milnet.named_parameters():
            if ('confounder' in pname):
                confounder_parms += [p]
                print('confounders:', pname)
            else:
                original_params += [p]

        print('lood ahead optimizer in our....')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()),
                                     lr=args.lr, betas=(0.5, 0.9),
                                     weight_decay=args.weight_decay)

        best_score = 0

        feats_TME = [os.path.abspath(os.path.join(args.folder_path, f)) for f in os.listdir(args.folder_path) if
                 os.path.isfile(os.path.join(args.folder_path, f))]

        with h5py.File(feats_TME[0], 'r') as hf:
            # 读取 node_features
            node_features = hf['node_features'][:]

            # 读取 edges
            edges = hf['edges'][:]

            # # 读取 labels
            # labels = hf['labels'][:]

            # 读取 x_img_256 和对应的 edge
            x_img_256 = hf['x_img_256'][:]
            x_img_256_edge = hf['x_img_256_edge'][:]

            # 读取 x_img_512 和对应的 edge
            x_img_512 = hf['x_img_512'][:]
            x_img_512_edge = hf['x_img_512_edge'][:]
        model4 = milnet.to(device)


        td4 = torch.load(r'./test_models/3_4.pth',
                         map_location=torch.device('cuda:0'))
        model4.load_state_dict(td4, strict=False)
        result = test(
            x_img_256, x_img_512,x_img_256_edge,x_img_512_edge,node_features, edges,
            model4,
            criterion,focal_loss_fn,cross_emb, alignment_loss_fn,
            optimizer,
            args,
            log_path)  # , recall4, precision4,PRC4
        # overall_conf_matrix += conf_matrix
        # save_results_to_txt(filename, fold, conf_matrix, test_labels, test_predictions, noe_hot_labels4)
        if result[0] == 1:
            label = 'STAS'
        elif result[0] == 0:
            label = 'Non-STAS'
        else:
            label = 'I donot know!'
        print(label)
        # 打开一个文件以写入模式（'w'）
        with open("./results.txt", "w") as file:
            # 将结果逐行写入文件
            for result in label:
                file.write(result)



if __name__ == '__main__':

    main()
