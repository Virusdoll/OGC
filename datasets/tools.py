import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn

def create_class_dependent_noisy_label(labels, trans_matrix, seed=0):
    '''
    create noisy labels from labels and noisy matrix
    '''
    
    np.random.seed(seed)
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels

def create_instance_dependent_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed=0): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building dataset...")
    
    np.random.seed(seed)
    
    label_num = num_classes

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)

def data_split(data, targets, split_percentage, seed=1):
   
    num_samples = int(targets.shape[0])
    np.random.seed(int(seed))
    train_set_index = np.random.choice(num_samples, int(num_samples*split_percentage), replace=False)
    index = np.arange(data.shape[0])
    val_set_index = np.delete(index, train_set_index)
    train_set, val_set = data[train_set_index, :], data[val_set_index, :]
    train_labels, val_labels = targets[train_set_index], targets[val_set_index]

    return train_set, val_set, train_labels, val_labels
