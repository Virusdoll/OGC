import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .tools import create_class_dependent_noisy_label

MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.24703233, 0.24348505, 0.26158768]

class SyntheticCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        self.true_targets = self.targets
        self.targets = create_class_dependent_noisy_label(self.targets, trans_matrix)
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        true_target = self.true_targets[index]
        return index, img, target, true_target
    
def cifar10_train(root, noise_type, noise_rate, *args, **kwargs):
    trans_matrix = get_trans_matrix(noise_rate, noise_type)
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = SyntheticCIFAR10(root=root,
                               train=True,
                               transform=transform,
                               trans_matrix=trans_matrix)
    return dataset

def cifar10_test(root, *ags, **kwargs):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = CIFAR10(root=root, train=False, transform=transform)
    return dataset

def get_trans_matrix(eta, noise_type, num_classes=10):
    T = np.eye(num_classes)
    
    if noise_type == 'sym':
        T = T * (1 - eta) \
            + (1 - T) * eta / (num_classes - 1)
    elif noise_type == 'asym':
        # truck -> automobile (9 -> 1)
        T[9, 9], T[9, 1] = 1. - eta, eta
        # bird -> airplane (2 -> 0)
        T[2, 2], T[2, 0] = 1. - eta, eta
        # cat <-> dog (3 <-> 5)
        T[3, 3], T[3, 5] = 1. - eta, eta
        T[5, 5], T[5, 3] = 1. - eta, eta
        # deer -> horse (4 -> 7)
        T[4, 4], T[4, 7] = 1. - eta, eta
    else:
        raise RuntimeError(f'Wrong noise type: {noise_type}')
    
    return T
