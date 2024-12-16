import numpy as np
from torchvision.datasets import CIFAR100
from torchvision import transforms

from .tools import create_class_dependent_noisy_label

MEAN = [0.5071, 0.4865, 0.4409]
STD = [0.2673, 0.2564, 0.2762]

class SyntheticCIFAR100(CIFAR100):
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
    
def cifar100_train(root, noise_type, noise_rate, *args, **kwargs):
    trans_matrix = get_trans_matrix(noise_rate, noise_type)
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = SyntheticCIFAR100(root=root,
                                train=True,
                                transform=transform,
                                trans_matrix=trans_matrix)
    
    return dataset

def cifar100_test(root, *args, **kwargs):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = CIFAR100(root=root, train=False, transform=transform)
    
    return dataset

def get_trans_matrix(eta, noise_type, num_classes=100):
    T = np.eye(num_classes)
    
    if noise_type == 'sym':
        T = T * (1 - eta) \
            + (1 - T) * eta / (num_classes - 1)
    elif noise_type == 'asym':
        num_superclasses = 20
        num_subclasses = 5
        for i in np.arange(num_superclasses):
            # build T for one superclass
            T_superclass = (1. - eta) * np.eye(num_subclasses)
            for j in np.arange(num_subclasses - 1):
                T_superclass[j, j + 1] = eta
            T_superclass[num_subclasses - 1, 0] = eta
            
            init, end = i * num_subclasses, (i + 1) * num_subclasses
            T[init:end, init:end] = T_superclass
    else:
        raise RuntimeError(f'Wrong noise type: {noise_type}')
    
    return T
