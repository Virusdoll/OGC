import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

from .tools import create_instance_dependent_noisy_label

MEAN = [0.5071, 0.4865, 0.4409]
STD = [0.2673, 0.2564, 0.2762]

class CIFAR100PDN(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, noise_rate=0.2):
        super().__init__(root, train, transform, target_transform, download)
        
        self.true_targets = self.targets

        # build instance noisy labels
        data = torch.from_numpy(self.data).float()
        targets = torch.tensor(self.targets)
        dataset = zip(data, targets)
        noise_labels = create_instance_dependent_noisy_label(noise_rate, dataset, targets, 100, 32*32*3, 0.1)
        self.targets = noise_labels.reshape(-1).tolist()
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        true_target = self.true_targets[index]
        return index, img, target, true_target

def cifar100pdn_train(root, noise_rate, *args, **kwargs):
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = CIFAR100PDN(root=root,
                          train=True,
                          transform=transform,
                          noise_rate=noise_rate)
    return dataset
