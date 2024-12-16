import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.24703233, 0.24348505, 0.26158768]

class CIFAR10N(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, noise_type='clean', noise_file_path=None):
        super().__init__(root, train, transform, target_transform, download)

        noise_file = torch.load(noise_file_path)
        if noise_type == 'clean':
            noise_labels = noise_file['clean_label']
        elif noise_type == 'aggregate':
            noise_labels = noise_file['aggre_label']
        elif noise_type == 'random1':
            noise_labels = noise_file['random_label1']
        elif noise_type == 'random2':
            noise_labels = noise_file['random_label2']
        elif noise_type == 'random3':
            noise_labels = noise_file['random_label3']
        elif noise_type == 'worst':
            noise_labels = noise_file['worse_label']
        else:
            raise RuntimeError(f'Wrong noise type: {noise_type}')
        self.true_targets = self.targets
        self.targets = noise_labels.reshape(-1).tolist()
        
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        true_traget = self.true_targets[index]
        return index, img, target, true_traget

def cifar10n_train(root, noise_type, noise_file_path, *args, **kwargs):
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = CIFAR10N(root=root,
                       train=True,
                       transform=transform,
                       noise_type=noise_type,
                       noise_file_path=noise_file_path)
    return dataset
