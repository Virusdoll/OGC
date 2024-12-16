import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

MEAN = [0.5071, 0.4865, 0.4409]
STD = [0.2673, 0.2564, 0.2762]

class CIFAR100N(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, noise_type='clean', noise_file_path=None):
        super().__init__(root, train, transform, target_transform, download)
        noise_file = torch.load(noise_file_path)
        if noise_type == 'clean':
            noise_labels = noise_file['clean_label']
        elif noise_type == 'noisy':
            noise_labels = noise_file['noisy_label']
        else:
            raise ValueError(f'Wrong noise type: {noise_type}')
        self.true_targets = self.targets
        self.targets = noise_labels.reshape(-1).tolist()
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        true_target = self.true_targets[index]
        return index, img, target, true_target

def cifar100n_train(root, noise_type, noise_file_path, *args, **kwargs):
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = CIFAR100N(root=root,
                        train=True,
                        transform=transform,
                        noise_type=noise_type,
                        noise_file_path=noise_file_path)
    return dataset
