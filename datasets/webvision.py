import os
from torchvision import transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class WebVisionMiniDataset:
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        
        if split == 'train':
            file_path = os.path.join(root, 'webvision_mini_train.txt')
        elif split == 'val':
            file_path = os.path.join(root, 'webvision_mini_val.txt')
        else:
            raise RuntimeError(f'Wrong split type: {split}')
        
        with open(file_path) as f:
            image_list = [(line.split()[0], int(line.split()[1])) \
                          for line in f]
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_file_name, target = self.image_list[index]
        image_path = os.path.join(self.root, image_file_name)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.split == 'train':
            return index, image, target, target
        else:
            return image, target

def webvisionmini_train(root, *args, **kwargs):
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4,
                                                           contrast=0.4,
                                                           saturation=0.4,
                                                           hue=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    
    dataset = WebVisionMiniDataset(root=root,
                                   split='train',
                                   transform=transform)
    return dataset

def webvisionmini_val(root, *args, **kwargs):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = WebVisionMiniDataset(root=root,
                                   split='val',
                                   transform=transform)
    return dataset