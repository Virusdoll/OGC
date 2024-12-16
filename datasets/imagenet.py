from torchvision import transforms
from torchvision.datasets import ImageNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class ImageNetMini(ImageNet):
    def __init__(self, root, split='val', **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        for _, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 49:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs

def imagenetmini_val(root, *args, **kwargs):
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    dataset = ImageNetMini(root=root,
                           split='val',
                           transform=transform)
    return dataset