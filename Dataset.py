import os

from PIL import Image
from torch.utils.data import Dataset



class ConditionalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith(".bmp"):  # 根据你的数据集调整文件扩展名
                    path = os.path.join(cls_dir, file)
                    x = int(cls)
                    self.samples.append((path, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target