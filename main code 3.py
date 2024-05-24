import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_files):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_list = []
        self.label_list = []

        for csv_file in csv_files:
            csv_reader = csv.reader(open(csv_file))
            for row in csv_reader:
                folder = '{}-cam0-rgb'.format(row[0])
                file_name = '{}-{:0>3s}.png'.format(folder, row[1])
                label = int(row[2])
                file_dir = os.path.join(folder, file_name)

                self.image_list.append(file_dir)
                self.label_list.append(label)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return image, label

# 示例用法，这里修改路径
root_dir = r'F:\大创\防跌倒' # 根目录
csv_files = [
    os.path.join(root_dir, 'urfall-cam0-adls.csv'),
    os.path.join(root_dir, 'urfall-cam0-falls.csv')
]
dataset = CustomDataset(root_dir, csv_files)

# 遍历数据集
for image, label in dataset:
    print(image, label)

