#训练出的模型参数大小22.8MB
#使用了UR Fall Detection Dataset
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def plot_loss_and_accuracy(train_losses, test_losses, test_accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, label='Training Loss', color=color)
    ax1.plot(test_losses, label='Testing Loss', linestyle='--', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(test_accuracies, label='Testing Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Training Loss and Testing Accuracy')
    plt.grid(True)
    plt.show()

class InceptionModule(nn.Module): 
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        # 1x1 Convolution branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 Convolution followed by 3x3 Convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 Convolution followed by 5x5 Convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # Max pooling followed by 1x1 Convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):  #注意实例化时,类别个数为2
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


model = GoogLeNet(2) #实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #设置训练平台
model = model.to(device) #转移

def init_weights(m):  #定义初始化权重函数
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights) #随机初始化权重

#创建数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
        ])
        self.image_list = []
        self.label_list = []

        # 读取摔倒样本
        fall_dir = os.path.join(root_dir, 'fall')  #修改摔倒类所在文件夹名称
        fall_images = os.listdir(fall_dir)
        for img_name in fall_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(fall_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(1)  # 摔倒的标签为1

        # 读取未摔倒样本
        adl_dir = os.path.join(root_dir, 'unfall') #修改未摔倒类所在文件夹的名称
        adl_images = os.listdir(adl_dir)
        for img_name in adl_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(adl_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(0)  # 未摔倒的标签为0

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 加载图像并进行预处理
        image = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return image, label


trainset = CustomDataset(r'C:\Users\Administrator\Desktop\1\train')
train_loader = DataLoader(trainset, batch_size=16, shuffle=True) #同理可设置验证集和测试集
testset = CustomDataset(r'C:\Users\Administrator\Desktop\1\test')
test_loader = DataLoader(testset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()  #定义损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #定义优化器

train_losses = []  # 用于记录每次训练结束后的训练集损失值
test_losses = []  # 用于记录每次训练结束后的测试集损失值
test_accuracies = []  # 用于记录每次训练结束后的测试集准确率

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0

    # 训练阶段
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_epoch_loss = running_loss / len(train_loader)
    train_losses.append(train_epoch_loss)

    # 测试阶段
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_epoch_loss = running_loss / len(test_loader)
        test_losses.append(test_epoch_loss)

        accuracy = correct / total
        test_accuracies.append(accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# 画图
plot_loss_and_accuracy(train_losses, test_losses, test_accuracies)

torch.save(model.state_dict(), r'C:\Users\Administrator\Desktop\model_weights.pth') #保存模型参数
#model.load_state_dict(torch.load(r'C:\Users\Administrator\Desktop\model_weights.pth')) #实例化模型后加载参数