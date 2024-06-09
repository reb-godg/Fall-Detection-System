import csv
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


# 定义训练数据集类
class TrainDataset(Dataset):
    def __init__(self, root_dir, csv_files):
        self.root_dir = root_dir
        self.transform = transforms.Compose([  # 图像预处理
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
        ])
        self.image_list = []  # 存储图像文件路径
        self.label_list = []  # 存储对应的标签

        # 读取CSV文件
        for csv_file in csv_files:
            csv_reader = csv.reader(open(csv_file))
            for row in csv_reader:
                folder = '{}-cam0-rgb'.format(row[0])
                file_name = '{}-{:0>3s}.png'.format(folder, row[1])
                label = int(row[2]) 
                if label == 0:
                    continue  # 跳过标签为0的样本
                if label == -1:
                    label = 0  # 将标签-1转换为0

                file_dir = os.path.join(folder, file_name)

                self.image_list.append(file_dir)  # 添加图像路径到列表
                self.label_list.append(label)  # 添加标签到列表

    def __len__(self):
        return len(self.image_list)  # 返回数据集的大小

    def __getitem__(self, idx):
        # 加载图像并进行预处理
        image = self.transform(Image.open(os.path.join(self.root_dir, self.image_list[idx])))
        label = self.label_list[idx]
        return image, label  # 返回图像及其对应的标签

# 创建训练数据加载器
# 示例用法，这里修改路径
root_dir = r'E:\code\data\traindata'  # 根目录
csv_files = [
    os.path.join(root_dir, 'urfall-cam0-adls.csv'),
    os.path.join(root_dir, 'urfall-cam0-falls.csv')
]

# 创建训练数据加载器
train_dataset = TrainDataset(root_dir, csv_files)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 设置批处理大小为8，并打乱数据


# 创建测试数据集类
class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([  # 数据预处理
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.RandomRotation(degrees=(-45, 45)),  # 随机旋转-45度到45度之间的角度
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
        ])
        self.image_list = []  # 存储图像文件路径
        self.label_list = []  # 存储对应的标签

        # 读取摔倒样本
        fall_dir = os.path.join(root_dir, 'fall')
        fall_images = os.listdir(fall_dir)
        for img_name in fall_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(fall_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(1)  # 摔倒的标签为1

        # 读取未摔倒样本
        adl_dir = os.path.join(root_dir, 'adl')
        adl_images = os.listdir(adl_dir)
        for img_name in adl_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(adl_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(0)  # 未摔倒的标签为0

    def __len__(self):
        return len(self.image_list)  # 返回数据集的大小

    def __getitem__(self, idx):
        # 加载图像并进行预处理
        image = self.transform(Image.open(self.image_list[idx]))
        label = self.label_list[idx]
        return image, label  # 返回图像及其对应的标签

# 创建测试数据加载器
test_dataset = TestDataset(r"E:\code\data\testdata")
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)  # 设置批处理大小为256，打乱数据

# 定义模型
model = torchvision.models.mobilenet_v2(weights=True)  # 使用预训练的MobileNetV2模型
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)  # 修改最后一层分类器，使其输出2个类别

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # Adam优化器，学习率为0.001，添加权重衰减

# 训练和测试循环
num_epochs = 1  # 训练轮数
train_loss_history = []
test_loss_history = []
accuracy_history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用
model.to(device)  # 将模型移动到设备（GPU或CPU）

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 学习率调度器，每7轮衰减学习率

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)  # 将图像移动到设备
        labels = labels.to(device)  # 将标签移动到设备

        optimizer.zero_grad()  # 梯度清零

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()  # 更新模型参数

        train_loss += loss.item() * images.size(0)  # 累加训练损失

    # 调整学习率
    scheduler.step()

    # 计算训练损失和准确率
    train_loss = train_loss / len(train_dataset)  # 计算平均训练损失
    train_loss_history.append(train_loss)  # 记录训练损失

    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)  # 将图像移动到设备
            labels = labels.to(device)  # 将标签移动到设备

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)  # 累加测试损失

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试损失和准确率
    test_loss = test_loss / len(test_dataset)  # 计算平均测试损失
    test_loss_history.append(test_loss)  # 记录测试损失
    accuracy = 100 * correct / total  # 计算准确率
    accuracy_history.append(accuracy)  # 记录准确率

    # 打印训练和测试结果
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.2f}%")

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')  # 保存模型参数到文件

# 绘制训练和测试损失函数曲线
plt.plot(range(1, num_epochs+1), train_loss_history, label='Train')
plt.plot(range(1, num_epochs+1), test_loss_history, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确率曲线
plt.plot(range(1, num_epochs+1), accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.show()
