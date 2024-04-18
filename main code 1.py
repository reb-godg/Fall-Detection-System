import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt

# 定义训练数据集类
class TrainDataset(Dataset):
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
        fall_dir = os.path.join(root_dir, 'fall-train')
        fall_images = os.listdir(fall_dir)
        for img_name in fall_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(fall_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(1)  # 摔倒的标签为1

        # 读取未摔倒样本
        adl_dir = os.path.join(root_dir, 'adl-train')
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

# 创建训练数据加载器
train_dataset = TrainDataset('E:/code/datasets/trainsets')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 创建测试数据集
class TestDataset(Dataset):
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
        fall_dir = os.path.join(root_dir, 'fall-tests')
        fall_images = os.listdir(fall_dir)
        for img_name in fall_images:
            if img_name.endswith('.png'):
                img_path = os.path.join(fall_dir, img_name)
                self.image_list.append(img_path)
                self.label_list.append(1)  # 摔倒的标签为1

        # 读取未摔倒样本
        adl_dir = os.path.join(root_dir, 'adl-tests')
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

# 创建测试数据加载器
test_dataset = TestDataset('E:/code/datasets/testsets')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 定义模型
model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 2)  # 修改最后一层分类器

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试循环
num_epochs = 3
train_loss_history = []
test_loss_history = []
accuracy_history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # 计算训练损失和准确率
    train_loss = train_loss / len(train_dataset)
    train_loss_history.append(train_loss)

    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试损失和准确率
    test_loss = test_loss / len(test_dataset)
    test_loss_history.append(test_loss)
    accuracy = 100 * correct / total
    accuracy_history.append(accuracy)

    # 打印训练和测试结果
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.2f}%")

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')

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
