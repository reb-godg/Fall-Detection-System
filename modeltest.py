import os
import cv2
import torch
import torchvision.transforms as transforms
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

'将20行-101行替换为所用模型'
'124行设置训练好的参数路径'
'125行实例化模型，一般只有分类的个数一个参数'
'视频中有动静才抽帧，抽出的帧可设置为在检测完后删除（225行）'
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


def load_frames_from_folder(folder):
    frames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            frame_path = os.path.join(folder, filename)
            frame = cv2.imread(frame_path)
            frames.append(frame)
    return frames

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小
    transforms.ToTensor(),           # 将图片转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图片归一化
])





# 加载已经训练好的模型
model_path = r'C:\Users\Administrator\Desktop\model_weights.pth'
model = GoogLeNet(2)  # 请替换为您自己的模型类
model.load_state_dict(torch.load(model_path))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 文件夹路径
folder_path = 'C:\\Users\\Administrator\\Desktop\\test\\'

# 初始化已推断过的图片集合
inferred_images = set()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 创建BackgroundSubtractor对象
history = 200  # 历史帧数
fgbg = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=False)

# 保存间隔
save_interval = 30
frame_count = 0
save_count = 0

# 持续监视文件夹内新图片的出现
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 对当前帧进行前景提取
    fgmask = fgbg.apply(frame)

    # 对前景进行处理以消除噪音
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 查找前景中的轮廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个大的外接矩形，围住所有运动对象
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)

    if all_contours:
        # 将所有轮廓合并为一个大的轮廓
        all_contours = cv2.convexHull(np.array(all_contours).reshape((-1, 1, 2)))

        # 计算所有轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(all_contours)

        # 绘制矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取矩形框中的像素
        roi = frame[y:y+h, x:x+w]

        # 每隔一定帧数保存图片
        if frame_count % save_interval == 0 and w * h > 40000:   #像素要求
            # 保存图片
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(folder_path, f'frame_{timestamp}.png'), roi)
            save_count += 1

    # 显示帧
    cv2.imshow('frame', frame)

    # 检查文件夹中新图片的出现
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.png')]
    for image_file in image_files:
        # 检查是否已经对该图片进行过推断
        if image_file not in inferred_images:
            # 读取图像
            image = cv2.imread(os.path.join(folder_path, image_file))

            # 预处理图像
            image = preprocess(image)
            image = image.unsqueeze(0)  # 添加 batch 维度

            # 使用模型进行推断
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()

            print(f'Image {image_file}: 类型: {predicted_class}')

            # 将图片添加到已推断过的图片列表中
            inferred_images.add(image_file)

            # 删除已推断过的图片（取消下行注释）
            #os.remove(os.path.join(folder_path, image_file))

    # 按 'q' 键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

# 释放摄像头对象
cap.release()
cv2.destroyAllWindows()

