import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# macOS에서 torch.multiprocessing 사용 시 'fork'로 설정
import torch.multiprocessing
torch.multiprocessing.set_start_method('fork')

# print(torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 데이터 로드 및 확인
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

for images, labels in train_loader:
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('{}\t'.format(classes[labels[j]]) for j in range(4)))
    break  # This is just to show the first batch; you can remove it to iterate over all batches

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

# loss function and
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i % 2000 == 1999):
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
image_batch, text_batch = next(iter(train_loader))

# imshow(torchvision.utils.make_grid(images))
# print('Ground truth: ', ' '.join('\t{}'.format(classes[labels[j]]) for j in range(4)))
print('Ground truth: ', ' '.join('\t{}'.format(classes[labels[j]]) for j in range(4)))

# 새로운 모델 인스턴스를 만들고 저장된 가중치를 불러옴
loaded_net = Net().to(device)
loaded_net.load_state_dict(torch.load(PATH))
loaded_net.eval()  # 모델을 평가 모드로 전환

# 이미지를 GPU로 이동
images = images.to(device)

# Model Test
outputs = loaded_net(images)
_, predicted = torch.max(outputs, 1)

# 예측 결과 표시
print('Predicted: ', ' '.join('\t{}'.format(classes[predicted[j]]) for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(100 * correct / total)

# 어떤 것을 잘 분류했고, 잘 못했는지 확인
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print("Accuracy of {}: {}%".format(classes[i], 100 * class_correct[i] / class_total[i]))