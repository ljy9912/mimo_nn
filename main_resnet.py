'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, reproduc
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--work_dir', default='./work_dirs', type=str, help='Directory to save data and log')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--act_fun', default='ReLU', type=str, help='Activation function')
parser.add_argument('--angle_tan', default=0.84, type=int, help='tangent value of the half-apex angle')
args = parser.parse_args()

reproduc(args.seed)

if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
if not os.path.isdir(args.work_dir + '/runs'):
    os.mkdir(args.work_dir + '/runs')
writer = SummaryWriter(args.work_dir + '/runs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18(act_fun=args.act_fun, angle_tan=args.angle_tan)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-4)

train_acc_list = np.array([])
valid_acc_list = np.array([])
train_loss_list = np.array([])
valid_loss_list = np.array([])

# Training
def train(epoch):
    global train_acc_list
    global train_loss_list
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss = train_loss / (batch_idx + 1)
    train_acc = correct * 100. / total
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    train_loss_list = np.append(train_loss_list, train_loss)
    train_acc_list = np.append(train_acc_list, train_acc)


def test(epoch):
    global best_acc
    global valid_acc_list
    global valid_loss_list
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        valid_loss = test_loss / (batch_idx+1)
        valid_acc = 100. * correct / total
        writer.add_scalar('Loss/test', valid_loss, epoch)
        writer.add_scalar('Accuracy/test', valid_acc, epoch)
        valid_loss_list = np.append(valid_loss_list, valid_loss)
        valid_acc_list = np.append(valid_acc_list, valid_acc)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.work_dir + '/checkpoint'):
            os.mkdir(args.work_dir + '/checkpoint')
        torch.save(state, args.work_dir + '/checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
    np.savetxt(args.work_dir + '/train_acc_list.txt', train_acc_list, delimiter=',')
    np.savetxt(args.work_dir + '/valid_acc_list.txt', valid_acc_list, delimiter=',')
    np.savetxt(args.work_dir + '/valid_loss_list.txt', valid_loss_list, delimiter=',')
    np.savetxt(args.work_dir + '/train_loss_list.txt', train_loss_list, delimiter=',')
