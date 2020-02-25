import functionConnective
import filter
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.datasets
import numpy as np
import pandas as pd
import os


BATCH_SIZE = 16


# 制作数据样本库
class mydatasets(torch.utils.data.Dataset):
    def __init__(self, file_dir, file_type):
        super().__init__()
        self.file_name = []
        self.targets = []
        for root, dirs, files in os.walk(file_dir, topdown=False):
            for file in files:
                if root[-4:] == 'fast':
                    self.targets.append(0)
                elif root[-4:] == 'medi':
                    self.targets.append(1)
                else:
                    self.targets.append(2)
                if file_type in file:
                    self.file_name.append(os.path.join(root, file))
        self.targets = torch.from_numpy(np.array(self.targets)).long()

    def __getitem__(self, item):
        data = pd.read_csv(self.file_name[item])
        data = data.values[:30, 1:]
        data = filter.get_eeg_signal_four_frequency_band(data)
        data = torch.from_numpy(functionConnective.phase_locked_matrix(data)).float()
        target = self.targets[item]

        return data, target

    def __len__(self):
        return len(self.file_name)


# 构造神经网络分类器并训练
class eegnet(nn.Module):
    def __init__(self):
        super(eegnet, self).__init__()
        self.conv1 = nn.Sequential(
            # n = ((in_channels - kernel + 2 * padding) / stride) + 1
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=3, padding=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 32 * 6 * 6
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 3, 3, groups=32),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # batch_size * 64 * 2 * 2
        )
        self.linear01 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)  # 将平面的(即有形状的矩阵)平展
        res = self.linear01(res)


        return res

if __name__ == '__main__':
    net = eegnet()
    datasets = mydatasets(r'D:\Files\SJTU\Study\MME_Lab\Teacher_Lu\click_number\eeg\process1.0', '.csv')

    train_datasets, test_datasets = torch.utils.data.random_split(dataset=datasets, lengths=[int(len(datasets)*0.8), len(datasets)-int(len(datasets)*0.8)])

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE,shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            pred = net(batch_x)
            loss = loss_func(pred, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 20 == 0:
                with torch.no_grad():
                    for _, (test_x, test_y) in enumerate(test_loader):
                        # test_x, test_y = test_x.cuda(), test_y.cuda()
                        test_output = net(test_x)
                        accuracy = (torch.argmax(test_output, dim=1) == test_y).float().numpy().sum() / BATCH_SIZE
                        print('epoch:{} | step:{:4d} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss, accuracy))
                        break
            # test_x = test_x[:10]
            # test_output = net(test_x)
            # pred = torch.argmax(test_output, 1).cpu().data.numpy().squeeze()
            # print(pred, '\n', test_y[:10])