import function_connective
import featureExtract
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from sklearn import preprocessing
import pandas as pd
import os
import time


BATCH_SIZE = 16


# 制作数据样本库
class myfunctionConnectiveDatasets(torch.utils.data.Dataset):
    def __init__(self, file_dir, file_type):
        super().__init__()
        self.file_name = []
        self.targets = []
        for root, dirs, files in os.walk(file_dir, topdown=False):
            for file in files:
                if root[-1] == '1':
                    self.targets.append(0)
                elif root[-1] == '2':
                    self.targets.append(1)
                elif root[-1] == '3':
                    self.targets.append(2)
                elif root[-1] == '4':
                    self.targets.append(3)
                else:
                    self.targets.append(4)
                if file_type in file:
                    self.file_name.append(os.path.join(root, file))
        self.targets = torch.from_numpy(np.array(self.targets)).long()

    def __getitem__(self, item):
        data = pd.read_csv(self.file_name[item])
        data = data.values[:, 1:]
        data = featureExtract.get_eeg_four_frequency_band(data)
        data = data[1:3]
        data = torch.from_numpy(function_connective.phase_locked_matrix(data)).float()
        target = self.targets[item]

        return data, target

    def __len__(self):
        return len(self.file_name)


class myLstmDatasets(torch.utils.data.Dataset):
    def __init__(self, file_dir, file_type):
        super().__init__()
        self.file_name = []
        self.targets = []
        for root, dirs, files in os.walk(file_dir, topdown=False):
            for file in files:
                if root[-1] == '1':
                    self.targets.append(0)
                elif root[-1] == '2':
                    self.targets.append(1)
                elif root[-1] == '3':
                    self.targets.append(2)
                elif root[-1] == '4':
                    self.targets.append(3)
                else:
                    self.targets.append(4)
                if file_type in file:
                    self.file_name.append(os.path.join(root, file))
        self.targets = torch.from_numpy(np.array(self.targets)).float().unsqueeze(-1).repeat(1, 58).unsqueeze(1)  # 为了后面把标签和训练数据一起打包

    def __getitem__(self, item):
        data = pd.read_csv(self.file_name[item])
        data = data.values[:, 1:]
        data = featureExtract.get_eeg_four_frequency_band(data)
        data = data[1:2]
        feature = []
        for band in range(len(data)):
            for i in range(len(data[0, 0]) // 50):
                data_ = data[band, :, i*50:(i+1)*50]
                feature1 = featureExtract.get_eeg_power(data_)
                feature2 = featureExtract.get_eeg_power_spectral_entropy(data_)
                feature.append(np.hstack((feature1, feature2)))

        feature = torch.from_numpy(np.array(feature)).float()
        target = self.targets[item]

        return torch.cat((target, feature), dim=0)  # 标签和训练数据一起打包


    def __len__(self):
        return len(self.file_name)


# 构造神经网络分类器并训练
class eegcnn(nn.Module):
    def __init__(self):
        super(eegcnn, self).__init__()
        self.conv1 = nn.Sequential(
            # n = ((in_channels - kernel + 2 * padding) / stride) + 1
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=3, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # batch_size * 32 * 8 * 8
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 3, groups=16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # batch_size * 64 * 3 * 3
            nn.BatchNorm2d(64),
        )

        self.linear01 = nn.Sequential(
            # nn.Dropout(p=0.25),
            nn.Linear(64 * 3 * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 3),
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)  # 将平面的(即有形状的矩阵)平展
        res = self.linear01(res)


        return res


class eeglstm(nn.Module):
    def __init__(self):
        super(eeglstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=58,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 3)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:,-1,:])

        return out


if __name__ == '__main__':

    import data_dir

    model = 'cnn'


    if model == 'cnn':
        net = eegcnn()
        # net = eegcnn().cuda()

        datasets = myfunctionConnectiveDatasets(data_dir.preprocess_dir, '.csv')

        train_datasets, test_datasets = torch.utils.data.random_split(dataset=datasets, lengths=[int(len(datasets) * 0.7), len(datasets) - int(len(datasets) * 0.7)])

        train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE*10, shuffle=True)

        loss_func = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=0.05)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1, gamma=0.85)

        for epoch in range(5):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                pred = net(batch_x)
                loss = loss_func(pred, batch_y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if step % 3 == 0:
                    with torch.no_grad():
                        for _, (test_x, test_y) in enumerate(test_loader):
                            # test_x, test_y = test_x.cuda(), test_y.cuda()
                            test_output = net(test_x)
                            accuracy = (torch.argmax(test_output, dim=1) == test_y).float().numpy().sum() / (
                                BATCH_SIZE*10) * 100
                            print(
                                'epoch:{} | step:{:4d} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss, accuracy))
                            break

            schedule.step(epoch)

        print('模型训练总用时：', time.process_time(), 's')
        # 保存整个模型
        torch.save(net, data_dir.project_dir + r'\eegcnn.pt')
    else:
        # *******************************LSTM******************************************#
        net = eeglstm()
        # net = eeglstm().cuda()

        datasets = myLstmDatasets(data_dir.preprocess_dir, '.csv')

        train_datasets, test_datasets = torch.utils.data.random_split(dataset=datasets, lengths=[int(len(datasets) * 0.7), len(datasets) - int(len(datasets) * 0.7)])

        # 此是为了训练变长序列数据################################
        def collate_fn(data):
            data.sort(key=lambda x: len(x), reverse=True)
            data_length = [len(sq) for sq in data]
            data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
            return data, data_length


        train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE * 10, shuffle=True,
                                                  collate_fn=collate_fn)
        #######################################################

        loss_func = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=0.05)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1, gamma=0.85)

        for epoch in range(3):
            for step, (batch, batch_x_len) in enumerate(train_loader):
                batch_x = batch[:,1:,:]
                batch_y = batch[:,0, 0].long()
                batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, [i-1 for i in batch_x_len], batch_first=True)
                # batch_x_pack, batch_y = batch_x_pack.cuda(), batch_y.cuda()
                pred = net(batch_x_pack)
                loss = loss_func(pred, batch_y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if step % 3 == 0:
                    with torch.no_grad():
                        for _, (test, test_x_len) in enumerate(test_loader):
                            test_x = test[:,1:,:]
                            test_y = test[:,0, 0].long()
                            test_x_pack = rnn_utils.pack_padded_sequence(test_x, [i-1 for i in test_x_len], batch_first=True)
                            # test_x_pack, test_y = test_x_pack.cuda(), test_y.cuda()
                            test_output = net(test_x_pack)
                            accuracy = (torch.argmax(test_output, dim=1) == test_y).float().numpy().sum() / (
                                    BATCH_SIZE * 10) * 100
                            print(
                                'epoch:{} | step:{:4d} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss,
                                                                                             accuracy))
                            break

            schedule.step(epoch)

        print('模型训练总用时：', time.process_time(), 's')
        # 保存整个模型
        torch.save(net, data_dir.project_dir + r'\eeglstm.pt')