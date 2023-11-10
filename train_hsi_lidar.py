# -*- coding:utf-8 -*-
import torch
import torch.optim as optim
import time
from module import weight_init
from test import result
from shared_specific_net import MTNet


def train_hsi_lidar(dataset, train_iter, test_iter, device, epoches, ITER, TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, X_test):
    for index_iter in range(ITER):
        train_loss_list = []
        train_acc_list = []
        net = MTNet(channels=144,
                    num_patches=81,
                    dim=144,
                    depth=2,
                    heads=2,
                    dim_head=32,
                    mlp_dim=64,
                    num_classes=15,
                    dropout=0.1).to(device)
        net.apply(weight_init)  # 网络权重初始化
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0)
        loss_hsi = torch.nn.CrossEntropyLoss()

        print('\niter:', index_iter + 1)
        print('TRAIN_SIZE: ', TRAIN_SIZE)
        print('TEST_SIZE: ', TEST_SIZE)
        print('TOTAL_SIZE: ', TOTAL_SIZE)
        print(
            '--------------------------------------------------Training on {}--------------------------------------------------\n'.format(
                device))
        start = time.time()
        for epoch in range(epoches):
            train_acc_sum, train_loss_sum = 0.0, 0.0
            time_epoch = time.time()
            #lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hsi, T_max=200, last_epoch=-1)
            lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=epoches//50, gamma=0.1, last_epoch=-1)
            for step, (X1, X2, X3, y) in enumerate(train_iter):
                X1 = X1.to(device)
                X2 = X2.to(device)
                X3 = X3.to(device)
                y = y.to(device)
                # 前向传播
                y_hat = net(X1, X2, X3)
                l = loss_hsi(y_hat, y.long())
                # 反向传播及优化
                optimizer.zero_grad()  # 梯度清零
                l.backward()
                optimizer.step()
                train_loss_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=-1) == y.to(device)).float().sum().cpu().item()
            lr_adjust.step()
            print('epoch %d, train loss %.6f, train acc %.4f, time %.2f sec' % (
                epoch + 1, train_loss_sum / len(train_iter),
                train_acc_sum / len(train_iter.dataset),
                time.time() - time_epoch))
            train_loss_list.append(train_loss_sum / len(train_iter))  # / batch_count)
            train_acc_list.append(train_acc_sum / len(train_iter.dataset))
            if train_loss_list[-1] <= min(train_loss_list):
                torch.save(net.state_dict(), './models/' + dataset + '.pt')
                print('**Successfully Saved Best model parametres!***\n')  # 保存在训练集上损失值最好的模型效果

            result(X_test=X_test, test_iter=test_iter, dataset=dataset, device=device, net=net)


    return net
