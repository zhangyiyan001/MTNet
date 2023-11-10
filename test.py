# -*- coding:utf-8 -*-
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import time
import numpy as np
from module import AA_andEachClassAccuracy, dis_groundtruth



def result(X_test, test_iter, dataset, device, net):
    net.load_state_dict(torch.load('./models/' + dataset + '.pt'))  # 加载保存好的模型
    print('\n***Start  Testing***\n')
    tick1 = time.time()
    y_test = []
    y_pred = []
    with torch.no_grad():
        for step, (X1, X2, X3, y) in enumerate(test_iter):
            net.eval()
            X1 = X1.to(device)
            X2 = X2.to(device)
            X3 = X3.to(device)
            y = y.to(device)
            y_hat = net(X1, X2, X3)
            y_pred.extend(y_hat.cpu().argmax(dim=1))
            y_test.extend(y.cpu())
            net.train()


    tick2 = time.time()
    Test_time = tick2 - tick1
    print('test_time:', Test_time)
    if dataset == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                    'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
    if dataset == 'Berlin':
        target_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil',
                        'Allotment', 'Commercial Area', 'Water']
    if dataset == 'Trento':
        target_names = ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    if dataset == 'MUUFL':
        target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                        'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    classification = classification_report(np.array(y_test), np.array(y_pred), target_names=target_names, digits=4)
    oa = accuracy_score(np.array(y_test), np.array(y_pred))
    confusion = confusion_matrix(np.array(y_test), np.array(y_pred))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.array(y_test), np.array(y_pred))
    print(oa, aa, kappa, each_acc)
    return oa, aa, kappa, each_acc


