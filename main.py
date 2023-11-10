# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年09月22日
"""
import torch
from dataset import load_data, generater, normalize, setup_seed, applyPCA
from train_hsi import train_hsi


setup_seed(seed=2021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = 'Houston'
HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset)

HSI_data = normalize(HSI_data, type=1)
LiDAR_data = normalize(LiDAR_data, type=1)

TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, X_test = generater(HSI_data,
                                                                     LiDAR_data,
                                                                     Train_data,
                                                                     Test_data,
                                                                     GT=GT,
                                                                     batch_size=64,
                                                                     windowSize=9)

model = train_hsi(dataset=dataset,
                  train_iter=train_iter,
                  test_iter=test_iter,
                  device=device,
                  epoches=200,
                  ITER=1,
                  TRAIN_SIZE=TRAIN_SIZE,
                  TEST_SIZE=TEST_SIZE,
                  TOTAL_SIZE=TOTAL_SIZE,
                  X_test=X_test)

