# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年09月22日
"""
import numpy as np
import scipy.io as sio
import os
from module import split_train_test_set, pixel_select, GetImageCubes, GetImageCubes_all
import torch
import torch.utils.data as Data
import random
from sklearn.decomposition import PCA
from einops import rearrange




def flip(*arrays):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        arrays = [np.fliplr(arr) for arr in arrays]
    if vertical:
        arrays = [np.flipud(arr) for arr in arrays]
    return arrays

def rotate(*arrays):
    rotate = np.random.random() > 0.5
    if rotate:
        angle = np.random.choice([1, 2, 3])
        arrays = [np.rot90(arr, k=angle) for arr in arrays]
    return arrays

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return alpha * data + beta * noise



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def load_data(dataset):
    #data_path = r'/home/ubuntu/dataset_RS/Multisource/data'
    data_path = r'D:\Program Files (x86)\Anaconda\jupyter_path\dataset'
    if dataset == 'Houston': #HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)=
        HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2013/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2013/LiDAR.mat'))['LiDAR']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TRLabel.mat'))['TRLabel']
        Test_data = sio.loadmat(os.path.join(data_path, 'Houston2013/TSLabel.mat'))['TSLabel']
        GT = sio.loadmat(os.path.join(data_path, 'Houston2013/gt.mat'))['gt']

    if dataset == 'Berlin':
        HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_HS_LR.mat'))['data_HS_LR']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_SAR_HR.mat'))['data_SAR_HR']
        Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TrainImage.mat'))['TrainImage']
        Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TestImage.mat'))['TestImage']

    if dataset == 'Trento':
        HSI_data = sio.loadmat(os.path.join(data_path, 'Trento/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Trento/LiDAR.mat'))['LiDAR']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Trento/TRLabel.mat'))['TRLabel']
        Test_data = sio.loadmat(os.path.join(data_path, 'Trento/TSLabel.mat'))['TSLabel']
        GT = sio.loadmat(os.path.join(data_path, 'Trento/gt.mat'))['gt']

    if dataset == 'MUUFL':
        HSI_data = sio.loadmat(os.path.join(data_path, 'MUUFL/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'MUUFL/LiDAR.mat'))['LiDAR']
        Train_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_train_150.mat'))['mask_train']
        Test_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_test_150.mat'))['mask_test']
        GT = sio.loadmat(os.path.join(data_path, 'MUUFL/gt.mat'))['gt']
        GT[GT==-1] = 0

    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def shape_new_X(x):
    x[:, 1::2, :] = np.flip(x[:, 1::2, :], axis=[-1])
    return x

def neighbour_band1(x, patch):  # x (32, 169, 144)
    x = rearrange(x, 'n c h w -> n (h w) c')
    (b, n, d) = x.shape
    x = shape_new_X(x)
    x_new = np.zeros((b, n+2*patch, d)) #(32, 169+6, 144)
    x_new[:, patch:patch+n, : ] = x
    x_new[:, :patch, :] = np.flip(x[:, :patch, :], axis=[-1])
    x_new[:, patch+n:, :] = np.flip(x[:, n-patch:, :], axis=[-1])
    # x_new_rev = np.flip(x_new, axis=[-2, -1])
    x1 = np.zeros((b, n, patch, d), dtype=float)
    for i in range(n):
        x1[:, i, :, :] = x_new[:, i:i + patch, :]
    out = rearrange(x1, 'b n p c -> b (n p) c')
    return out

def neighbour_band2(x, patch):  # x (32, 169, 144)
    x = rearrange(x, 'n c h w -> n c w h')
    x = rearrange(x, 'n c w h -> n (w h) c')
    (b, n, d) = x.shape
    x = shape_new_X(x)
    x_new = np.zeros((b, n+2*patch, d)) #(32, 169+6, 144)
    x_new[:, patch:patch+n, : ] = x
    x_new[:, :patch, :] = np.flip(x[:, :patch, :], axis=[-1])
    x_new[:, patch+n:, :] = np.flip(x[:, n-patch:, :], axis=[-1])
    # x_new_rev = np.flip(x_new, axis=[-2, -1])
    x1 = np.zeros((b, n, patch, d), dtype=float)
    for i in range(n):
        x1[:, i, :, :] = x_new[:, i:i + patch, :]
    out = rearrange(x1, 'b n p c -> b (n p) c')
    return out

# x1 = np.random.rand(2832, 144, 11, 11)
# print(neighbour_band(x1, patch=3).shape)
def pixel_selection(gt, train_pixel):
    test_pixels = gt.copy()
    kinds = np.unique(gt).shape[0]-1
    for i in range(kinds):
        num = np.sum(train_pixel==(i+1))
        val_num = int(num * 0.7)
        temp1 = np.where(train_pixel==(i+1))
        temp2 = random.sample(range(num), val_num)
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本
    train_pixels = gt - test_pixels
    return train_pixels, test_pixels


def gain_neighborhood_band(x_train, band, band_patch=3, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)#(2832, 121, 144)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float) #(2832, 121*3, 144)
    # 中心区域
    x_train_band[:, nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, :i+1] = x_train_reshape[:, :, band-i-1:]
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, i+1:] = x_train_reshape[:, :, :band-i-1]
        else:
            x_train_band[:, i:(i+1), :(nn-i)] = x_train_reshape[:, 0:1, (band-nn+i):]
            x_train_band[:, i:(i+1), (nn-i):] = x_train_reshape[:, 0:1, :(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, :band-i-1] = x_train_reshape[:, :, i+1:]
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, band-i-1:] = x_train_reshape[:, :, :i+1]
        else:
            x_train_band[:, (nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:, 0:1, :(i+1)]
            x_train_band[:, (nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:, 0:1, (i+1):]
    return x_train_band

def generater(X_hsi, X_lidar, train_pixels, test_pixels, GT, batch_size, windowSize):
    #train_pixels, test_pixels = pixel_selection(GT, train_pixels)
    x_train_hsi, y_train_hsi = GetImageCubes(input_data=X_hsi, pixels_select=train_pixels, windowSize=windowSize) #(2832, 11, 11, 144)
    x_test_hsi, y_test_hsi = GetImageCubes(input_data=X_hsi, pixels_select=test_pixels, windowSize=windowSize) #(2832, 11, 11, 144)


    x_train_hsi_band = gain_neighborhood_band(x_train=x_train_hsi, band=144, band_patch=3, patch=9)
    x_test_hsi_band = gain_neighborhood_band(x_train=x_test_hsi, band=144, band_patch=3, patch=9)

    x_train_lidar, y_train_lidar = GetImageCubes(input_data=X_lidar, pixels_select=train_pixels, windowSize=windowSize) #(2832, 1, 11, 11)
    x_test_lidar, y_test_lidar = GetImageCubes(input_data=X_lidar, pixels_select=test_pixels, windowSize=windowSize)

    TRAIN_SIZE = x_train_hsi.shape[0]
    TEST_SIZE = x_test_hsi.shape[0]
    TOTAL_SIZE = x_train_hsi.shape[0] + x_test_hsi.shape[0]

    print('X_train:{}\nX_test:{}\nX_all:{}'.format(x_train_hsi.shape[0], x_test_hsi.shape[0], x_train_hsi.shape[0]+x_test_hsi.shape[0]))

    hsi_train_tensor = torch.from_numpy(x_train_hsi).type(torch.FloatTensor)
    hsi_test_tensor = torch.from_numpy(x_test_hsi).type(torch.FloatTensor)

    hsi_train_band_tensor = torch.from_numpy(x_train_hsi_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    hsi_test_band_tensor = torch.from_numpy(x_test_hsi_band.transpose(0, 2, 1)).type(torch.FloatTensor)

    lidar_train_tensor = torch.from_numpy(x_train_lidar).type(torch.FloatTensor)
    lidar_test_tensor = torch.from_numpy(x_test_lidar).type(torch.FloatTensor)


    y_train = torch.from_numpy(y_train_hsi).type(torch.int64)
    y_test = torch.from_numpy(y_test_hsi).type(torch.int64)

    torch_train = Data.TensorDataset(hsi_train_tensor, lidar_train_tensor, hsi_train_band_tensor, y_train)
    torch_test = Data.TensorDataset(hsi_test_tensor, lidar_test_tensor, hsi_test_band_tensor, y_test)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    return TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, x_test_hsi

def normalize(X, type):
    x = np.zeros(shape=X.shape, dtype='float32')
    if type == 1:
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)

    if type == 2:
        for i in range(X.shape[2]):
            min = np.min(X[:, :, i])
            max = np.max(X[:, :, i])
            scale = max - min
            if scale == 0:
                scale = 1e-5
            x[:, :, i] = (X[:, :, i] - min) / scale
    return x

