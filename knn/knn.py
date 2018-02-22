"""
这里我用ndarray把predict函数里的部分重写了一下
更简洁了一些
"""


import pandas as pd
import numpy as np
import cv2
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features


def Predict(testset, trainset, train_labels, k=5):
    predict = []
    count = 0

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        count += 1
        if count % 5000 == 0:
            print(count)

        knn_list = np.zeros((1, 2))    # 初始化，存放当前k个最近邻居

        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            knn_list = np.append(knn_list, [[dist, label]], axis=0)

        # 剩下的点
        for i in range(k, len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离

            # 寻找10个邻近点中距离最远的点
            max_index = np.argmax(knn_list[:, 0])
            max_dist = np.max(knn_list[:, 0])

            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = [dist, label]

        # 上面代码计算全部运算完之后，即说明已经找到了离当前test_vec最近的10个train_vec
        # 统计选票
        class_total = 10
        class_count = [0 for i in range(class_total)]
        for dist, label in knn_list:
            class_count[int(label)] += 1

        # 找出最大选票数
        label_max = max(class_count)

        # 最大选票数对应的class
        predict.append(class_count.index(label_max))

    return np.array(predict)


if __name__ == '__main__':

    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values
    imgs = data[:, 1:]
    labels = data[:, 0]

    features = get_hog_features(imgs)

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels =  \
        train_test_split(features, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    k = 10  # 设置k的大小

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    print('knn do not need to train')
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = Predict(test_features, train_features, train_labels, k)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
