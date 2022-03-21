import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('./winemag-data-130k-v2.csv')
nominals = ['country', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title', 'variety', 'winery']
numerics = ['points', 'price']

# 输出标称数据（每个可能取值的频数）
def PrintNominalData():
    for i in nominals:
        print(i + ':')
        print(dataset[i].value_counts())
        print('\n\n')

# 输出数值属性（五数概括与缺失值个数）
def PrintNumericAttributes():
    for i in numerics:
        temp = dataset[i].dropna()
        print(i + ':')
        print('最小值：      ' + str(min(temp)))
        print('第1四分位数： ' + str(np.percentile(temp, 25)))
        print('中位数：      ' + str(np.percentile(temp, 50)))
        print('第3四分位数： ' + str(np.percentile(temp, 75)))
        print('最大值：      ' + str(max(temp)))
        print('缺失值：      ' + str(dataset[i].isnull().sum()))
        print('\n\n')

# 数据可视化
def Visualize():
    plt.figure(figsize = (20, 10))
    plt.subplot(1, 2, 1)
    plt.hist(dataset['price'].values.tolist())
    plt.title('price')
    plt.subplot(1, 2, 2)
    plt.boxplot(dataset['price'].values.tolist())
    plt.title('price')
    plt.show()

# 将缺失部分剔除
def DeleteMissingData():
    dataset = dataset.dropna()

# 用最高频率值来填补缺失值
def ReplaceWithMode():
    dataset['price'].fillna(int(dataset['price'].mode()[0]), inplace = True)

# 通过属性的相关关系来填补缺失值
def ReplaceWithLR():
    # 划分训练集
    trainset = dataset.dropna(subset = ['price'])
    x_train = trainset['points'].to_frame()
    y_train = trainset['price'].to_frame()
    
    # 训练线性回归模型，计算出截距与回归系数
    model = LinearRegression()
    model.fit(x_train, y_train)
    intercept = int(model.intercept_)
    coef = round(model.coef_, 1)

    # 用线性回归模型的预测值填补缺失值
    predictset = dataset[np.isnan(dataset['price'])]
    x_pred = predictset['points']
    y_pred = predictset['price']
    for i in y_pred.keys():
        dataset.loc[i, 'price'] = dataset.loc[i, 'points'] * coef + intercept

# 通过数据对象之间的相似性来填补缺失值
def ReplaceWithKNN():
    # 划分训练集
    trainset = dataset.dropna(subset = ['price'])
    x_train = trainset['points'].to_frame()
    y_train = trainset['price'].to_frame()

    # 训练KNN模型
    model = KNeighborsClassifier(n_neighbors = 3, weights = "distance")
    model.fit(x_train, np.ravel(y_train))

    # 用KNN模型的预测值填补缺失值
    predictset = dataset[np.isnan(dataset['price'])]
    x_pred = predictset['points']
    y_pred = predictset['price']
    for i in y_pred.keys():
        dataset.loc[i, 'price'] = model.predict(dataset.loc[i, 'points'])

if __name__ == '__main__':
    PrintNominalData()
    PrintNumericAttributes()
    Visualize()

    print("请输入1，2，3或4以选择缺失值处理方式\n")
    ipt = input()
    if ipt == 1:
        DeleteMissingData()
        Visualize()
    elif ipt == 2:
        ReplaceWithMode()
        Visualize()
    elif ipt == 3:
        ReplaceWithLR()
        Visualize()
    elif ipt == 4:
        ReplaceWithKNN()
        Visualize()
    else:
        print("Invalid Input!\n")