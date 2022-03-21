import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./games.csv')
nominals = ['away_final_score', 'away_team', 'date', 'elapsed_time', 'home_final_score', 'home_team', 'umpire_1B', 'umpire_2B', 'umpire_3B', 'umpire_HP', 'venue_name', 'weather','wind', 'delay']
numerics = ['away_final_score', 'elapsed_time', 'home_final_score', 'delay']

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

# 绘制直方图
def VisualizeByHist():
    plt.figure(figsize = (15, 15))
    plt.subplot(2, 2, 1)
    plt.hist(dataset['away_final_score'].values.tolist())
    plt.title('away_final_score')
    plt.subplot(2, 2, 2)
    plt.hist(dataset['elapsed_time'].values.tolist())
    plt.title('elapsed_time')
    plt.subplot(2, 2, 3)
    plt.hist(dataset['home_final_score'].values.tolist())
    plt.title('home_final_score')
    plt.subplot(2, 2, 4)
    plt.hist(dataset['delay'].values.tolist())
    plt.title('delay')
    plt.show()

# 绘制盒图
def VisualizeByBoxplot():
    plt.figure(figsize = (15, 15))
    plt.subplot(2, 2, 1)
    plt.boxplot(dataset['away_final_score'].values.tolist())
    plt.title('away_final_score')
    plt.subplot(2, 2, 2)
    plt.boxplot(dataset['elapsed_time'].values.tolist())
    plt.title('elapsed_time')
    plt.subplot(2, 2, 3)
    plt.boxplot(dataset['home_final_score'].values.tolist())
    plt.title('home_final_score')
    plt.subplot(2, 2, 4)
    plt.boxplot(dataset['delay'].values.tolist())
    plt.title('delay')
    plt.show()

if __name__ == '__main__':
    PrintNominalData()
    PrintNumericAttributes()
    VisualizeByHist()
    VisualizeByBoxplot()