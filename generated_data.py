import numpy as np
import pandas as pd

def generate_data(n):
    """
    按照题目要求的分布，生成样本数据集
    :param n: 生成数据的条数
    :return: 数据集[DataFrame]
    """
    np.random.seed(42)

    X = np.random.normal(loc=0, scale=1, size=n)
    epsilon = np.random.normal(loc=0, scale=0.5, size=n)
    Y = X + np.sqrt(abs(X)) * epsilon
    W = np.random.choice([1, -1], size=n)
    epsilon_wild = epsilon * W
    Y_wild = X + np.sqrt(abs(X)) * epsilon_wild

    data = pd.DataFrame({'X': X
                        , 'epsilon': epsilon
                        , 'Y': Y
                        , 'W': W
                        , 'Y_wild': Y_wild})
    # print(data.head(8))

    return data

# generate_data(300)