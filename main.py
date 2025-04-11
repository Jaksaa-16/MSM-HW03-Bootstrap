import numpy as np
import pandas as pd

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 定义数据量
n = 300

# 生成X，服从标准正态分布 N(0, 1)
X = np.random.normal(loc=0, scale=1, size=n)

# 生成e，服从均值为0，标准差为0.5的正态分布 N(0, 0.5^2)
e = np.random.normal(loc=0, scale=0.5, size=n)

# 计算Y，公式为 Y = X + e * X
Y = X + e * X

# 将数据组合成一个DataFrame（方便查看和保存）
data = pd.DataFrame({'X': X, 'e': e, 'Y': Y})

# 打印前几行数据
print(data.head())

# 如果需要保存到CSV文件，可以使用以下命令：
# data.to_csv('generated_data.csv', index=False)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
