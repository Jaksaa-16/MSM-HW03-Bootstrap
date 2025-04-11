import numpy as np
import matplotlib.pyplot as plt

def generate_data(n):
    """
    创造适用于分位数重抽样的样本数据集
    :param n: 样本容量[int]
    :return: 样本数据集[array]
    """
    np.random.seed(42)
    data = np.random.uniform(0, 1, n)
    # print(data)
    return data

def bootstrap(b_times, boot_size, n, data, q_level):
    """
    bootstrap方法实现
    :param b_times: 重抽样次数[int]
    :param boot_size: 每次抽样的数量[int]
    :param n: 样本容量[int]
    :param data: 样本数据集[array]
    :param q_level: 分位数水平(0,100)[float64]
    :return: bootstrap重抽样估计值[array]
    """
    bootstrap_quantiles = []

    for _ in range(b_times):
        indices = np.random.randint(0, n, boot_size)
        sample = data[indices]
        quantile_value = np.percentile(sample, q_level)
        bootstrap_quantiles.append(quantile_value)

    return bootstrap_quantiles

def draw_histogram(array, color):
    """
    绘制估计值分布的直方图
    :param array: 估计值[array]
    :param color: 直方图填充颜色[string]
    :return: 将绘制的图片以.png格式保存在本地路径
    """
    plt.hist(array, bins=30, alpha=0.7, color=color)
    plt.title('Bootstrap Distribution of the 75% Sample Quantile')
    plt.xlabel('75% Sample Quantile')
    plt.ylabel('Frequency')
    plt.savefig(f'3-j.png', dpi=300, bbox_inches='tight')

def draw_histogram_try_b(ax, array, color, title):
    """
    遍历绘组图
    :param ax: 画布绘图点
    :param array: 估计值[array]
    :param color: 直方图填充颜色[string]
    :param title: 图片标题[string]
    :return: 将绘制的图片以.png格式保存在本地路径
    """
    ax.hist(array, bins=30, alpha=0.7, color=color)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('75% Sample Quantile', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

def try_b(try_times, result, times, var, color):
    """
    用于遍历B，观察估计值分布情况改变
    :param try_times: 尝试次数（须是完全平方数）[int]
    :param result: 估计值[list]
    :param times: 估计次数[list]
    :param var: 每次的估计总方差[list]
    :param color: 直方图填充颜色[string]
    :return: 将绘制的图片以.png格式保存在本地路径
    """
    fig, axes = plt.subplots(nrows=int(np.sqrt(try_times))
                           , ncols=int(np.sqrt(try_times))
                           , figsize=(12, 10))

    colors = [color] * len(result)
    for i, ax in enumerate(axes.flat):  # 遍历所有子图（axes.flat 展平二维数组）
        draw_histogram_try_b(ax, result[i], colors[i], f'B={times[i]},var={var[i]}')

    plt.tight_layout()
    plt.savefig(f'3-i-2.png', dpi=300, bbox_inches=None)

n = 1000
data_3 = generate_data(n)

# Q(h)
B = n
q = 75
quantile = bootstrap(B, n, n, data_3, q)
draw_histogram(quantile, 'salmon')

# Q(i)
B_try_times = 25
q = 75
boot = 300
result_list = []
B_list = []
var_list = []
for b in range(B_try_times):
    B = 500 + b * 20
    B_list.append(B)
    res = bootstrap(B, boot, n, data_3, q)
    result_list.append(res)
    var = round(np.sum(np.abs(np.array(res) - q/100) ** 2)/B, 6)
    # var = round(abs(np.mean(res - q/100, axis=0), 4)
    var_list.append(var)

try_b(B_try_times, result_list, B_list, var_list, 'salmon')

# Q(j)
B = n
q = 99.5
quantile = bootstrap(B, n, n, data_3, q)
draw_histogram(quantile, 'salmon')