import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import generated_data as gd

def fit_ols(x, y):
    """
    对X, Y训练数据进行OLS回归，得到参数β的估计量
    :param x: 特征变量[Array]
    :param y: 目标变量[Array]
    :return:
        斜率[array]
        截距[Float64]
    """
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    beta_ols_intercept = model.intercept_
    beta_ols_coef = model.coef_

    return beta_ols_intercept, beta_ols_coef


def draw_histogram(data, content, method, color):
    """
    对结果绘制直方图，以便观察分布情况
    :param data: 画图的数据[array]
    :param content: 是对哪个数据画的图[string]
    :param method: 是用那种方法得到的结果[string]
    :param color: 直方图的填充颜色[string]
    :return: 将绘制的图片以.png格式保存在本地路径
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=10, color=color, edgecolor='black', alpha=0.7)

    plt.title(f'Histogram of {content} using {method}', fontsize=16)
    plt.xlabel(f'{content}', fontsize=14)
    plt.ylabel(f'Frequency', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{method}-{content}.png', dpi=300, bbox_inches='tight')


def naive_bootstrap(boot_size, b_times, x, y):
    """
    使用naive_bootstrap，对样本进行OLS回归，计算参数的方差
    :param boot_size: 每次bootstrap抽取的样本数（≤样本总数）[int]
    :param b_times: bootstrap的次数[int]
    :param x: 特征变量[Array]
    :param y: 目标变量[Array]
    :return:
        截距的方差[Float64]
        斜率的方差[Float64]
    """
    bootstrap_betas_intercept = []
    bootstrap_betas_coef = []

    for _ in range(b_times):
        indices = np.random.randint(0, y.shape[0], boot_size)
        X_boot = x[indices]
        Y_boot = y[indices]

        intercept_boot, coef_boot = fit_ols(X_boot, Y_boot)
        bootstrap_betas_intercept.append(intercept_boot)
        bootstrap_betas_coef.append(coef_boot)

    bootstrap_betas_intercept = np.array(bootstrap_betas_intercept)
    bootstrap_betas_coef = np.array(bootstrap_betas_coef)
    var_intercept = np.var(bootstrap_betas_intercept, ddof=1)
    var_slope = np.var(bootstrap_betas_coef, ddof=1)
    print(f"var_intercept: {var_intercept}")
    print(f"var_slope: {var_slope}")

    return var_intercept, var_slope, bootstrap_betas_intercept, bootstrap_betas_coef

# data_300 = gd.generate_data(300)
# xi = np.array(data_300['X'])
# yi = np.array(data_300['Y'])
#
# var_i, var_s, boot_i, boot_c = naive_bootstrap(300, 1000, xi, yi)
#
# draw_histogram(boot_i, 'estimated_intercept', 'naive bootstrap', 'salmon')
# draw_histogram(boot_c, 'estimated_coef', 'naive bootstrap', 'salmon')



