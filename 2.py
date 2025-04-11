import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split

def generate_data(n_samples, n_features):
    """
    生成一个适用于lasso回归的数据集
    :param n_samples: 样本容量[int]
    :param n_features: 特征变量数量[int]
    :return: 数据集[dataframe, array]
    """
    np.random.seed(42)

    X_array = np.random.randn(n_samples, n_features)
    column_name = [f'X{i + 1}' for i in range(n_features)]
    X_df = pd.DataFrame(X_array, columns=column_name)
    True_Theta = np.zeros(n_features)
    True_Theta[:5] = np.array([1, 2, 3, 4, 5])
    Epsilon = np.random.normal(loc=0, scale=0.5, size=n_samples)
    Y_array = X_array.dot(True_Theta) + Epsilon
    Y_df = pd.DataFrame(Y_array, columns=['Y'])
    Data = pd.concat([X_df, Y_df], axis=1)
    # print(Data.head(5))

    return Data, X_array, X_df, True_Theta, Epsilon, Y_array, Y_df

def fit_lasso(x_train, y_train):
    """
    构建lasso模型拟合函数，首先使用CV获取最佳超参数，再将最优参数输入拟合模型
    :param x_train: 训练集-特征[array]
    :param y_train: 训练集-目标[array]（压平到一维）
    :return:
        theta_hat: 估计量θ[array]
        optimal_alpha: lasso模型的最优参数α[array]
    """
    lasso_cv = LassoCV(cv=5, random_state=42).fit(x_train, y_train)
    optimal_alpha = lasso_cv.alpha_
    # print(f"optimal_alpha: {optimal_alpha}")

    lasso_optimal = Lasso(alpha=optimal_alpha, fit_intercept=True).fit(x_train, y_train)
    theta_hat = lasso_optimal.coef_
    # print(f"theta_hat: {theta_hat}")

    return theta_hat, optimal_alpha

def bootstrap(boot_size, b_times, x_train, y_train, theta_hat_original):
    """
    bootstrap估计参数
    :param boot_size: 每次抽样的数量[int](≤ n_samples)
    :param b_times: 重抽样的次数[int]
    :param x_train: 训练集-特征[array]
    :param y_train: 训练集-目标[array]（压平到一维）
    :param theta_hat_original: 先用样本集全体拟合一次lasso，获取初始估计量[array]
    :return:
        bias: bootstrap得到的θ估计值的均值相较于初始估计量的偏误[array]
        bootstrap_thetas: bootstrap得到的θ估计值[array]
    """
    bootstrap_thetas = []

    for _ in range(b_times):
        indices = np.random.randint(0, y_train.shape[0], boot_size)
        x_boot = x_train[indices]
        y_boot = y_train[indices]

        theta_hat_boot, alpha_selected = fit_lasso(x_boot, y_boot)
        bootstrap_thetas.append(theta_hat_boot)

    bootstrap_thetas = np.array(bootstrap_thetas)
    bias = np.mean(bootstrap_thetas, axis=0) - theta_hat_original

    # print(f"bias: {bias}")

    return bias, bootstrap_thetas


def ci_percentile(bootstrap_thetas, alpha_1):
    """
    用percentile方法求估计量θ在一定水平下的置信区间
    :param bootstrap_thetas: bootstrap得到的θ估计值[array]
    :param alpha_1: 置信水平[float64]
    :return: 各特征变量的参数θ的置信区间[dataframe]
    """
    CI = pd.DataFrame(columns=['feature', 'lower_bound', 'upper_bound'])

    feature_index = [f'X{i + 1}' for i in range(bootstrap_thetas.shape[1])]
    lower_bound = np.percentile(bootstrap_thetas, alpha_1 / 2 * 100, axis=0)
    upper_bound = np.percentile(bootstrap_thetas, (1 - alpha_1 / 2) * 100, axis=0)

    CI['feature'] = feature_index
    CI['lower_bound'] = lower_bound
    CI['upper_bound'] = upper_bound
    # print(CI)
    return CI


def ci_percentile_t(bootstrap_thetas, theta_hat_original, alpha_2):
    """
    用percentile-t方法求估计量θ在一定水平下的置信区间
    :param bootstrap_thetas: bootstrap得到的θ估计值[array]
    :param theta_hat_original: 初始估计量[array]
    :param alpha_2: 置信水平[float64]
    :return: 各特征变量的参数θ的置信区间[dataframe]
    """
    CI = pd.DataFrame(columns=['feature', 'lower_bound', 'upper_bound'])
    feature_index = [f'X{i + 1}' for i in range(bootstrap_thetas.shape[1])]

    t_stats = (bootstrap_thetas - theta_hat_original) / np.std(bootstrap_thetas, axis=0, ddof=1)
    lower_t = np.percentile(t_stats, alpha_2 / 2 * 100, axis=0)
    upper_t = np.percentile(t_stats, (1 - alpha_2 / 2) * 100, axis=0)

    ci_lower_t = theta_hat_original + lower_t * np.std(bootstrap_thetas, axis=0, ddof=1)
    ci_upper_t = theta_hat_original + upper_t * np.std(bootstrap_thetas, axis=0, ddof=1)

    CI['feature'] = feature_index
    CI['lower_bound'] = ci_lower_t
    CI['upper_bound'] = ci_upper_t
    # print(CI)
    return CI





data_300_20, x_array, x_df, theta, epsilon, y_array, y_df = generate_data(300, 20)
xi = np.array(x_df)
yi = np.array(y_df)
yi = np.squeeze(yi)
x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(xi, yi, test_size=0.2, random_state=42)

theta_hat_original_l, alpha = fit_lasso(x_train_l, y_train_l)
# print(f"theta_hat_original_l: {theta_hat_original_l}")
bias, thetas = bootstrap(100, 1000, x_train_l, y_train_l, theta_hat_original_l)

# Q(e) & Q(f)
alpha = 0.05
print('percentile')
ci_percentile(thetas, alpha)
print('percentile-t')
ci_percentile_t(thetas, theta_hat_original_l, alpha)

