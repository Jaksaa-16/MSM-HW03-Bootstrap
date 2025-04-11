import numpy as np
from sklearn.linear_model import LinearRegression
import generated_data as gd
import a


data_300 = gd.generate_data(300)
xi = np.array(data_300['X'])
yi = np.array(data_300['Y_wild'])

var_i_w, var_s_w, boot_i_w, boot_c_w = a.naive_bootstrap(300, 1000, xi, yi)

a.draw_histogram(boot_i_w, 'estimated_intercept', 'wild bootstrap', 'lightskyblue')
a.draw_histogram(boot_c_w, 'estimated_coef', 'wild bootstrap', 'lightskyblue')

