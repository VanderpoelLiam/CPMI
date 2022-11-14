import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm

lamb = np.array([5.0000E-01 , 2.5000E-01 , 2.0318E-01 , 1.2500E-01 , 1.5850E-01 , 1.3116E-01 , 1.2137E-01 , 1.1525E-01 , 8.7584E-02 , 6.2500E-02 , 4.3715E-02 , 4.0373E-02 , 3.5432E-02 , 3.1250E-02 , 1.5625E-02 , 7.8125E-03 , 3.9062E-03 , 1.9531E-03 , 9.7656E-04, 0])

lm_9 = np.array([5.913, 23.05, 23.89, 24.86, 24.46, 24.76, 24.89, 24.87, 24.96, 25.12, 25.19, 25.20, 25.22, 25.22, 25.2, 25.23, 25.22, 25.21, 25.21, 26.04])
lm_full_9 = np.array([11.63, 23.37, 24.25, 24.96, 24.66, 24.89, 24.98, 24.97, 25.01, 25.12, 25.20, 25.19, 25.19, 25.19, 25.23, 25.23, 25.19, 25.21, 25.2, 26.04])

lm_11 = np.array([5.024, 23.04, 23.96, 24.46, 24.80, 24.83, 24.84, 24.92, 25.06, 25.16, 25.23, 25.23, 25.24, 25.28, 25.29, 25.29, 25.28, 25.26, 25.27, 26.16])
lm_full_11 = np.array([12.53, 23.79, 24.44, 24.87, 25.04, 25.06, 25.07, 25.11, 25.18, 25.28, 25.26, 25.25, 25.26, 25.3, 25.34, 25.29, 25.29, 25.27, 25.27, 26.16])

# filename = "lambda_search_9"
filename = "lambda_search_11"
lm = lm_11
lm_full = lm_full_11

# -------------------------------------

plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel(r'$\mathregular{\lambda}$', labelpad=10)
ax.set_ylabel(r'ROUGE $\mathregular{F_L}$ score', labelpad=10)

ax.plot(lamb, lm, marker='o', linewidth=2, label='LM')
ax.plot(lamb, lm_full, marker='o', linewidth=2, label='LM_FULL')
ax.legend(frameon=False, fontsize=16)
plt.subplots_adjust(bottom=0.15)

path = "/home/liam/Dropbox/ETH/Courses/Research/Thesis/figs/"
plt.savefig(path + filename, dpi = 300)
# plt.show()
