import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

#x-axis ranges from -5 and 5 with .001 steps
x = np.arange(-2, 2, 0.001)

#define multiple normal distributions
# plt.plot(x, norm.pdf(x, 0, 1), label='μ: 0, σ: 1')
plt.plot(x, norm.pdf(x, 0.5, 0.3), label='Norm μ:0.5, σ: 1')
plt.plot(x, beta.pdf(x, 5, 2), label='beta 8,5')

#add legend to plot
plt.legend()
plt.savefig("fig1.jpg")