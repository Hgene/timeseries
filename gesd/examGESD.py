import esd
import numpy as np

#data generate from normal distribution
x = [np.random.normal(0,1) for i in range(0,30)]

K = 8
alpha = 0.1

result = esd.gesd(x, K, alpha)
print(result)