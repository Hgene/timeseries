import numpy as np
from scipy.stats import t
import copy

def gesd(x_univariate_timeSeries, K, alpha=0.05):
    ''' find Top K anomalous data points using hypothesis test that extract R_i and lambda_i to explain how anomalous data point is
       x_univariate_timeSeries : univariate time series data
       K : number of test, equal to maximum number of detected ouliers by gesd algorithm
       alpha : significant level
    '''

    x = copy.deepcopy(x_univariate_timeSeries)
    result = np.zeros(len(x))
    idxs = [x for x in range(len(x))]

    for k in range(K):
        s_mean = np.mean(x)
        s_std = np.std(x)
        n = len(x)

        i = np.argmax(abs(x - s_mean))

        p = 1 - alpha / (2 * (n - (i + 1) + 1))
        t_p_df = t.ppf(p, df=n - (i + 1) + 1)

        R_i = max(abs(x - s_mean)) / s_std
        lambda_i = ((n - i) * t_p_df) / (np.sqrt(n - (i + 1) - 1 + t_p_df ** 2) * (n - (i + 1) + 1))

        if R_i > lambda_i:
            result[idxs[i]] = 1
            del x[i]
            del idxs[i]
        else:
            return result

    return result