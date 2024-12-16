import numpy as np
from sklearn.mixture import GaussianMixture

def fit_gmm(X):
    '''
    Input:
        X: np.array, shape (n_samples, n_features)
    Return:
        mean_left: float
        covar_left: float
        mean_right: float
        covar_right: float
    '''
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=1e-3)
    gmm.fit(X)

    left_idx = gmm.means_.reshape(-1).argmin()
    mean_left = gmm.means_.reshape(-1)[left_idx]
    covar_left = gmm.covariances_.reshape(-1)[left_idx]
    
    right_idx = gmm.means_.reshape(-1).argmax()
    mean_right = gmm.means_.reshape(-1)[right_idx]
    covar_right = gmm.covariances_.reshape(-1)[right_idx]
    
    return mean_left, covar_left, mean_right, covar_right

def binary_search(x_min, x_max, func, obj_v,
                  tol=1e-3, max_iter=10, is_func_decrease=False):
    def cost(x):
        if is_func_decrease:
            return obj_v - func(x)
        return func(x) - obj_v

    iter_count = 0

    cost_x_max = cost(x_max)
    cost_x_min = cost(x_min)

    if cost_x_max < 0:
        return x_max
    
    if cost_x_min > 0:
        return x_min
    
    while iter_count < max_iter:
        x_mid = (x_min + x_max) / 2.
        cost_x_mid = cost(x_mid)

        if np.abs(cost_x_mid) < tol:
            return x_mid
        
        if cost_x_mid > 0:
            x_max = x_mid
        else:
            x_min = x_mid
        
        iter_count += 1
    
    return x_max