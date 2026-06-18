import numpy as np 
import math , copy
def gradient_des(x , y, w_in , b_in , cost_fun , num_iteration , grad_func , alpha):

    j_histroy = []
    w = copy.deepcopy(w_in)  
    b = b_in

    for i in range(num_iteration):
        dj_db ,  dj_dw = grad_func(x , y , w , b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i <100000:
            j_histroy.append(cost_fun(x , y , w ,b))

        if i% math.ceil(num_iteration / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_histroy[-1]:8.2f}   ")
        
    return w, b, j_histroy

def compute_cost(X, y, w, b): 
    
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def compute_gradient(x , y , w , b):
    m , n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0 

    for i in range(m):
        err = (np.dot(x[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
            dj_db = dj_db + err

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw