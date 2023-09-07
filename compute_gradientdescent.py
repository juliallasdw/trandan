import math,copy

import numpy as np
import matplotlib as plt
x_train=np.array([1,2])
y_train=np.array([300,500])
#cost function
def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
    total_cost=1/(2*m)*cost
    return total_cost
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb =w*x[i]+b
        dj_dw=dj_dw+(f_wb-y[i])*x[i]
        dj_db=dj_db+(f_wb-y[i])
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db
def gradient_descent(x,y,w_in,b,alpha,num_iters,cost_function,gradient_function):
    J_history=[]
    p_history=[]
    b=0
    w=0
    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(x,y,w,b)
        b=b-alpha*dj_db
        w=w-alpha*dj_dw
        if i<100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        if i% math.ceil(num_iters/10)==0:
            print(f"Interation{i:4}:Cost {J_history[-1]:0.2e}",
              f"dj_dw:{dj_dw:0.3e},dj_db:{dj_db:0.3e}",
              f"w:{w:0.3e},b:{b:0.5e}")
    return w,b,J_history,p_history
w_init=0
b_init=0
iterations=10000
alpha=1.0e-2
w_final,b_final,J_hist,p_hist=gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations,compute_cost,compute_gradient)
print(f"(w,b)) found by gradient descent:({w_final:8.4f},{b_final:8.4f})" )
