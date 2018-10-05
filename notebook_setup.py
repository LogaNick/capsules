# -*- coding: utf-8 -*-
"""
Testing functions for the Notebook locally
"""

import numpy as np

def squash(s):
    """
    A non-linearity that squashes a vector between 0 and 1
    From Equation (1) in Sabour et. al 2017:
        
        v = ((||s||**2) / (1 + ||s||**2)) (s/||s||)
    
    Input:
        s:      column vector
    Output:
        v:      input s squashed between 0 and 1
    
    """
    norm = np.linalg.norm(s)
    norm_squared = norm ** 2
    
    v = (norm_squared / (1.0 + norm_squared)) * (s / norm)
    
    return v

def softmax(b):
    """
    Computes the softmax (as per Equation (3) in Sabour et al. 2017)
    
    c_i = exp(b_i) / sum_j(exp(b_j))
    
    See the following for numerical stability considerations: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    
    Input:
        b:      vector (if matrix, will treat each column as a vector)
    Output:
        c:      vector of softmaxes for each element in b
    """
    c = np.exp(b) / np.sum(np.exp(b), axis=0)
    
    return c

def dynamic_routing(U_j_given_i, iterations, input_capsule_layer, output_capsule_layer):
    """
    IN PROGRESS
    """
    num_capsules_layer_i = len(input_capsule_layer)
    num_capsules_layer_j = len(output_capsule_layer)
    # for all capsule i in layer l and capsule j in layer (l + 1_: b_ij <- 0)
    B = np.matrix(np.zeros((num_capsules_layer_j, num_capsules_layer_i)))
    V = np.zeros((num_capsules_layer_j, num_capsules_layer_i))
    
    
    # for r iterations do
    for iteration in range(iterations):
        # for capsule i in layer l: c_i <- softmax(b_i)
        print("Iteration: {}\n============".format(iteration))
        C = softmax(B)
        print("C: {}".format(C))
        # for all capsule j in layer (l + 1): s_j <- sum_i(c_ij * u_j_given_i)
        S = np.zeros((num_capsules_layer_j, num_capsules_layer_i)) # pre-allocating
        V = np.zeros((num_capsules_layer_j, num_capsules_layer_i))
        for j in range(num_capsules_layer_j):
            S[j, :] = np.multiply(C[j, :], U_j_given_i[j, :])
            print("S{}: {}".format(j, S[j, :]))
            # for all capsule j in layer (l + 1): v_j <- squash(s_j)
            V[j, :] = squash(S[j, :]) #S[j, :]# 
            print("V{}: {}".format(j, V[j, :]))
        
        # for all capsule i in layer l and capsule j in layer (l + 1): b_ij <- b_ij + u_j_given_i dot v_j
        for i in range(num_capsules_layer_i):
            for j in range(num_capsules_layer_j):
                B[j, i] = B[j, i] + np.dot(U_j_given_i[j, :], V[j, :])
                print("B[{}, {}]: {}".format(j, i, B[j, i]))
    
    return V
    

u1 = np.matrix([1, 1]).T
u2 = np.matrix([2, 2]).T

squash(u1)

U_j_give_i = np.concatenate([u1, u2], axis=1).T
V = dynamic_routing(U_j_give_i, 3, U_j_give_i, U_j_give_i)