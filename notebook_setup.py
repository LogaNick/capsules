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
    
# GOAL: Pass data from input, through the capsules without changing it, then
#       use a fully connected layer to subtract the second capsule (input)
#       from the first
#
# Network:
#       Input: [x1, y1, x2, y2]
#       Output: [x2-x1, y2-1]
#


u1 = np.array([1, 2])
u2 = np.array([4, 3])

# Weight matrices W_i_j
W_1_1 = np.eye(2)
W_1_2 = np.zeros((2, 2))

W_2_1 = np.zeros((2, 2))
W_2_2 = np.eye(2)#np.zeros((2, 2)) #np.identity(2)

# u_j_given_i = W_ij * u_i
u_1_given_1 = np.matmul(W_1_1, u1)
u_1_given_2 = np.matmul(W_2_1, u2)

u_2_given_1 = np.matmul(W_1_2, u1)
u_2_given_2 = np.matmul(W_2_2, u2)

# Dynamic routing unrolled...
# b_ij = 0
b_1_1 = 0
b_1_2 = 0
b_2_1 = 0
b_2_2 = 0


# r iterations
r = 3
for _ in range(r):
    print("iteration {}".format(_))
    print("b_1: [{}, {}]".format(b_1_1, b_1_2))
    print("b_2: [{}, {}]".format(b_2_1, b_2_2))
    # c_i = softmax(b_i)
    # b_1
    denominator = np.exp(b_1_1) + np.exp(b_1_2)    
    c_1_1 = np.exp(b_1_1) / denominator
    c_1_2 = np.exp(b_1_2) / denominator
    print("c_1: [{}, {}]".format(c_1_1, c_1_2))
    
    # b_2
    denominator = np.exp(b_2_1) + np.exp(b_2_2)
    c_2_1 = np.exp(b_2_1) / denominator
    c_2_2 = np.exp(b_2_2) / denominator
    print("c_2: [{}, {}]".format(c_2_1, c_2_2))
    
    # This is where it gets confusing. u_1_given_2 is a vector/matrix (2, 1)
    # for all capsule j in layer (l + 1): s_j <- sum_i(c_ij * u_j_given_i)
    s_1 = c_1_1 * u_1_given_1 + c_2_1 * u_1_given_2 
    s_2 = c_1_2 * u_2_given_1 + c_2_2 * u_2_given_2
    print("s_1: {}".format(s_1.T))
    print("s_2: {}".format(s_2.T))
    
    
    # for all capsule j in layer (l + 1): v_j <- squash(s_j)
    v_1 = s_1 #squash(s_1)
    v_2 = s_2 #squash(s_2)
    print("v_1: {}".format(v_1.T))
    print("v_2: {}".format(v_2.T))
    
    # for all capsule i in layer l and capsule j in layer (l + 1): b_ij <- b_ij + u_j_given_i dot v_j
    b_1_1 = b_1_1 + np.asscalar(np.dot(u_1_given_1, v_1))
    b_1_2 = b_1_2 + np.asscalar(np.dot(u_2_given_1, v_2))
    b_2_1 = b_2_1 + np.asscalar(np.dot(u_1_given_2, v_1))
    b_2_2 = b_2_2 + np.asscalar(np.dot(u_2_given_2, v_2))
    



# Output linear layer with weights 
# TODO: Fully connected layer
weights_output_1 = np.array([-1, -1]) # x
weights_output_2 = np.array([1, 1]) # y
biases = np.array([0, 0])
output1 = weights_output_1[0] * v_1[0] + weights_output_2[0] * v_2[0] + biases[0]
output2 = weights_output_1[1] * v_1[1] + weights_output_2[1] * v_2[1] + biases[1]

print("Output (x, y): ({},{})".format(output1, output2))

"""
U_j_give_i = np.concatenate([u1, u2], axis=1).T
V = dynamic_routing(U_j_give_i, 3, U_j_give_i, U_j_give_i)
"""