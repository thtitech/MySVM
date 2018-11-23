from constants import *
import numpy as np
import cvxopt as cv

class MySVM:
    def __init__(self, C=1.0, kernel="rbf"):
        self.C = C
        self.kernel = kernel_dct[kernel] if kernel in kernel_dct else kernel

        self.w = None
        self.support_vector = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        gram_matrix = self.kernel(X)

        P = gram_matrix
        P = np.c_[P, np.zeros((len(X), len(X)))]
        P = np.r_[P, np.zeros((len(X), 2*len(X)))]
        P = cv.matrix(P)
        
        q = cv.matrix(np.array([1 if i < len(X) else 0 for i in range(2*len(X))], dtype="double"))
        G = cv.matrix(-np.identity(2*len(X)))
        h = cv.matrix(np.zeros(2*len(X)))
        
        A = np.c_[np.array([y]), np.zeros((1, len(X)))]
        for i in range(len(X)):
            A = np.r_[A, np.array([[1 if j == i or j == (i + len(X)) else 0 for j in range(2*len(X))]])]
        A = cv.matrix(A)
        
        b = np.array([0 if i == 0 else self.C for i in range(len(X)+1)])
        b = cv.matrix(b)

        print("END Make Matrix")
        sol = cv.solvers.qp(P, q, G=G, h=h, A=A, b=b)

        print(sol["x"])
