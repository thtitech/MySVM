from constants import *
import numpy as np
import cvxopt as cv

class MySVM:
    def __init__(self, C=1.0, kernel="rbf"):
        self.C = C
        self.kernel = kernel_dct[kernel] if kernel in kernel_dct else kernel
        self.w = None
        self.support_vector = None
        self.b = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        gram_matrix = self.kernel(X)
        assert len(X) == len(y), "not same shape label and feature"
        N = len(X)

        T = np.array([[y[i] * y[j] for j in range(N)] for i in range(N)])
        P = gram_matrix * T
        P = cv.matrix(P)

        q = cv.matrix(-np.ones(N))
        G = cv.matrix(np.r_[np.identity(N), -np.identity(N)])
        h = cv.matrix(np.r_[self.C*np.ones(N).T, np.zeros(N).T])
        
        A = cv.matrix(np.array([y], dtype="double"))
        b = cv.matrix(0.0)

        sol = cv.solvers.qp(P, q, G=G, h=h, A=A, b=b)
        #print(len(list(filter(lambda x: x > eps, sol["x"]))))
        
        index_list = list(filter(lambda x: sol["x"][x] > eps, range(N)))
        self.w = np.array(sol["x"])[index_list].reshape(len(index_list)) * y[index_list]
        #print(np.array(sol["x"]).shape)
        print(self.w)
        self.support_vector = X[index_list]
        # calc b
        tmp_list = []
        for i in index_list:
            tmp = 0
            for j in index_list:
                tmp += (sol["x"][j] * y[j] * gram_matrix[i][j])
            tmp_list.append(y[i]-tmp)
        self.b = np.mean(tmp_list)
        
    def predict(self, X):
        assert self.w is not None or self.support_vector is not None or self.b is not None, "not call fit method yet"
        #print(self.w.shape)
        #print(self.kernel(X, Y=self.support_vector).shape)
        #np.dot(self.w, self.kernel(X, Y=self.support_vector))
        y = np.dot(np.array([self.w]), self.kernel(X, Y=self.support_vector).T) + self.b
        y = y.reshape(len(X))
        return np.array([1 if pred > 0 else -1 for pred in y])
