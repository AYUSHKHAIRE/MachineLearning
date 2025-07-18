import numpy as np


class my_multiple_linear_regression:
    def __init__(self):
        self.coef_ = None
        self.intersept_ = None
        self.error = 0
        self.score = None

    def fit(self, X_train, y_train):
        # insert 1 in front to handle beta0
        X_train = np.insert(X_train,0,1,axis=1)
        # calculate coefficients
        # B = ( X^T . X )^( -1 ) . ( X^T ).( Y )
        x_train_trans = X_train.T
        dot_prod_1 = np.dot(x_train_trans,X_train)
        dot_prod_1_inv = np.linalg.inv(dot_prod_1)
        # get coefs of eqn
        betas = dot_prod_1_inv.dot(x_train_trans).dot(y_train)
        self.intersept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        y_pred = np.dot(
            X_test,self.coef_
        ) + self.intersept_
        return y_pred