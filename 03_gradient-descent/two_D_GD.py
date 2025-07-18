import numpy as np


class gradient_Descent_Regressor:
    def __init__(self,no_of_epoches , learning_rate):
        self.m = 100
        self.c = -120
        self.error = 0
        self.score = None
        self.n_e = no_of_epoches
        self.l_r = learning_rate

    def fit(self, x_train, y_train):
        for i in range(self.n_e):
            loss_slope_c = -2 * np.sum(
                y_train - self.m*x_train.ravel() - self.c
            )
            loss_slope_m = -2 * np.sum(
                (y_train - self.m*x_train.ravel() - self.c)
                *
                x_train.ravel()
            )
            self.c = self.c - (
                self.l_r * loss_slope_c
            )
            self.m = self.m - (
                self.l_r * loss_slope_m
            )
            print(self.m,self.c)

    def predict(self, X_test):
        return self.m * X_test + self.c
