from sklearn.tree import DecisionTreeRegressor
import numpy as np


class gradBoostRegressor:
    def __init__(self, loss_fun='OLS', m=10, learning_rate=0.1):
        self.loss_fun = loss_fun
        self.m = m
        self.learning_rate = learning_rate
        self.f_0 = None
        self.models = []

    def fit(self, X, y):
        if self.loss_fun == 'OLS':
            self.f_0 = y.mean()
            residuals = y - self.f_0
            for i in range(self.m):
                Dtr = DecisionTreeRegressor()
                Dtr.fit(X, residuals)
                prediction = Dtr.predict(X)
                residuals -= self.learning_rate * prediction
                self.models.append(Dtr)
        elif self.loss_fun == 'MAE':
            self.f_0 = np.median(y)
            residuals = y - self.f_0
            for i in range(self.m):
                Dtr = DecisionTreeRegressor()
                Dtr.fit(X, residuals)
                prediction = Dtr.predict(X)
                residuals -= self.learning_rate * \
                    np.sign(residuals - prediction)
                self.models.append(Dtr)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fun}")

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.f_0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred


class gradBoostClassifier:
    def __init__(self, loss_fun='log-odds', m=10, learning_rate=0.1):
        self.loss_fun = loss_fun
        self.m = m
        self.learning_rate = learning_rate
        self.f_0 = None
        self.models = []

    def fit(self, X, y):
        if self.loss_fun == 'log-odds':
            self.f_0 = y.mean()
            residuals = y - self.f_0
            for i in range(self.m):
                Dtr = DecisionTreeRegressor()
                Dtr.fit(X, residuals)
                prediction = Dtr.predict(X)
                residuals -= self.learning_rate * prediction
                self.models.append(Dtr)
        elif self.loss_fun == 'MAE':
            self.f_0 = np.median(y)
            residuals = y - self.f_0
            for i in range(self.m):
                Dtr = DecisionTreeRegressor()
                Dtr.fit(X, residuals)
                prediction = Dtr.predict(X)
                residuals -= self.learning_rate * \
                    np.sign(residuals - prediction)
                self.models.append(Dtr)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fun}")

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.f_0)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
