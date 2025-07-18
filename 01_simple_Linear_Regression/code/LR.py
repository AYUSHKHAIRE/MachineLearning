import numpy as np

class my_linear_regression:
    def __init__(self):
        self.m = None
        self.b = None
        self.error = 0
        self.score = None

    def fit(self, X_train, y_train):
        numerator = 0
        denominator = 0

        for i in range(X_train.shape[0]):
            numerator += (X_train[i] - X_train.mean()) * \
                (y_train[i] - y_train.mean())
            denominator += (X_train[i] - X_train.mean())**2

        self.m = numerator / denominator
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b

    def c_mean_absolute_error(self, y_test, y_pred):
        total_error = 0
        n = len(y_test)
        for i in range(n):
            error = abs(y_test[i] - y_pred[i])
            total_error += error
        abs_error = total_error / n
        self.error  = abs_error
        return self.error

    def c_mean_squared_error(self, y_test, y_pred):
        total_error = 0
        n = len(y_test)
        for i in range(n):
            error = y_test[i] - y_pred[i]
            total_error += error**2
        sq_error = total_error / n
        self.error = sq_error
        return self.error

    def c_root_mean_squared_error(self, y_test, y_pred):
        total_error = 0
        n = len(y_test)
        for i in range(n):
            error = y_test[i] - y_pred[i]
            total_error += error**2
        mean_sq_error = total_error / n
        rmse = np.sqrt(mean_sq_error)
        self.error = rmse
        return self.error

    def c_root_mean_squared_error(self, y_test, y_pred):
        total_error = 0
        n = len(y_test)
        for i in range(n):
            error = y_test[i] - y_pred[i]
            total_error += error**2
        mean_sq_error = total_error / n
        rmse = np.sqrt(mean_sq_error)
        self.error = rmse
        return self.error

    def c_r2_score(self, y_test, y_pred):
        n = len(y_test)
        mean_y_test = np.mean(y_test)
        sq_error_for_mean_line = 0
        sq_error_for_regression_line = 0
        for i in range(n):
            val_reg = y_test[i] - y_pred[i]
            val_reg = val_reg ** 2
            sq_error_for_regression_line += val_reg
            val_mean = y_test[i] - mean_y_test
            val_mean = val_mean ** 2
            sq_error_for_mean_line += val_mean
        tmp = sq_error_for_regression_line / sq_error_for_mean_line
        val_to_return = 1 - tmp
        self.score = val_to_return
        return self.score


    def c_adjusted_r2_score(self, y_test, y_pred):
        r2_score = self.c_r2_score(y_test,y_pred)
        n = len(y_test)
        k = 1
        adj_r2_score = 1 - (
            (
                (
                    1-r2_score
                )
                *
                (
                    n - 1
                )
                /
                (
                    n-1-k
                )
            )
        )
        self.score = adj_r2_score
        return self.score
