import numpy as np

class my_n_ridge:

    def __init__(
            self, 
            alpha=0.1
        ):

        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self,x_train,y_train):
        x_train = np.insert(
            x_train,
            0,
            1,
            axis=1,
        )
        I = np.identity(
            x_train.shape[1]
        )

        I[0][0] = 0

        # The reason of using I[0][0]=0 is that because in our metrix W the first term is basically intercept not slope and we have to multiply Lambda with only slopes.
        # Thats why first term became zero and lambda is multiplied with only slopes.

        result = np.linalg.inv(
            np.dot(
                x_train.T,
                x_train
            )
            +
            self.alpha * I
        ).dot(
            x_train.T
        ).dot(
            y_train
        )

        self.intercept_ = result[0]
        self.coef_ = result[1:]

    def predict(self , x_test):
        return np.dot(x_test, self.coef_) + self.intercept_