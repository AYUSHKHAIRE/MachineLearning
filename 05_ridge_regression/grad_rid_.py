import numpy as np

class MyRidgeGD:

    def __init__(
            self, 
            epochs, 
            learning_rate, 
            alpha
        ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(
            self, 
            X_train, 
            y_train
        ):

        self.coef_ = np.ones(
            X_train.shape[1]
        )
        self.intercept_ = 0
        thetha = np.insert(
            self.coef_, 0, self.intercept_
        )

        X_train = np.insert(
            X_train, 0, 1, axis=1
        )

        for i in range(self.epochs):
            thetha_der = np.dot(
                X_train.T, X_train
                ).dot(
                thetha
                ) - np.dot(
                    X_train.T, y_train
                    ) + self.alpha*thetha
            thetha = thetha - self.learning_rate*thetha_der

        self.coef_ = thetha[1:]
        self.intercept_ = thetha[0]

    def predict(self, X_test):

        return np.dot(X_test, self.coef_) + self.intercept_