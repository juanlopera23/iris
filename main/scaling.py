from sklearn.preprocessing import StandardScaler


def standard_scaler(x_train,x_test):

    scaler=StandardScaler()
    scaler.fit(x_train)
    X_train_scaler= scaler.transform(x_train)
    X_test_scaler= scaler.transform(x_test)

    return X_train_scaler,X_test_scaler