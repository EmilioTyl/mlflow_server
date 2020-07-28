import numpy as np
import pandas as pd

def get_dataset(uri):
    data = pd.read_csv(uri, sep=';')
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, random_state=seed)
    train, valid = train_test_split(train, random_state=seed)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1).as_matrix()
    train_x = (train_x).astype('float32')
    train_y = train[["quality"]].as_matrix().astype('float32')
    valid_x = (valid.drop(["quality"], axis=1).as_matrix()).astype('float32')

    valid_y = valid[["quality"]].as_matrix().astype('float32')

    test_x = (test.drop(["quality"], axis=1).as_matrix()).astype("float32")
    test_y = test[["quality"]].as_matrix().astype("float32")
    return data, train_x, train_y, valid_x, valid_y, test_x, test_y 
