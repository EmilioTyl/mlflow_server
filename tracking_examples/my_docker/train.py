import mlflow
from mlflow import log_metric, log_param, log_artifacts

import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import pickle

def eval_metrics(pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    return rmse, mae, r2

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha")
    parser.add_argument("--l1")
    args = parser.parse_args()

    alpha = float(args.alpha)
    l1 = float(args.l1)

    abs_path = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "wine-quality.csv"
    dataset_path = os.path.join(abs_path, dataset_name)
    data = pd.read_csv(dataset_path)

    train, test = train_test_split(data)

    x_train = train.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    x_test = train.drop(["quality"], axis=1)
    y_test = train[["quality"]]

   
    experiment_name = "Quality"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="dockerized_run"):
        model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=88)
        model.fit(x_train, y_train)

        log_param("alpha", alpha)
        log_param("l1", l1)

        #Train metrics
        pred = model.predict(x_train)
        rmse, mae, r2 = eval_metrics(pred, y_train)
        
        log_metric("train_rmse", rmse)
        log_metric("train_mae", mae)
        log_metric("train_r2", r2)

        #Test metrics
        pred = model.predict(x_test)
        rmse, mae, r2 = eval_metrics(pred, y_test)
        
        log_metric("test_rmse", rmse)
        log_metric("test_mae", mae)
        log_metric("test_r2", r2)
        
        pkl_filename = "model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)




        
