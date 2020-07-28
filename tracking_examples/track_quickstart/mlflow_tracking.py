import os
from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import pdb
"""
MLflow allows you to group runs under experiments, 
which can be useful for comparing runs intended to tackle a particular task. 
"""
if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    mlflow.set_tracking_uri("http://localhost:5000/")
    e_name = "example"
    e = mlflow.get_experiment_by_name(e_name)
    if e is None: mlflow.create_experiment(e_name)
    mlflow.set_experiment(e_name)
    with mlflow.start_run(run_name='example'):
        for i in range(0,8):
            log_metric(key="quality", value= i*4, step=i) 

        log_param("param1", randint(0, 100))

        log_metric("foo", random())
        log_metric("foo", random() + 1)
        log_metric("foo", random() + 2)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/test.txt", "w") as f:
            f.write("hello world!"+str(random()))
        log_artifacts("outputs")
