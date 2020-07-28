from nn_model import generate_model

import warnings

import math

import tensorflow.keras as keras
import numpy as np
import pandas as pd

import click

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.keras


def eval_and_log_metrics(prefix, actual, pred, epoch):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
    return rmse

class MLflowCheckpoint(Callback):
    """
    Example of Keras MLflow logger.
    Logs training metrics and final model with MLflow.
    We log metrics provided by Keras during training and keep track of the best model (best loss
    on validation dataset). Every improvement of the best model is also evaluated on the test set.
    At the end of the training, log the best model with MLflow.
    """

    def __init__(self, test_x, test_y, loss="rmse"):
        self._test_x = test_x
        self._test_y = test_y
        self.train_loss = "train_{}".format(loss)
        self.val_loss = "val_{}".format(loss)
        self.test_loss = "test_{}".format(loss)
        self._best_train_loss = math.inf
        self._best_val_loss = math.inf
        self._best_model = None
        self._next_step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the best model at the end of the training run.
        """
        if not self._best_model:
            raise Exception("Failed to build any model")
        mlflow.log_metric(self.train_loss, self._best_train_loss, step=self._next_step)
        mlflow.log_metric(self.val_loss, self._best_val_loss, step=self._next_step)
        #mlflow.keras.log_model(self._best_model, "model")

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return
        self._next_step = epoch + 1
        train_loss = logs["loss"]
        val_loss = logs["val_loss"]
        mlflow.log_metrics({
            self.train_loss: train_loss,
            self.val_loss: val_loss
        }, step=epoch)

        if val_loss < self._best_val_loss:
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_train_loss = train_loss
            self._best_val_loss = val_loss
            #self._best_model = keras.models.clone_model(self.model)
            #self._best_model.build(self.model.layers[0].input_shape)
            #self._best_model.set_weights([x.copy() for x in self.model.get_weights()])
            preds = self.model.predict(self._test_x)
            eval_and_log_metrics("test", self._test_y, preds, epoch)

@click.command(help="Trains an Keras model on wine-quality dataset.")
@click.option("--epochs", type=click.INT, default=100, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=16,)
@click.option("--learning-rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--momentum", type=click.FLOAT, default=.9, help="SGD momentum.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.argument("training_data")
def run(training_data, epochs, batch_size, learning_rate, momentum, seed):
    train(training_data, epochs, batch_size, learning_rate, momentum, seed)
    
def train(training_data, epochs, batch_size, learning_rate, momentum, seed):
    warnings.filterwarnings("ignore")
    data = pd.read_csv(training_data, sep=';')
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, random_state=seed)
    train, valid = train_test_split(train, random_state=seed)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1).to_numpy()
    train_x = (train_x).astype('float32')
    train_y = train[["quality"]].to_numpy().astype('float32')
    valid_x = (valid.drop(["quality"], axis=1).to_numpy()).astype('float32')

    valid_y = valid[["quality"]].to_numpy().astype('float32')

    test_x = (test.drop(["quality"], axis=1).to_numpy()).astype("float32")
    test_y = test[["quality"]].to_numpy().astype("float32")


    #with mlflow.start_run():
    if epochs == 0:  # score null model
        eval_and_log_metrics("train", train_y, np.ones(len(train_y)) * np.mean(train_y),
                                epoch=-1)
        eval_and_log_metrics("val", valid_y, np.ones(len(valid_y)) * np.mean(valid_y), epoch=-1)
        eval_and_log_metrics("test", test_y, np.ones(len(test_y)) * np.mean(test_y), epoch=-1)
    else:
        mlflow_callback = MLflowCheckpoint(test_x, test_y)
        model = generate_model(train_x, learning_rate, momentum)
        model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(valid_x, valid_y),
                        callbacks=[mlflow_callback])
            
if __name__=="__main__":
    run()


