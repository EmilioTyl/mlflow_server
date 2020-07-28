Adapted Hyperparameter Tuning Example with Docker env
------------------------------

Example of how to do hyperparameter tuning with MLflow and GPyOpt.

Some issues encounter during this adaption.
Main idea is to have MLProject file with entry points for optimization and training tasks.

Training should register loss values in the MLFlow tracking server. GSoOpt create a main run and nested runs fr each itaration.
In order tu use docker env, decopuple the trainning python function from accesing through MlProject. In the original implementation if 
only docker env is add (instead of conda env) an error will arise since internally it uses MLFlow project to start training during GPyOPt.
Hence it is necssary to have a separate training python function that executes the training loop.




examples/hyperparam/MLproject has 2 targets:
  * train
    train simple deep learning model on the wine-quality dataset from our tutorial.
    It has 2 tunable hyperparameters: ``learning-rate`` and ``momentum``.
    Contains examples of how Keras callbacks can be used for MLflow integration.
 
  * gpyopt
    use `GPyOpt <https://github.com/SheffieldML/GPyOpt>`_ to optimize hyperparameters of train.
    GPyOpt can run multiple mlflow runs in parallel if run with ``batch-size`` > 1 and ``max_p`` > 1.


Running this Example
^^^^^^^^^^^^^^^^^^^^

[OPTIONAL] build docker image

.. code-block:: bash
  docker build -t mlflow-wine-env -f Dockerfile . 

Run MLFLOW project
.. code-block:: bash
  mlflow run -e train .

.. code-block:: bash
  mlflow run -e gpyopt .

Visualize with 

.. code-block:: bash
  mlflow ui