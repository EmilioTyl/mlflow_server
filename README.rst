MLFlow server with Docker and MySQL
------------------------------
MLFlow track parameters, models, and experiments.s

Two approaches:
* local server created executed by docker-compose-local.yml . It consist of mlflow server conected to a MySQL DB, running locally.
* deploying version (not tested). Same as local but using a reverse nginx proxy server



Running this Example
^^^^^^^^^^^^^^^^^^^^

[OPTIONAL] build docker image

.. code-block:: bash
  docker-compose up -f docker-compose-local.yml -d --build 

