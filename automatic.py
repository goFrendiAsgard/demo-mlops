from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import time
import mlflow
import mlflow.sklearn
import logging

# For this exammple:
# - we assume there is a scheduler that run for a certain interval
# - We assume the data is slowly increased everytime

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

INTERVAL = 5
INIT_DATA_COUNT = 1000
MAX_DATA_COUNT = 10000
STEP = 100


def run(previous_run_id: Optional[str], df: pd.DataFrame) -> str:
    logging.info(f'🔎 Data count : {df.shape[0]}')

    # Data preprocessing
    df['Numeric Type'] = df['Type'].replace({'H': 2, 'M': 1, 'L': 0})

    # Split data
    X = df[[
        'Numeric Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]]
    y = df['Failure Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if previous_run_id is not None:
        logging.info(f'🧪 Start evaluating  : {previous_run_id}')

        # Evaluate previous model
        old_clf = mlflow.sklearn.load_model(
            f'runs:/{previous_run_id}/RandomForestModel'
        )
        y_pred = old_clf.predict(X_test)
        old_clf_accuracy = accuracy_score(y_test, y_pred)
        logging.info(
            f'🧪 Finish evaluating : {previous_run_id}, accuracy: {old_clf_accuracy}'  # noqa
        )
        if old_clf_accuracy > 0.98:
            return previous_run_id

    # Retrain
    mlflow.start_run(run_name='training')

    # get active run
    active_run = mlflow.active_run()
    active_run_id = active_run.info.run_id
    logging.info(f'💪 Start training  : {active_run_id}')

    # Train
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Log Model
    mlflow.sklearn.log_model(clf, "RandomForestModel")

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)

    logging.info(f'💪 Finish training : {active_run_id}, accuracy: {accuracy}')

    with open('./last-run-id.txt', 'w') as f:
        f.write(active_run_id)

    mlflow.end_run()
    return active_run_id


previous_run_id: Optional[str] = None
for MAX_LIMIT in range(INIT_DATA_COUNT, MAX_DATA_COUNT, STEP):
    # Load data
    df = pd.read_csv('./data/predictive_maintenance.csv')
    df = df[0:MAX_LIMIT]

    # Evaluate model
    previous_run_id = run(previous_run_id, df)
    time.sleep(INTERVAL)
