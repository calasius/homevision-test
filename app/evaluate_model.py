import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def evaluate_model():
    df = pd.read_csv("/app/data/home-listings-example.csv")
    X_columns = [
        "BathsTotal",
        "BedsTotal",
        "CDOM",
        "LotSizeAreaSQFT",
        "SqFtTotal",
        "ElementarySchoolName",
    ]
    X = df[X_columns]
    y_columns = ["ClosePrice"]
    y = df[y_columns]

    # remove rows with nulls in target also in input_df
    y = y[y.notnull().all(axis=1)]
    X = X[X.index.isin(y.index)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open("/app/model/model.joblib", "rb") as f:
        model = joblib.load(f)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    _logger.info(f"RMSE: {rmse}")


if __name__ == "__main__":
    evaluate_model()
