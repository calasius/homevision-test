import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train_model():
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

    categorical_features = ["ElementarySchoolName"]
    numerical_features = X.columns.difference(categorical_features)

    # Build a pipeline for preprocessing
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    _logger.info("Training the model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    _logger.info(f"Model root mean squared error: {np.sqrt(mse)}")

    # Save the pipeline using joblib
    _logger.info("Saving the model...")
    with open("/app/model/model.joblib", "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    train_model()
