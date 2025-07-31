import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.ohe = None
        self.label_encoders = {}
        self.scaler = None
        self.categorical_cols = None
        self.low_cardinality_cols = None
        self.high_cardinality_cols = None
        self.numeric_cols = None

    def fit_transform(self, X, y=None):
        X = X.copy()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Split categorical into low/high cardinality
        self.low_cardinality_cols = [col for col in self.categorical_cols if X[col].nunique() <= 10]
        self.high_cardinality_cols = [col for col in self.categorical_cols if X[col].nunique() > 10]

        # One-hot encode low-cardinality
        if self.low_cardinality_cols:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe_arr = self.ohe.fit_transform(X[self.low_cardinality_cols])
            ohe_df = pd.DataFrame(ohe_arr, columns=self.ohe.get_feature_names_out(self.low_cardinality_cols), index=X.index)
            X = X.drop(self.low_cardinality_cols, axis=1)
            X = pd.concat([X, ohe_df], axis=1)

        # Label encode high-cardinality
        for col in self.high_cardinality_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Scale numeric
        if self.numeric_cols:
            self.scaler = StandardScaler()
            X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X

    def transform(self, X):
        X = X.copy()
        # Use the same columns as during fit
        if self.low_cardinality_cols:
            ohe_arr = self.ohe.transform(X[self.low_cardinality_cols])
            ohe_df = pd.DataFrame(ohe_arr, columns=self.ohe.get_feature_names_out(self.low_cardinality_cols), index=X.index)
            X = X.drop(self.low_cardinality_cols, axis=1)
            X = pd.concat([X, ohe_df], axis=1)
        for col in self.high_cardinality_cols:
            le = self.label_encoders[col]
            X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        return X

def handle_missing_values(data):
    # Handle missing values in the dataset
    # Fill missing values with the mean for numerical features
    for column in data.select_dtypes(include=['float64', 'int']).columns:
        data[column] = data[column].fillna(data[column].mean())
    return data


def encode_categorical_features(data, one_hot_max=10):
    # Encode categorical features: one-hot for low cardinality, label for high cardinality
    from sklearn.preprocessing import LabelEncoder
    data = data.copy()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_unique = data[col].nunique()
        if n_unique <= one_hot_max:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data


def scale_features(data):
    # Only scale numeric columns
    from sklearn.preprocessing import StandardScaler
    data = data.copy()
    num_cols = data.select_dtypes(include=['float64', 'int']).columns
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

def split_data(data, target_column, test_size=0.2):
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)