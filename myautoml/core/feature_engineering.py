def handle_missing_values(data):
    # Handle missing values in the dataset
    # Example: Fill missing values with the mean for numerical features
    for column in data.select_dtypes(include=['float64', 'int']).columns:
        data[column].fillna(data[column].mean(), inplace=True)
    return data

def encode_categorical_features(data):
    # Encode categorical features using one-hot encoding
    return pd.get_dummies(data, drop_first=True)

def scale_features(data):
    # Scale features to a standard range
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def split_data(data, target_column, test_size=0.2):
    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)