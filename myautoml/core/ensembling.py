def blend_models(models, X, y):
    predictions = [model.predict(X) for model in models]
    blended_prediction = sum(predictions) / len(models)
    return blended_prediction

def stack_models(models, X_train, y_train, X_test):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Generate predictions for the meta-model
    meta_features = []
    for model in models:
        model.fit(X_train_meta, y_train_meta)
        preds = model.predict(X_val_meta)
        meta_features.append(preds)
    
    # Stack the predictions
    meta_features = np.column_stack(meta_features)
    
    # Train a meta-model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y_val_meta)
    
    # Generate predictions for the test set
    test_meta_features = []
    for model in models:
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_meta_features.append(test_preds)
    
    test_meta_features = np.column_stack(test_meta_features)
    final_predictions = meta_model.predict(test_meta_features)
    
    return final_predictions