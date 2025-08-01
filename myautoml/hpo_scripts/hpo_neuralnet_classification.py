import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from myautoml.core.hpo_optuna import run_optuna_hpo

df = pd.read_csv('train_new.csv')
TARGET_COL = df.columns[-1]
X = df.drop(['id', TARGET_COL], axis=1)
y = df[TARGET_COL]

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

num_classes = len(le.classes_)

def build_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_scaled.shape[1],)))
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, 256)
        model.add(keras.layers.Dense(num_hidden, activation='relu'))
        dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

param_space = {
    'build_fn': build_model,
    'batch_size': lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128]),
    'epochs': lambda trial: trial.suggest_int('epochs', 20, 100),
}

best_params, best_score, study = run_optuna_hpo(
    build_model,
    param_space,
    X_scaled,
    y_enc,
    metric='accuracy',
    direction='maximize',
    n_trials=30,
    use_hyperband=False,
    is_classification=True,
    is_neuralnet=True
)
print('Optuna TPE best params:', best_params)
print('Optuna TPE best score:', best_score)

best_params_hb, best_score_hb, study_hb = run_optuna_hpo(
    build_model,
    param_space,
    X_scaled,
    y_enc,
    metric='accuracy',
    direction='maximize',
    n_trials=30,
    use_hyperband=True,
    is_classification=True,
    is_neuralnet=True
)
print('Optuna Hyperband best params:', best_params_hb)
print('Optuna Hyperband best score:', best_score_hb)
