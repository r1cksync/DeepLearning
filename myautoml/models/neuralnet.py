import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class NeuralNetClassifier:
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], input_dim=self.input_dim, activation=self.activation))
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
        if self.output_dim == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(self.output_dim, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X, y, epochs=20, batch_size=32):
        self.model.fit(np.array(X), np.array(y), epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        preds = self.model.predict(np.array(X))
        if self.output_dim == 2:
            return (preds > 0.5).astype(int).flatten()
        else:
            return np.argmax(preds, axis=1)

class NeuralNetRegressor:
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], input_dim=self.input_dim, activation=self.activation))
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
        model.add(Dense(self.output_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model

    def fit(self, X, y, epochs=20, batch_size=32):
        self.model.fit(np.array(X), np.array(y), epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(np.array(X)).flatten()

class NeuralNet:
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential()
        model.add(Dense(self.hidden_layers[0], input_dim=self.input_dim, activation=self.activation))
        
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation))
        
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)