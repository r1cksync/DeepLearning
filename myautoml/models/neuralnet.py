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