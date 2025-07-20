import numpy as np

np.random.seed(0)


class ActivationFunction:
    FUNC_TYPES = {"ReLU", "Sigmoid", "Softmax", "Linear"}

    def __init__(self, name):
        if name not in ActivationFunction.FUNC_TYPES:
            raise NotImplementedError(f"activation function not implemented for type: {name}")
        self.name = name

    def forward(self, inputs):
        if self.name == "ReLU":
            return np.maximum(0, inputs)
        if self.name == "Sigmoid":
            return 1 / (1 + np.exp(inputs))
        if self.name == "Softmax":
            exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        if self.name == "Linear":
            return inputs


class LayerDense:
    def __init__(self, units, n_inputs, activation_function, weights=None, biases=None):
        self.units = units
        self.n_inputs = n_inputs
        self.weights = 0.1 * np.random.randn(n_inputs, units) if weights is None else weights
        self.biases = np.zeros((1, units)) if biases is None else biases
        self.activation_function = ActivationFunction(activation_function)

    def forward(self, batch):
        output = np.dot(batch, self.weights) + self.biases
        return self.activation_function.forward(output)

    def set_weights(self, weights, biases):
        if weights.shape != self.weights.shape or biases.shape != biases.shape:
            raise Exception("shape mismatch while setting weights")
        self.weights = weights
        self.biases = biases

    def get_weights(self):
        return self.weights, self.biases


class SequentialModel:
    def __init__(self):
        self.layers = []

    def add_layer(self, units, n_inputs=None, activation_function="Linear", weights=None, biases=None):
        if n_inputs:
            layer = LayerDense(units, n_inputs, activation_function, weights, biases)
        elif len(self.layers) > 0:
            previous_units = self.layers[-1].units
            layer = LayerDense(units, previous_units, activation_function, weights, biases)
        else:
            raise Exception("number of inputs cannot be None for input layer")
        self.layers.append(layer)

    def get_weights(self):
        return self.layers

    def set_weights(self):
        pass

    def copy(self):
        model = SequentialModel()
        for layer in self.layers:
            model.add_layer(layer.units, layer.n_inputs, layer.activation_function.name, layer.weights, layer.biases)
        return model

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def save(self, file_name):
        if not file_name.endswith('.npy'):
            raise Exception(f"invalid model file: {file_name}")
        model = []
        for layer in self.layers:
            model.append([layer.units, layer.n_inputs, layer.activation_function.name, layer.weights, layer.biases])

        np.save(file_name, np.array(model))

    @staticmethod
    def load(file_name: str):
        if not file_name.endswith('.npy'):
            raise Exception(f"invalid model file: {file_name}")
        layers = np.load(file_name, allow_pickle=True)
        model = SequentialModel()
        for layer in layers:
            model.add_layer(layer[0], layer[1], layer[2], layer[3], layer[4])
        return model

# model = SequentialModel()
# model.add_layer(24, 4, "ReLU")
# model.add_layer(24, activation_function="ReLU")
# model.add_layer(4)

# inputs = [[0.93, 1.86, 1.45, 3.56],
#           [3.2, 2.7, 6.87, 4.98]]
#
# print(model.predict(inputs))
# model.save("model.npy")
#
# model = SequentialModel.load("model.npy")
# print(model.predict(inputs))
