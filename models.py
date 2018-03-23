import numpy as np

class Net():
    def __init__(self, weights):
        # an affine operation: y = Wx + b
        self.dimensions = [(24, 40), (40, 40), (40, 4)]
        self.layers = []
        self.biases = []
        if not weights is None:
            tmp = 0
            for d in self.dimensions:
                length = np.prod(d)
                self.layers.append(np.reshape(weights[tmp: tmp + length], d))
                tmp += length
                self.biases.append(np.reshape(weights[tmp: tmp + d[-1]], (1, d[-1])))
                tmp += d[-1]

    def set_model_params(self, weights):
        self.layers = []
        self.biases = []

        tmp = 0
        for d in self.dimensions:
            length = np.prod(d)
            self.layers.append(np.reshape(weights[tmp: tmp + length], d))
            tmp += length
            self.biases.append(np.reshape(weights[tmp: tmp + d[-1]], (1, d[-1])))
            tmp += d[-1]
        
    def forward(self, x):
        working_tensor = x
        for i in xrange(len(self.layers)):
            affine = np.dot(working_tensor, self.layers[i]) + self.biases[i]
            working_tensor = np.tanh(affine)
        return working_tensor[0]

    def num_flat_features(self):
        ret = 0
        for d in self.dimensions:
            ret += np.prod(d)
            ret += d[-1]
        return ret