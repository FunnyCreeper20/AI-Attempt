import numpy as np


class Dense:
    def __init__(self, input_size, output_size, rng):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = rng.uniform(-limit, limit, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.input_cache = None

    def forward(self, inputs):
        self.input_cache = inputs
        return inputs @ self.weights + self.bias

    def backward(self, grad_output, learning_rate):
        batch_size = self.input_cache.shape[0]
        grad_weights = self.input_cache.T @ grad_output / batch_size
        grad_bias = grad_output.mean(axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input


class Sigmoid:
    def __init__(self):
        self.output_cache = None

    def forward(self, inputs):
        self.output_cache = 1.0 / (1.0 + np.exp(-inputs))
        return self.output_cache

    def backward(self, grad_output, learning_rate):
        del learning_rate
        return grad_output * self.output_cache * (1.0 - self.output_cache)


class SequentialNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, grad_output, learning_rate):
        gradient = grad_output
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)


def softmax(logits):
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy_loss(probabilities, targets):
    batch_size = targets.shape[0]
    clipped_probs = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(targets * np.log(clipped_probs)) / batch_size

