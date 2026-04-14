import numpy as np

class NeuralBigramModel:
    def __init__(self, vocab_size, hidden_size, context_size, rng):
        self.context_size = context_size
        input_limit = np.sqrt(6 / (vocab_size + hidden_size))
        output_limit = np.sqrt(6 / (hidden_size + vocab_size))

        self.input_to_hidden_weights = rng.uniform(
            -input_limit, input_limit, size=(vocab_size, hidden_size)
        )
        self.hidden_bias = np.zeros((1, hidden_size))
        self.hidden_to_output_weights = rng.uniform(
            -output_limit, output_limit, size=(hidden_size, vocab_size)
        )
        self.output_bias = np.zeros((1, vocab_size))

    def train(
        self,
        context_indices,
        target_indices,
        epochs=20,
        learning_rate=0.1,
        batch_size=256,
        report_every=None,
    ):
        sample_count = len(context_indices)
        training_log = []

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(sample_count)
            shuffled_contexts = context_indices[permutation]
            shuffled_targets = target_indices[permutation]
            epoch_loss = 0.0

            for start in range(0, sample_count, batch_size):
                stop = start + batch_size
                batch_contexts = shuffled_contexts[start:stop]
                batch_targets = shuffled_targets[start:stop]

                hidden_linear, hidden_activation, output_linear = self.forward(batch_contexts)
                probabilities = softmax(output_linear)
                batch_loss = cross_entropy_loss_from_indices(probabilities, batch_targets)
                epoch_loss += batch_loss * len(batch_contexts)

                grad_logits = probabilities
                grad_logits[np.arange(len(batch_targets)), batch_targets] -= 1.0
                grad_logits /= len(batch_targets)

                grad_hidden_to_output_weights = hidden_activation.T @ grad_logits
                grad_output_bias = grad_logits.sum(axis=0, keepdims=True)

                grad_hidden_activation = grad_logits @ self.hidden_to_output_weights.T
                grad_hidden_linear = (
                    grad_hidden_activation * hidden_activation * (1.0 - hidden_activation)
                )
                grad_hidden_bias = grad_hidden_linear.sum(axis=0, keepdims=True)

                grad_input_to_hidden_weights = np.zeros_like(self.input_to_hidden_weights)
                np.add.at(
                    grad_input_to_hidden_weights,
                    batch_contexts.reshape(-1),
                    np.repeat(grad_hidden_linear / self.context_size, self.context_size, axis=0),
                )

                self.hidden_to_output_weights -= learning_rate * grad_hidden_to_output_weights
                self.output_bias -= learning_rate * grad_output_bias
                self.input_to_hidden_weights -= learning_rate * grad_input_to_hidden_weights
                self.hidden_bias -= learning_rate * grad_hidden_bias

            average_loss = epoch_loss / sample_count
            if report_every and (epoch == 1 or epoch % report_every == 0 or epoch == epochs):
                training_log.append((epoch, average_loss))

        return training_log

    def forward(self, context_indices):
        context_vectors = self.input_to_hidden_weights[context_indices]
        hidden_linear = context_vectors.mean(axis=1) + self.hidden_bias
        hidden_activation = sigmoid(hidden_linear)
        output_linear = hidden_activation @ self.hidden_to_output_weights + self.output_bias
        return hidden_linear, hidden_activation, output_linear

    def predict_distribution(self, context_indices, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        context_array = np.array([context_indices], dtype=int)
        _, _, logits = self.forward(context_array)
        adjusted_logits = logits / temperature
        return softmax(adjusted_logits)[0]

    def predict_next_index(self, context_indices, rng, temperature=1.0):
        probabilities = self.predict_distribution(context_indices, temperature=temperature)
        return rng.choice(len(probabilities), p=probabilities)

def sigmoid(inputs):
    return 1.0 / (1.0 + np.exp(-inputs))


def softmax(logits):
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy_loss_from_indices(probabilities, target_indices):
    clipped = np.clip(probabilities[np.arange(len(target_indices)), target_indices], 1e-12, 1.0)
    return -np.mean(np.log(clipped))
