import numpy as np
import re

# The CUSTOMIZABLE things:
temperature = 1 
question = "The capital of France is" 
top_guess_amount = 5 
embedding_dim = 16

# Page 2: The Corpus
with open("corpus.txt", "r") as file:
    text = re.sub(r'[^\w\s]', '', file.read().lower())
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# Page 4: Tokenization
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}

# Page 9: Embedding Table (Initialized with random numbers)
# In NumPy, this is just a standard 2D array (Matrix)
embedding_table = np.random.randn(vocab_size, embedding_dim) * 0.1

# Page 13: Learned Weight Matrices (Attention)
# These replace the nn.Linear layers from PyTorch
Wq = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wk = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wv = np.random.randn(embedding_dim, embedding_dim) * 0.1

# Page 19: Layer Norm Parameters (Gamma and Beta)
gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

# The Output "Mouth" (Language Model Head)
lm_head_weight = np.random.randn(embedding_dim, vocab_size) * 0.1

# Page 16: Softmax function written in NumPy
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 2. Improved Question Function with "Thinking" (The NumPy Forward Pass)
def ask_question(prompt, temp=temperature):
    words_in_prompt = prompt.lower().split()

    # Check if words exist in our vocab
    for w in words_in_prompt:
        if w not in word_to_int:
            return f"Error: '{w}' is not in my corpus!"

    # Page 10: Convert words to IDs
    input_ids = [word_to_int[w] for w in words_in_prompt]

    # Page 9: Look up vectors in the Embedding Matrix
    x = embedding_table[input_ids]

    # Page 13: The Attention Head (Matrix Multiplication)
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # Page 14: Contextual Relevance (Q * K)
    scores = (Q @ K.T) / np.sqrt(embedding_dim)

    # Page 16: Softmax to get percentages
    weights = softmax(scores)

    # Page 17: Apply Values to weights
    attention_output = weights @ V

    # Page 18: Residual Connection (Add original x to the result)
    x = x + attention_output

    # Page 19: Layer Norm (Gamma and Beta)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    x = gamma * (x - mean) / (std + 1e-6) + beta

    # Final Step: Project to Vocabulary (The "Mouth")
    # We take the vector of the LAST word to predict the next
    last_word_vector = x[-1]
    logits = (last_word_vector @ lm_head_weight) / temp
    
    # Final Softmax for probability distribution
    probs = softmax(logits)

    # See the top_guess_amount guesses
    top_indices = np.argsort(probs)[-top_guess_amount:][::-1]
    print(f"\nAI is thinking (NumPy Mode)...")
    for i, idx in enumerate(top_indices):
        print(f"  Option {i + 1}: {int_to_word[idx]} ({probs[idx] * 100:.1f}%)")

    # Page 16: Choice
    next_word_id = np.argmax(probs)
    return int_to_word[next_word_id]

# BACKWARDS PASS BEGINS
# --- STEP 1: INITIALIZE GRADIENTS ---
# These are the "Fixes" we will apply to the weights
learning_rate = 0.005

for epoch in range(iterations):
    # --- FORWARD PASS (As before, but saving variables for backprop) ---
    # 1. Embedding
    x = embedding_table[tokens] # Shape: [Sequence_Length, 16]
    
    # 2. Attention
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    scores = (Q @ K.T) / np.sqrt(embedding_dim)
    probs = softmax(scores) # The Attention Map (Page 16)
    context = probs @ V
    
    # 3. Output
    logits = context @ lm_head_weight # Page 21
    predictions = softmax(logits)

    # --- BACKWARD PASS (Page 3: Fixing what's wrong) ---
    
    # 1. Error at the Output (Difference between guess and reality)
    # We create a 'target' matrix (One-Hot) to see where we missed
    target = np.zeros_like(predictions)
    for i, t_idx in enumerate(tokens[1:]): # Predicting next word
        target[i, t_idx] = 1.0
    
    d_logits = (predictions[:-1] - target) / len(tokens) # Initial error signal

    # 2. Gradient for the Mouth (lm_head)
    # How much did the output weights contribute to the error?
    d_lm_head = context[:-1].T @ d_logits
    d_context = d_logits @ lm_head_weight.T

    # 3. Gradient for Attention (The Calculus of Page 14)
    # This is the "Chain Rule" in action
    d_V = probs[:-1].T @ d_context
    d_probs = d_context @ V.T
    
    # Backprop through Softmax
    d_scores = probs[:-1] * (d_probs - np.sum(d_probs * probs[:-1], axis=-1, keepdims=True))
    d_scores /= np.sqrt(embedding_dim)

    # Gradients for Wq, Wk, Wv (The Lenses)
    d_Wq = x[:-1].T @ (d_scores @ K[:-1])
    d_Wk = x[:-1].T @ (d_scores.T @ Q[:-1])

    # --- STEP 2: UPDATE WEIGHTS (The Optimizer / Page 6) ---
    # Weight_new = Weight_old - (LR * Gradient)
    Wq -= learning_rate * d_Wq
    Wk -= learning_rate * d_Wk
    Wv -= learning_rate * d_Wv
    lm_head_weight -= learning_rate * d_lm_head

    if epoch % 100 == 0:
        # Calculate Loss (Cross Entropy)
        loss = -np.mean(np.sum(target * np.log(predictions[:-1] + 1e-9), axis=1))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
