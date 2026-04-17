import numpy as np
import re

# --- 1. CONFIGURATION (The "Knobs") ---
iterations = 2000  # Slide 6: Number of training tests
learning_rate = .6  # Slide 6: Size of the "Fix"
temperature = 1.0  # Slide 16: Radicality/Creativity
embedding_dim = 16  # Slide 9: Dimension of word "flavor"
top_guess_amount = 5  # Number of options to show
question = "The capital of France is"

# --- 2. DATA PREPARATION (Slide 2 & 4) ---
with open("corpus.txt", "r") as file:
    # Clean punctuation so "France." and "France" are the same token
    text = re.sub(r'[^\w\s]', '', file.read().lower())

words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# Tokenization: Mapping words to ID numbers
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}
tokens = np.array([word_to_int[w] for w in words])

# --- 3. INITIALIZATION (Slide 13 & 19) ---
# Randomly starting weights before the "Millions of tests"
embedding_table = np.random.randn(vocab_size, embedding_dim) * 0.1
Wq = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wk = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wv = np.random.randn(embedding_dim, embedding_dim) * 0.1
lm_head_weight = np.random.randn(embedding_dim, vocab_size) * 0.1


def softmax(x):
    # Slide 16: Percentage Logic
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# --- 4. TRAINING LOOP (The Backward Pass - Slide 3, 6, 21, 22) ---
print(f"Starting NumPy Training on {vocab_size} unique words...")

for epoch in range(iterations + 1):
    # --- FORWARD PASS (Slide 10-17) ---
    x_emb = embedding_table[tokens]
    Q = x_emb @ Wq
    K = x_emb @ Wk
    V = x_emb @ Wv

    # Slide 14: Relevance scores
    scores = (Q @ K.T) / np.sqrt(embedding_dim)
    attn_weights = softmax(scores)
    context = attn_weights @ V

    logits = context @ lm_head_weight
    probs = softmax(logits)

    # --- BACKWARD PASS (Slide 22: Calculus) ---
    targets = tokens[1:]
    # Slice everything to N-1 to match the target length
    x_train = x_emb[:-1]  # [389, 16]
    q_train = Q[:-1]  # [389, 16]
    k_train = K[:-1]  # [389, 16]
    v_train = V[:-1]  # [389, 16]
    # Crop the attention map to a [389, 389] square
    attn_train = attn_weights[:-1, :-1]

    # 1. Error signal (How far off were we?)
    d_logits = probs[:-1].copy()
    for i, target_idx in enumerate(targets):
        d_logits[i, target_idx] -= 1.0
    d_logits /= len(targets)

    # 2. Mouth Gradients (lm_head)
    dW_head = context[:-1].T @ d_logits  # [16, 389] @ [389, vocab] = [16, vocab]
    d_context = d_logits @ lm_head_weight.T  # [389, vocab] @ [vocab, 16] = [389, 16]

    # 3. Value Gradients (Wv)
    d_V = attn_train.T @ d_context  # [389, 389] @ [389, 16] = [389, 16]
    dWv = x_train.T @ d_V  # [16, 389] @ [389, 16] = [16, 16]

    # 4. Attention Map Gradients (Softmax)
    d_attn_weights = d_context @ v_train.T  # [389, 16] @ [16, 389] = [389, 389]
    d_scores = attn_train * (d_attn_weights - np.sum(d_attn_weights * attn_train, axis=-1, keepdims=True))
    d_scores /= np.sqrt(embedding_dim)

    # 5. Lens Gradients (Wq and Wk) - FIXED ORDER
    dWq = x_train.T @ (d_scores @ k_train)  # [16, 389] @ ([389, 389] @ [389, 16]) = [16, 16]
    dWk = x_train.T @ (d_scores.T @ q_train)  # [16, 389] @ ([389, 389] @ [389, 16]) = [16, 16]

    # --- UPDATE WEIGHTS (Slide 6) ---
    lm_head_weight -= learning_rate * dW_head
    Wv -= learning_rate * dWv
    Wq -= learning_rate * dWq
    Wk -= learning_rate * dWk

    # Update Embeddings (Meanings)
    for i, t_idx in enumerate(tokens[:-1]):
        grad_emb = (d_scores[i] @ k_train @ Wq.T) + (d_scores.T[i] @ q_train @ Wk.T) + (d_V[i] @ Wv.T)
        embedding_table[t_idx] -= learning_rate * grad_emb

    if epoch % 500 == 0:
        loss = -np.mean(np.log(probs[:-1][range(len(targets)), targets] + 1e-9))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# --- 5. INFERENCE (The "Final Exam") ---
def ask_question(prompt, temp=temperature):
    words_in_prompt = prompt.lower().split()
    for w in words_in_prompt:
        if w not in word_to_int: return f"Error: '{w}' not in corpus"

    # Step-by-Step Forward Pass for the user question
    input_ids = [word_to_int[w] for w in words_in_prompt]
    x = embedding_table[input_ids]

    # Attention Logic
    q_act = x @ Wq
    k_act = x @ Wk
    v_act = x @ Wv
    score_act = (q_act @ k_act.T) / np.sqrt(embedding_dim)
    weight_act = softmax(score_act)
    context_act = weight_act @ v_act

    # Output Logic with Temperature
    logits = (context_act[-1] @ lm_head_weight) / temp
    probs = softmax(logits)

    # Show Top Guesses
    top_indices = np.argsort(probs)[-top_guess_amount:][::-1]
    print(f"\nAI is thinking about: '{prompt}'")
    for i, idx in enumerate(top_indices):
        print(f"  Option {i + 1}: {int_to_word[idx]} ({probs[idx] * 100:.1f}%)")

    return int_to_word[np.argmax(probs)]


# --- FINAL EXECUTION ---
print("\n" + "=" * 30)
answer = ask_question(question)
print(f"Final Prediction: {answer}")
