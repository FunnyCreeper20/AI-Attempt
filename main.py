import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# The CUSTOMIZABLE things:
iterations = 1500 # 500-1500 iterations is a good amount because it tests the AI that many times
learning_rate = 0.005 # Smaller rate means more stable but longer, while larger means less stable but shorter
temperature = 1 # Smaller number means more conservative. Larger means more radical
question = "The capital of France is" # This is the question being asked. Relate it to the corpus

# Page 2: The Corpus
with open("corpus.txt", "r") as file:
    # This regex removes punctuation so 'France.' would become 'france'
    text = re.sub(r'[^\w\s]', '', file.read().lower())
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

# Page 4: Tokenization
word_to_int = {w: i for i, w in enumerate(vocab)}
int_to_word = {i: w for w, i in word_to_int.items()}
tokens = torch.tensor([word_to_int[w] for w in words], dtype=torch.long)

embedding_dim = 16
embedding_table = nn.Embedding(vocab_size, embedding_dim)

# This is the forward pass
class AttentionHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = (Q @ K.transpose(-2, -1)) / (embedding_dim**0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attention = AttentionHead(emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attention(x)
        return self.ln(x)

# Create the model components
model = TransformerBlock(embedding_dim)
# FIX: Move lm_head outside the loop so it can actually be trained!
lm_head = nn.Linear(embedding_dim, vocab_size)

# 1. Update the Optimizer to include EVERYTHING
# We need to train the Embeddings, the Transformer, AND the Output Head
optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(embedding_table.parameters()) +
    list(lm_head.parameters()),
    lr=learning_rate  # Slightly lower learning rate for more stability
)

print("Starting deep training...")
# Increase to 1500 iterations to let the weights settle
for epoch in range(iterations+1):
    optimizer.zero_grad()

    # Forward pass
    x_emb = embedding_table(tokens)
    output = model(x_emb)
    logits = lm_head(output)

    # We shift the labels so the model learns to predict the NEXT word
    # logits[:-1] is the prediction, tokens[1:] is the actual next word
    loss = F.cross_entropy(logits[:-1], tokens[1:])

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 2. Improved Question Function with "Thinking" (Top Guesses)
def ask_question(prompt, temp=temperature):
    model.eval()
    words_in_prompt = prompt.lower().split()

    # Check if words exist in our vocab
    for w in words_in_prompt:
        if w not in word_to_int:
            return f"Error: '{w}' is not in my corpus!"

    input_ids = torch.tensor([word_to_int[w] for w in words_in_prompt], dtype=torch.long)

    with torch.no_grad(): # Disable the "Backward Pass" (Page 3) math to save memory
        embeddings = embedding_table(input_ids)
        output = model(embeddings) # Pages 13 - 19
        next_word_logits = lm_head(output[-1]) / temp
        probs = F.softmax(next_word_logits, dim=-1)

        # See the top 2 guesses to check the AI's "confidence"
        top_probs, top_ids = torch.topk(probs, 10)
        print(f"\nAI is thinking...")
        for i in range(10):
            print(f"  Option {i + 1}: {int_to_word[top_ids[i].item()]} ({top_probs[i].item() * 100:.1f}%)")

        next_word_id = torch.argmax(probs).item()

    return int_to_word[next_word_id]


# Final Test
answer = ask_question(question)
print(f"Final Prediction: {answer}")
