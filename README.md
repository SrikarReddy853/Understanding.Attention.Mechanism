
#  UNDERSTANDING ATTENTION MECHANISM
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Import PyTorch for tensor operations
# Import functional API for softmax
# Import matplotlib for visualization
# STEP 1: INPUT SENTENCE (WORDS)
# Students can change this sentence to test different inputs
words = ["I", "love", "AI"]
print("Input Words:", words)
# STEP 2: CONVERT WORDS TO ENCODER OUTPUTS (TENSORS)
# In real scenarios, these would come from an embedding layer or encoder
# Here we manually assign a 2-dimensional vector for simplicity
encoder_outputs = torch.tensor([
[0.1, 0.3],   # Representation for "I"
[0.8, 0.2],   # Representation for "love"
[0.9, 0.7]
# Representation for "AI"
], dtype=torch.float)
print("\nEncoder Outputs (Keys & Values):")
print(encoder_outputs)   # Display tensor representations
# STEP 3: DECODER HIDDEN STATE (QUERY)
# The decoder hidden state represents the current decoding step
# This acts as the "Query" to ask the encoder what to focus on
decoder_hidden = torch.tensor([0.7, 0.6], dtype=torch.float)
print("\nDecoder Hidden State (Query):", decoder_hidden)
# STEP 4: COMPUTE ATTENTION SCORES
# Attention score = dot product between Query and each Key (encoder
# outputs)
# Higher score = more relevant word
attention_scores = torch.matmul(encoder_outputs, decoder_hidden)
# Convert scores to probabilities (softmax) so they sum to 1
attention_weights = F.softmax(attention_scores, dim=0)
print("\nAttention Weights (Original Query):", attention_weights)
# STEP 5: COMPUTE CONTEXT VECTOR
# Context vector = weighted sum of encoder outputs
# This vector represents the combined information the decoder should focus
# on
context_vector = torch.sum(attention_weights.unsqueeze(1) *
encoder_outputs, dim=0)
print("\nContext Vector:", context_vector)
# STEP 6: VISUALIZE ORIGINAL ATTENTION WEIGHTS
weights_np = attention_weights.detach().numpy()  # Convert tensor to NumPy
# for plotting
plt.figure(figsize=(6, 4))
plt.bar(words, weights_np)
# weights
# Bar chart of attention
plt.title("Original Attention Weight Distribution")
plt.xlabel("Input Words")
plt.ylabel("Attention Weight")
plt.show()
# STEP 7: CHANGE DECODER HIDDEN STATE TO SEE ATTENTION SHIFT
# Simulate a different decoder query to see how attention changes
decoder_hidden_new = torch.tensor([0.2, 0.9], dtype=torch.float)
new_scores = torch.matmul(encoder_outputs, decoder_hidden_new)
new_weights = F.softmax(new_scores, dim=0)
print("\nNew Decoder Hidden State:", decoder_hidden_new)
print("New Attention Weights:", new_weights)
# STEP 8: VISUALIZE ATTENTION SHIFT WITH COMPARISON
new_weights_np = new_weights.detach().numpy()
# Plot original and new attention weights together
plt.figure(figsize=(8, 4))
plt.bar(words, weights_np, alpha=0.6, label='Original Query')
plt.bar(words, new_weights_np, alpha=0.6, label='New Query')
plt.title("Attention Shift Between Queries")
plt.xlabel("Input Words")
plt.ylabel("Attention Weight")
plt.legend()
plt.show()
# Optional: Draw arrows to show how attention shifts visually
plt.figure(figsize=(8, 4))
for i, (w_old, w_new) in enumerate(zip(weights_np, new_weights_np)):
    plt.arrow(i, w_old, 0, w_new - w_old, head_width=0.1,
              head_length=0.02, color='red')
    plt.bar(words, new_weights_np, alpha=0.6)
plt.title("Attention Shift Arrows (Red)")
plt.xlabel("Input Words")
plt.ylabel("Attention Weight")
plt.show()
