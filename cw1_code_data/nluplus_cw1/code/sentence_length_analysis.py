import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


data_dir = "data"
files = ["/home/gvidias/nlu_cw1/cw1_code_data/nluplus_cw1/data/wiki-train.txt", "/home/gvidias/nlu_cw1/cw1_code_data/nluplus_cw1/data/wiki-dev.txt"]
files2 = ['wiki-train', 'wiki-dev']
bins = [0, 10, 20, 30, 40, 50, float("inf")]
bin_labels = ["<10", "10-20", "21-30", "31-40", "41-50", ">50"]

# Function to count tokens in a sentence
def count_tokens(sentence):
    return len(sentence.split())

# Process a single file (only first 1000 sentences)
def process_file(file):
    sentence_counts = []
    if not os.path.exists(file):
        print(f"File not found: {file}")
        return sentence_counts
    
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            sentence = line.split(".")[0].strip()  # Extract text before the first period
            if sentence:
                num_tokens = count_tokens(sentence)
                sentence_counts.append(num_tokens)
    
    return sentence_counts

# Get sentence lengths for each file
lengths_per_file = {file: process_file(file) for file in files}

# Create a DataFrame with separate percentage columns for each file
df = pd.DataFrame(index=bin_labels)

for file, lengths, label in zip(files, lengths_per_file.values(), files2):
    length_bins = pd.cut(lengths, bins=bins, labels=bin_labels, right=True)
    length_distribution = Counter(length_bins)
    total_sentences = sum(length_distribution.values())
    length_percentage = {k: (v / total_sentences) * 100 for k, v in length_distribution.items()}
    
    df[label] = [length_percentage.get(label, 0) for label in bin_labels]

df.index.name = "Number of Tokens"

# Save output to a CSV file
df.to_csv("sentence_length_distribution.csv")

# Plot bar chart
ax = df.plot(kind="bar", figsize=(10, 6))
plt.title("Sentence Length Distribution (First 1000 Sentences)")
plt.xlabel("Number of Tokens")
plt.ylabel("Percentage of Sentences")
plt.xticks(rotation=45)
plt.legend(title="Files")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot
plt.savefig("sentence_length_distribution.png")
plt.show()
