import pandas as pd
import matplotlib.pyplot as plt

def plot_dependency_distribution(file_path):
    # Load the dataset (assuming tab-separated values)
    df = pd.read_csv(file_path, sep='\t')

    # Compute the absolute difference
    df['dependency_distance'] = abs(df['verb_idx'] - df['subj_idx'])

    # Define bins for categorizing the distances
    bins = [0, 5, 10, df['dependency_distance'].max()]
    labels = ['0 - 5', '5 - 10', '> 10']
    
    # Categorize the data into bins
    df['distance_category'] = pd.cut(df['dependency_distance'], bins=bins, labels=labels, right=False)

    # Count occurrences in each bin
    bin_counts = df['distance_category'].value_counts().sort_index()

    # Plot distribution
    plt.figure(figsize=(8, 5))
    bin_counts.plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
    plt.xlabel("Dependency Distance Range")
    plt.ylabel("Sentence Count")
    plt.title("Distribution of Sentences by Dependency Distance in wiki-test file")
    plt.xticks(rotation=45)
    # Save the plot
    plt.savefig("new Distribution of Sentences by Dependency Distance in wiki-test file")
    plt.show()

    # Print the counts for each bin
    print("Distribution of Dependency Distances:")
    print(bin_counts)

# Example usage
file_path = "/home/gvidias/nlu_cw1/cw1_code_data/nluplus_cw1/data/wiki-test.txt"  # Update with your actual file path
plot_dependency_distribution(file_path)