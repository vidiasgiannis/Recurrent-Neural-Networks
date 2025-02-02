import pandas as pd

def split_dependencies(file_path):
    # Load the dataset (assuming tab-separated values)
    df = pd.read_csv(file_path, sep='\t', dtype=str)

    # Compute the absolute difference
    df['dependency_distance'] = abs(df['verb_idx'].astype(int) - df['subj_idx'].astype(int))

    # Define short and long-range dependencies
    short_dep = df[df['dependency_distance'].isin([1, 2, 3])]
    long_dep = df[df['dependency_distance'] > 3]

    # Define output filenames
    short_dep_file = file_path.replace(".txt", "_short.txt")
    long_dep_file = file_path.replace(".txt", "_long.txt")

    # Save to separate text files keeping the original format
    short_dep.to_csv(short_dep_file, sep='\t', index=False, header=False, mode='w')
    long_dep.to_csv(long_dep_file, sep='\t', index=False, header=False, mode='w')

    print(f"Short-range dependencies saved to {short_dep_file}")
    print(f"Long-range dependencies saved to {long_dep_file}")

# Example usage
file_path = "/home/gvidias/nlu_cw1/cw1_code_data/nluplus_cw1/data/wiki-test.txt"
split_dependencies(file_path)
