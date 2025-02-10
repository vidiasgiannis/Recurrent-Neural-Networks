import os
import csv
import re

def extract_values_from_filename(filename):
    """Extract model name, hdim, lookback, and lr values from the filename."""
    match = re.search(r'([a-zA-Z0-9-_]+)_hdim_(\d+)_lookback_(\d+)_lr_([\d\.]+)', filename)
    if match:
        model_name = match.group(1)
        hdim = int(match.group(2))
        lookback = int(match.group(3))
        lr = float(match.group(4).rstrip('.'))  # Remove any trailing period
        return model_name, hdim, lookback, lr
    return None, None, None, None

def extract_loss_accuracy(filepath):
    """Extract loss and accuracy values from the file contents, ensuring only the last valid occurrences are used."""
    results = {
        "Full Set": {"loss": None, "accuracy": None},
        "Short Set": {"loss": None, "accuracy": None},
        "Medium Set": {"loss": None, "accuracy": None},
        "Long Set": {"loss": None, "accuracy": None}
    }
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Reverse iterate to find last occurrences first
    for line in reversed(lines):
        match = re.search(r'(Full|Short|Medium|Long) Set - Loss: ([\d\.]+), Accuracy: ([\d\.]+)', line)
        if match:
            set_name = match.group(1) + " Set"
            results[set_name]["loss"] = float(match.group(2))
            results[set_name]["accuracy"] = float(match.group(3))
            
            # Stop once all sets have been found
            if all(v["loss"] is not None and v["accuracy"] is not None for v in results.values()):
                break
    
    return results

def process_files(directory, output_csv):
    """Process all files in the directory and save extracted data to a CSV file."""
    data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            model_name, hdim, lookback, lr = extract_values_from_filename(filename)
            if hdim is None:
                print(f"Skipping file (no valid hdim found): {filename}")
                continue
            
            loss_accuracy_data = extract_loss_accuracy(filepath)
            
            # Ensure at least one loss/accuracy value is not None before appending
            if all(v["loss"] is None and v["accuracy"] is None for v in loss_accuracy_data.values()):
                print(f"Skipping file (no valid loss/accuracy found): {filename}")
                continue
            
            print(f"Processing file: {filename}")
            data.append([
                model_name, hdim, lookback,
                loss_accuracy_data["Full Set"]["loss"],loss_accuracy_data["Short Set"]["loss"], 
                loss_accuracy_data["Medium Set"]["loss"],
                loss_accuracy_data["Long Set"]["loss"],  loss_accuracy_data["Full Set"]["accuracy"], loss_accuracy_data["Short Set"]["accuracy"],
                loss_accuracy_data["Medium Set"]["accuracy"], loss_accuracy_data["Long Set"]["accuracy"]
            ])
    
    # Sort data by model name, hdim, then lookback
    data.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model_name", "hdim", "lookback", "full_loss", "short_loss",  "medium_loss",  "long_loss", "full_acc", "short_acc", "medium_acc", "long_acc"])
        writer.writerows(data)
    
    print(f"CSV file saved: {output_csv}")

# Usage
directory = "/Users/radhikasatish/nlu_cw1/NLU-Coursework-1-Recurrent-Neural-Networks/cw1_code_data/nluplus_cw1/train_size_50000_long_short_med_20_epochs"  # Change this to your actual directory
output_csv = "20_epochs_output"
process_files(directory, output_csv)
