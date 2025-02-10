import os
import itertools
import subprocess

# Define the parameter values
learning_rates = [0.5]
hdims = [25, 50]
lookbacks = [0, 2, 5, 7, 10]
models = ["train-np-gru", "train-np-rnn"]  # Models to test


data_dir = "/Users/radhikasatish/nlu_cw1/NLU-Coursework-1-Recurrent-Neural-Networks/cw1_code_data/nluplus_cw1/data"  # Replace with your actual data directory
runner_path = "/Users/radhikasatish/nlu_cw1/NLU-Coursework-1-Recurrent-Neural-Networks/cw1_code_data/nluplus_cw1/code/runner.py"


# Create all combinations of parameters
parameter_combinations = list(itertools.product(models, hdims, lookbacks, learning_rates))

# Directory to store the output logs
output_dir = "train_size_50000_long_short_med_20_epochs"

os.makedirs(output_dir, exist_ok=True)

# Iterate over each combination and run the experiment
for model, hdim, lookback, learning_rate in parameter_combinations:
    output_file = os.path.join(
        output_dir, f"{model}_hdim_{hdim}_lookback_{lookback}_lr_{learning_rate}.txt"
    )
    command = [
        "python", runner_path, model, data_dir,

        str(hdim), str(lookback), str(learning_rate)
    ]

    with open(output_file, "w") as outfile:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT, check=True)

print("All experiments completed. Outputs are saved in the 'experiment_outputs' folder.")
