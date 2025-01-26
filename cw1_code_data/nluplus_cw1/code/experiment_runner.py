import os
import itertools
import subprocess

# Define the parameter values
learning_rates = [0.5, 0.1, 0.05, 1]
hdims = [25, 50, 100]
lookbacks = [0, 2, 5, 7, 10]

data_dir = "/Users/radhikasatish/nlu_cw1/NLU-Coursework-1-Recurrent-Neural-Networks/cw1_code_data/nluplus_cw1/data"  # Replace with your actual data directory
runner_path = "/Users/radhikasatish/nlu_cw1/NLU-Coursework-1-Recurrent-Neural-Networks/cw1_code_data/nluplus_cw1/code/runner.py"
# Create all combinations of parameters
parameter_combinations = list(itertools.product(hdims, lookbacks, learning_rates))

# Directory to store the output logs
output_dir = "experiment_outputs"
os.makedirs(output_dir, exist_ok=True)

# Store subprocesses
processes = []

# Iterate over each combination and run the experiment
for hdim, lookback, learning_rate in parameter_combinations:
    output_file = os.path.join(
        output_dir, f"hdim_{hdim}_lookback_{lookback}_lr_{learning_rate}.txt"
    )
    command = [
        "python", runner_path, "train-lm-rnn", data_dir,
        str(hdim), str(lookback), str(learning_rate)
    ]


    with open(output_file, "w") as outfile:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT, check=True)


print("All experiments completed. Outputs are saved in the 'experiment_outputs' folder.")
