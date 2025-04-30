import yaml
import json
import os
import torch
import time  # Added for timing
from setting import initialize_settings
from gnn_model import gnn_model
from AR_model import AR_model
from greedy_placement import greedy_placement
import numpy as np
from msg_nodes_pytorch import msg_server_processing
from msg_services_pytorch import msg_component_processing
from ar_pytorch import AR_pytorch

# Load configuration from YAML file
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extract configuration parameters
num_epochs = config["model"]["num_epochs"]
num_samples = config["model"]["num_samples"]
num_layers = config["model"]["num_layers"]
hidden_dim = config["model"]["hidden_dim"]
cpu_hidden_dim = config["model"]["cpu_hidden_dim"]
device = config["model"]["device"]

# Determine device and set dim accordingly
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        dim = hidden_dim
    else:
        device = "cpu"
        dim = cpu_hidden_dim
else:
    device = device.lower()
    dim = hidden_dim if device == "cuda" else cpu_hidden_dim

# Initialize or load parameters from settings.json
parameters = initialize_settings(
    config_path="configs/config.yaml", settings_path="configs/setting.json"
)


# Start timing the entire script
total_start_time = time.time()


# Load configuration from config.yaml
with open("configs/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract parameters from config
num_samples = config["model"]["num_samples"]  # Number of samples (64 from config.yaml)
dim = num_samples  # Assuming dim is the total number of samples

# Define output directory
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Initialize loop variables
i = 1
j = dim

# Iterate over instances
# for ins in range(1, num_samples + 1):
# if j <= num_samples:
#  for x in range(i, j + 1):
#      file_path = f"data/generated/instance_{x}.json"
#      if os.path.exists(file_path):
#          with open(file_path, "r") as f:
#              data = json.load(f)
#              print(f"Processing instance_{x}.json")


# Assuming total_start_time, num_samples, parameters, device, dim, output_dir are defined elsewhere
for x in range(1, num_samples + 1):
    file_path = f"data/generated/instance_{x}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            print(f"Processing instance_{x}.json")
            # Extract sections
            sections = {
                "nodes": data.get("computingNodes", [x]),
                "helpers": data.get("helperNodes", [x]),
                "users": data.get("usersNodes", [x]),
                "services": data.get("services", [x]),
                "newcomponentsConnections": data.get("componentConnections", [x]),
                "newInfraConnections": data.get("infraConnections", [x]),
                "results": data.get("results", {x}),
            }

            # Save each section to a separate JSON file
            for section_name, section_data in sections.items():
                output_file = os.path.join(output_dir, f"{section_name}_{x}.json")
                with open(output_file, "w") as out_f:
                    json.dump(section_data, out_f, indent=4)
                print(f"Saved {section_name} to {output_file}")

            msg_server_processing(
                x,
                parameters,
                device=device,
            )
            msg_component_processing(
                x,
                parameters,
                device=device,
            )
            AR_pytorch(
                x,
                parameters,
                device=device,
            )
    else:
        print(f"File {file_path} does not exist.")

print("Processing complete.")

# End timing the entire script
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Processed {num_samples} instances successfully.")
print(f"Device used: {device}, Hidden dimension: {dim}")
print(f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
