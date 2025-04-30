import json
import os
import torch
import yaml


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
charnum_service = config["model"]["charnum_service"]
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]
charnum_node = config["model"]["charnum_node"]
charnum_component = config["model"]["charnum_component"]

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


# Sigmoid function
def sigmoid(x):
    return torch.sigmoid(x)


# ReLU function
def relu(x):
    return torch.relu(x)


# Normalize a vector between 0.1 and 1
def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if (
        v_max == v_min or torch.isnan(v_max) or torch.isnan(v_min)
    ):  # Handle NaN or equal values
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


# Main program
def msg_server_processing(
    x,
    parameters,
    device="cpu",
    settings_path="configs/setting.json",
):
    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    UN, VN, WN, WNE, AN, BN, CN = (
        parameters["UN"],
        parameters["VN"],
        parameters["WN"],
        parameters["WNE"],
        parameters["AN"],
        parameters["BN"],
        parameters["CN"],
    )
    # Step 1: Reading and normalizing nodes.json
    # print("Step 1: Reading and normalizing nodes.json")
    with open(f"data/processed/nodes_{x}.json", "r", encoding="utf-8") as f:
        content = f.read()
        nodes_data = json.loads(content)

    num_servers = len(nodes_data)
    # print(f"  - Number of servers: {num_servers}")

    server_vectors = []
    for node in nodes_data:
        specs = node["characteristics"]
        vec = [specs["cpu"], specs["memory"], specs["disk"], specs["reliabilityScore"]]
        server_vectors.append(vec)

    server_vectors = torch.tensor(server_vectors, dtype=torch.float32)  # Shape: (20, 4)
    # print("  - Server vectors shape:", server_vectors.shape)
    for i, col_name in enumerate(["cpu", "memory", "disk", "reliabilityScore"]):
        col = server_vectors[:, i]
        # print(f"  - {col_name}: min={torch.min(col)}, max={torch.max(col)}")

    norm_servers = torch.stack(
        [normalize_vector(server_vectors[:, i]) for i in range(server_vectors.shape[1])]
    ).T
    # print("  - Server data normalized")

    # Step 2: Computing initial h_S_0 (Layer 0)
    # print("\nStep 2: Computing initial h_S_0 (Layer 0)")
    # WS = torch.rand(4, 128) * 2 - 1  # 4×128 random matrix between -1 and 1
    WS = WN
    h_S_layers = [norm_servers @ WS]  # h_S_0: (20 × 128)
    # print("  - WS matrix created and multiplied with normalized servers")
    for i in range(num_servers):
        formatted_h = [f"{x:.2f}" for x in h_S_layers[0][i][:5]]
        print(f"  - h_S_0[Server {i + 1}]: {formatted_h}...")

    # Step 3: Reading and normalizing infraConnections
    # print("\nStep 3: Reading and normalizing infraConnections")
    with open(
        f"data/processed/newInfraConnections_{x}.json", "r", encoding="utf-8"
    ) as f:
        infra_data = json.load(f)  # 20×20 matrix

    bandwidth = torch.zeros((num_servers, num_servers))
    delay = torch.zeros((num_servers, num_servers))
    for i in range(num_servers):
        for j in range(num_servers):
            if infra_data[i][j] != [0, 0]:  # Skip self-connections
                bandwidth[i][j] = infra_data[i][j][0]
                delay[i][j] = infra_data[i][j][1]
            else:
                bandwidth[i][j] = float("-inf")  # No connection
                delay[i][j] = float("inf")  # No connection

    finite_bandwidth = bandwidth[torch.isfinite(bandwidth)]
    finite_delay = delay[torch.isfinite(delay)]
    if len(finite_bandwidth) > 0 and len(finite_delay) > 0:
        norm_bandwidth = bandwidth.clone()
        norm_delay = delay.clone()
        norm_bandwidth[torch.isfinite(bandwidth)] = normalize_vector(finite_bandwidth)
        norm_delay[torch.isfinite(delay)] = normalize_vector(finite_delay)
    else:
        norm_bandwidth = torch.full_like(bandwidth, 0.1)
        norm_delay = torch.full_like(delay, 0.1)
    # print("  - Bandwidth and Delay data normalized")

    # WES = torch.rand(2, 128) * 2 - 1  # 2×128 random matrix
    WES = WNE
    # print("  - WES matrix created (2×128 for bandwidth and delay)")

    e_S_dict = {}
    for i in range(num_servers):
        for j in range(i + 1, num_servers):
            if bandwidth[i][j] != float("-inf"):
                edge_key = f"Server {i + 1} <-> Server {j + 1}"
                edge_features = torch.tensor(
                    [norm_bandwidth[i][j], norm_delay[i][j]], dtype=torch.float32
                )
                e_S_dict[edge_key] = edge_features @ WES
    # print("  - Edge representations (e_S) computed")

    # Step 4: Computing sigmoid of e_S
    print("\nStep 4: Computing sigmoid of e_S")
    sigmoid_e_S = {key: sigmoid(val) for key, val in e_S_dict.items()}
    for key in list(sigmoid_e_S.keys())[:3]:
        formatted_e = [f"{float(x):.2f}" for x in sigmoid_e_S[key][:5]]
    # print(f"  - sigmoid(e_S[{key}]): {formatted_e}...")
    # print("  - (Only first 3 edges shown for brevity)")

    # Step 5: Initializing MS and NS matrices
    # print("\nStep 5: Initializing MS and NS matrices (used for all layers)")
    # MS = torch.rand(128, 128) * 2 - 1  # 128×128 random matrix
    # NS = torch.rand(128, 128) * 2 - 1  # 128×128 random matrix
    MS = UN
    NS = VN
    # print("  - MS and NS matrices initialized")

    # Step 6: Server Message Passing (4 layers)
    num_layers = 4
    # print(
    #   f"\nStep 6: Starting iterations for {num_layers} layers (Server Message Passing)"
    # )
    for layer in range(num_layers):
        # print(f"\n  Layer {layer} -> Layer {layer + 1}:")

        h_S_current = h_S_layers[layer]  # Shape: (20, 128)
        MS_h_S = h_S_current @ MS  # Shape: (20, 128)
        NS_h_S = h_S_current @ NS  # Shape: (20, 128)
        # print(f"    - MS_h_S and NS_h_S computed for Layer {layer}")

        h_S_next = h_S_current.clone()  # Shape: (20, 128)

        for i in range(num_servers):
            # print(f"    - Processing Server {i + 1}:")
            neighbors = [
                j
                for j in range(num_servers)
                if bandwidth[i][j] != float("-inf") and i != j
            ]
            # print(f"      - Neighbors: {[f'S{j + 1}' for j in neighbors]}")

            neighbor_sum = torch.zeros((1, dim))
            for j in neighbors:
                edge_key = (
                    f"Server {i + 1} <-> Server {j + 1}"
                    if i < j
                    else f"Server {j + 1} <-> Server {i + 1}"
                )
                if edge_key in sigmoid_e_S:
                    sig_e_ij = sigmoid_e_S[edge_key]  # Shape: (1, dim)
                    ns_h_j = NS_h_S[j]  # Shape: (dim,)
                    neighbor_sum += sig_e_ij * ns_h_j  # Shape: (1, dim)
            # print(f"      - Neighbor aggregation computed")

            ms_h_i = MS_h_S[i]  # Shape: (dim,)
            aggr_result = ms_h_i + neighbor_sum.flatten()  # Shape: (dim,)
            norm_aggr = normalize_vector(aggr_result)  # Shape: (dim,)
            relu_result = relu(norm_aggr)  # Shape: (dim,)
            # print(f"      - MS_h_S + aggregation normalized and ReLU applied")

            h_S_next[i] = h_S_current[i] + relu_result
            formatted_h = [f"{x:.2f}" for x in h_S_next[i][:5]]
            # print(f"      - h_S_{layer + 1}[Server {i + 1}]: {formatted_h}...")

        h_S_layers.append(h_S_next)

    # Step 7: Edge Message Passing (4 layers)
    # print(f"\nStep 7: Starting edge message passing for {num_layers} layers")
    # XS = torch.rand(128, 128) * 2 - 1  # 128×128 random matrix
    # YS = torch.rand(128, 128) * 2 - 1  # 128×128 random matrix
    # ZS = torch.rand(128, 128) * 2 - 1  # 128×128 random matrix
    XS = AN
    YS = BN
    ZS = CN
    # print("  - X, Y, Z matrices initialized")

    e_S_layers = [e_S_dict]  # Start with initial e_S

    for layer in range(num_layers):
        # print(f"\n  Layer {layer} -> Layer {layer + 1} (Edge Message Passing):")
        e_S_current = e_S_layers[layer]
        h_S_current = h_S_layers[layer]
        e_S_next = e_S_current.copy()

        for edge_key in e_S_current.keys():
            # print(f"    - Processing Edge {edge_key}:")
            s1, s2 = map(lambda x: int(x.split(" ")[1]), edge_key.split(" <-> "))
            i, j = s1 - 1, s2 - 1  # 0-based indices

            e_ij = e_S_current[edge_key]  # Shape: (1, 128)
            h_i = h_S_current[i]  # Shape: (128,)
            h_j = h_S_current[j]  # Shape: (128,)

            XS_e_ij = e_ij @ XS  # Shape: (1, 128)
            YS_h_i = h_i @ YS  # Shape: (128,)
            ZS_h_j = h_j @ ZS  # Shape: (128,)

            aggr = XS_e_ij.flatten() + YS_h_i + ZS_h_j  # Shape: (128,)
            norm_aggr = normalize_vector(aggr)  # Shape: (128,)
            relu_result = relu(norm_aggr)  # Shape: (128,)
            # print(
            #   f"      - XS*e_ij + YS*h_i + ZS*h_j computed, normalized, and ReLU applied"
            # )

            e_S_next[edge_key] = e_ij + relu_result  # Shape: (1, 128)
            formatted_e = [f"{x:.2f}" for x in e_S_next[edge_key][:5]]
            # print(f"      - e_S_{layer + 1}[{edge_key}]: {formatted_e}...")

        e_S_layers.append(e_S_next)

    # Step 8: Save final layers
    final_h_S = h_S_layers[-1]
    final_h_S_rounded = torch.round(final_h_S * 100) / 100  # Round to 2 decimals
    final_e_S = e_S_layers[-1]
    final_e_S_rounded = {
        key: torch.round(val * 100) / 100 for key, val in final_e_S.items()
    }

    with open(f"data/processed/msg_server_{x}.json", "w", encoding="utf-8") as f:
        json.dump(final_h_S_rounded.tolist(), f, indent=4)
    with open(f"data/processed/msg_Sedge_{x}.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in final_e_S_rounded.items()}, f, indent=4)
    # print(f"\nStep 8: Final h_S (Layer 4) saved to 'msg_server1.json'")
    # print(f"Step 8: Final e_S (Layer 4) saved to 'msg_Sedge1.json'")


#   print(f"\nFinal h_S (Layer 4):")
#  for i, h in enumerate(final_h_S_rounded):
#     formatted_h = [f"{x:.2f}" for x in h[:5]]
#    print(f"  h_S_4[Server {i + 1}]: {formatted_h}...")

#   print(f"\nFinal e_S (Layer 4):")
#  for edge_key, e in final_e_S_rounded.items():
#     formatted_e = [f"{x:.2f}" for x in e[:5]]
#    print(f"  e_S_4[{edge_key}]: {formatted_e}...")


# Run the program
# if __name__ == "__main__":
#   msg_server_processing()
