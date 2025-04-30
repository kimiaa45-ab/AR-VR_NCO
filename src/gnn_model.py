import torch
import json
import os
import yaml
import numpy as np


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min:
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


def save_gnn_outputs(
    batch_idx,
    microservices_embedding,
    microservices_edges_embedding,
    computingnodes_embedding,
    computingnodes_edges_embedding,
    output_dir="data/processed",
):
    """
    Save GNN outputs to JSON files in the specified directory.

    Args:
        batch_idx (int): Index of the batch (e.g., 1 for batch 1-32, 2 for batch 33-64).
        microservices_embedding (list): List of microservices embeddings for the batch.
        microservices_edges_embedding (list): List of microservices edges embeddings.
        computingnodes_embedding (list): List of computing nodes embeddings.
        computingnodes_edges_embedding (list): List of computing nodes edges embeddings.
        output_dir (str): Directory to save JSON files (default: 'data/processed').
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays and tensors to lists for JSON serialization
    def to_serializable(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: to_serializable(value) for key, value in data.items()}
        return data

    # Define file paths
    files = {
        "microservices_embedding": f"{output_dir}/microservices_embedding_batch_{batch_idx}.json",
        "microservices_edges_embedding": f"{output_dir}/microservices_edges_embedding_batch_{batch_idx}.json",
        "computingnodes_embedding": f"{output_dir}/computingnodes_embedding_batch_{batch_idx}.json",
        "computingnodes_edges_embedding": f"{output_dir}/computingnodes_edges_embedding_batch_{batch_idx}.json",
    }

    # Save each output to its corresponding JSON file
    for name, file_path in files.items():
        data = locals()[name]  # Get the data by name
        serialized_data = to_serializable(data)
        with open(file_path, "w") as f:
            json.dump(serialized_data, f, indent=4)
        print(f"Saved {name} to {file_path}")


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


# Convert infraconnection to the number of computing nodes
def truncate_computingnodes_edges(computingnodes_edges, charnum_node, device):
    truncated_edges = []
    for idx, edges in enumerate(computingnodes_edges):
        edges_tensor = (
            torch.tensor(edges, dtype=torch.float32, device=device)
            if not isinstance(edges, torch.Tensor)
            else edges.to(device)
        )
        if (
            edges_tensor.ndim != 3
            or edges_tensor.shape[0] < charnum_node
            or edges_tensor.shape[1] < charnum_node
        ):
            print(
                f"Warning: Instance {idx}: Invalid shape {edges_tensor.shape} for charnum_node {charnum_node}"
            )
            truncated_edges.append(edges_tensor.tolist())
            continue
        truncated_tensor = edges_tensor[:charnum_node, :charnum_node, :]
        truncated_edges.append(truncated_tensor.tolist())
    return truncated_edges


def gnn_model(
    microservices,
    microservices_edges,
    computingnodes,
    computingnodes_edges,
    parameters,
    batch_idx=1,
    device="cpu",
    settings_path="configs/setting.json",
):
    """
    Custom GNN model to process microservices and their edges, producing embeddings and saving parameters.

    Args:
        microservices (list): List of microservice data for dim samples.
        microservices_edges (list): List of componentConnections matrices.
        computingnodes (list): List of computing nodes.
        computingnodes_edges (list): List of infraConnections.
        parameters (dict): Parameters (US, UN, WS, WN, WSE, WNE, AS, BS, CS, DS, AN, BN, CN).
        batch_idx (int): Index of the batch for saving outputs.
        device (str): Device to run computations ('cuda' or 'cpu').
        settings_path (str): Path to save settings.json.

    Returns:
        tuple: (microservices_embedding, microservices_edges_embedding, computingnodes_embedding,
                computingnodes_edges_embedding, updated_parameters)
    """
    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    US, VS, WS, WSE, AS, BS, CS, DS, UN, VN, WN, WNE, AN, BN, CN = (
        parameters["US"],
        parameters["VS"],
        parameters["WS"],
        parameters["WSE"],
        parameters["AS"],
        parameters["BS"],
        parameters["CS"],
        parameters["DS"],
        parameters["UN"],
        parameters["VN"],
        parameters["WN"],
        parameters["WNE"],
        parameters["AN"],
        parameters["BN"],
        parameters["CN"],
    )

    # Initialize output lists
    microservices_embedding = []
    microservices_edges_embedding = []
    computingnodes_embedding = []
    computingnodes_edges_embedding = []

    # Process microservices
    computingnodes_edges = truncate_computingnodes_edges(
        computingnodes_edges, charnum_node, device
    )
    for microservice, connections in zip(microservices, microservices_edges):
        # Extract services
        component_data = []
        for service in microservice.get("services", []):
            service_data = {
                "serviceID": service.get("serviceID", 0),
                "components": [
                    {
                        "cpu": comp["characteristics"]["cpu"],
                        "memory": comp["characteristics"]["memory"],
                        "dataSize": comp["characteristics"]["dataSize"],
                        "disk": comp["characteristics"]["disk"],
                        "reliabilityScore": comp["characteristics"]["reliabilityScore"],
                    }
                    for comp in service.get("components", [])
                ],
                "userID": service.get(
                    "userID", microservice.get("usersNodes", [{}])[0].get("nodeID", 0)
                ),
                "helperID": service.get(
                    "helperID",
                    microservice.get("helperNodes", [{}])[0].get("nodeID", 0),
                ),
            }
            component_data.append(service_data)

        # Extract and convert componentConnections to dictionary
        dependency_data = {}
        for service in component_data:
            service_name = f"Service{service['serviceID']}"
            app_deps = {}
            num_components = len(service["components"])
            if (
                connections
                and isinstance(connections, list)
                and all(isinstance(row, list) for row in connections)
            ):
                try:
                    for i in range(len(connections)):
                        for j in range(len(connections[i])):
                            if (
                                connections[i][j] > 0
                                and i < num_components
                                and j < num_components
                            ):
                                edge_key = (
                                    f"e_c{i+1}_c{j+1}" if i < j else f"e_c{j+1}_c{i+1}"
                                )
                                app_deps[edge_key] = float(connections[i][j])
                except (TypeError, IndexError) as e:
                    print(
                        f"Warning: Invalid componentConnections format for {service_name}: {e}. Skipping."
                    )
            else:
                print(
                    f"Warning: componentConnections is not a valid matrix for {service_name}. Skipping."
                )
            dependency_data[service_name] = app_deps

        app_h_C_final = {}
        app_e_C_final = {}

        for service in component_data:
            service_name = f"Service{service['serviceID']}"
            components = service["components"]
            user_id = service["userID"]
            helper_id = service["helperID"]
            num_components = len(components)

            # Build feature vectors
            component_vectors = torch.tensor(
                [
                    [
                        c["cpu"],
                        c["memory"],
                        c["dataSize"],
                        c["disk"],
                        c["reliabilityScore"],
                    ]
                    for c in components
                ],
                dtype=torch.float32,
            ).to(device)
            norm_components = torch.stack(
                [
                    normalize_vector(component_vectors[:, i])
                    for i in range(component_vectors.shape[1])
                ]
            ).T
            norm_components = torch.round(norm_components * 100) / 100

            # Compute h_C_0
            h_C_layers = [norm_components @ WS]

            # Process dependencies
            app_deps = dependency_data.get(service_name, {})
            dep_values = torch.tensor(
                [val for val in app_deps.values() if val > 0], dtype=torch.float32
            ).to(device)
            norm_deps = (
                normalize_vector(dep_values)
                if len(dep_values) > 0
                else torch.zeros(1, device=device)
            )
            e_C_dict = {}
            idx = 0
            for edge_key, val in app_deps.items():
                if val > 0:
                    e_C_dict[edge_key] = norm_deps[idx]
                    idx += 1
            sigmoid_e_C = {key: sigmoid(val) for key, val in e_C_dict.items()}

            # Component message passing
            for layer in range(num_layers - 1):
                h_C_current = h_C_layers[layer]
                US_h_C = h_C_current @ US
                US_h_C = torch.round(US_h_C * 100) / 100
                VS_h_C = h_C_current @ VS
                VS_h_C = torch.round(VS_h_C * 100) / 100
                h_C_next = h_C_current.clone()
                h_C_next = torch.round(h_C_next * 100) / 100

                for i in range(num_components):
                    neighbors = set()
                    for edge_key in e_C_dict.keys():
                        c1, c2 = map(int, edge_key.split("_c")[1:])
                        if c1 == i + 1:
                            neighbors.add(c2 - 1)
                        elif c2 == i + 1:
                            neighbors.add(c1 - 1)

                    neighbor_sum = torch.zeros(h_C_current.shape[1], device=device)
                    for j in neighbors:
                        if j >= num_components:
                            continue
                        edge_key = f"e_c{i+1}_c{j+1}" if i < j else f"e_c{j+1}_c{i+1}"
                        if edge_key in sigmoid_e_C:
                            sig_e_ij = sigmoid_e_C[edge_key]
                            sig_e_ij = torch.round(sig_e_ij * 100) / 100
                            nc_h_j = VS_h_C[j]
                            neighbor_sum += sig_e_ij * nc_h_j
                            neighbor_sum = torch.round(neighbor_sum * 100) / 100

                    us_h_i = US_h_C[i]
                    aggr_result = us_h_i + neighbor_sum
                    norm_aggr = normalize_vector(aggr_result)
                    norm_aggr = torch.round(norm_aggr * 100) / 100
                    relu_result = relu(norm_aggr)
                    relu_result = torch.round(relu_result * 100) / 100
                    h_C_next[i] = h_C_current[i] + relu_result

                h_C_layers.append(h_C_next)

            # Edge message passing
            e_C_layers = [dict(e_C_dict)]
            for layer in range(num_components - 1):
                e_C_current = e_C_layers[layer]
                h_C_current = h_C_layers[layer]
                e_C_next = e_C_current.copy()
                for edge_key in e_C_current.keys():
                    c1, c2 = map(int, edge_key.split("_c")[1:])
                    i, j = c1 - 1, c2 - 1
                    if i >= num_components or j >= num_components:
                        continue
                    e_ij = e_C_current[edge_key] * WSE
                    h_i, h_j = h_C_current[i], h_C_current[j]
                    AS_e_ij = e_ij @ AS
                    AS_e_ij = torch.round(AS_e_ij * 100) / 100
                    DS_e_ij = e_ij @ DS
                    DS_e_ij = torch.round(DS_e_ij * 100) / 100
                    BS_h_i = h_i @ BS
                    BS_h_i = torch.round(BS_h_i * 100) / 100
                    CS_h_j = h_j @ CS
                    CS_h_j = torch.round(CS_h_j * 100) / 100
                    aggr = AS_e_ij + DS_e_ij + BS_h_i + CS_h_j
                    aggr = torch.round(aggr * 100) / 100
                    norm_aggr = normalize_vector(aggr)
                    norm_aggr = torch.round(norm_aggr * 100) / 100
                    relu_result = relu(norm_aggr)
                    relu_result = torch.round(relu_result * 100) / 100
                    e_C_next[edge_key] = e_C_current[edge_key] + torch.mean(relu_result)
                e_C_layers.append(e_C_next)

            # Store final layer
            final_h_C_rounded = torch.round(h_C_layers[-1] * 100) / 100
            final_e_C_rounded = {
                key: round(float(val), 2) for key, val in e_C_layers[-1].items()
            }

            app_h_C_final[service_name] = final_h_C_rounded.tolist()
            app_e_C_final[service_name] = final_e_C_rounded

        microservices_embedding.append(app_h_C_final)
        microservices_edges_embedding.append(app_e_C_final)

    # Process computing nodes
    for nodes, infra_connections in zip(computingnodes, computingnodes_edges):
        # Extract nodes
        node_data = []
        for server in nodes:
            server_data = {
                "nodeID": server.get("nodeID", 0),
                "nodeTier": server.get("nodeTier", 0),
                "cpu": server["characteristics"]["cpu"],
                "memory": server["characteristics"]["memory"],
                "disk": server["characteristics"]["disk"],
                "reliabilityScore": server["characteristics"]["reliabilityScore"],
            }
            node_data.append(server_data)

        # Extract and convert infraConnections to dictionary
        dependency_data = {}
        for node in node_data:
            node_name = f"Node{node['nodeID']}"
            app_deps = {}
            num_nodes = len(node_data)
            if (
                infra_connections
                and isinstance(infra_connections, list)
                and all(isinstance(row, list) for row in infra_connections)
            ):
                try:
                    for i in range(len(infra_connections)):
                        for j in range(len(infra_connections[i])):
                            if (
                                infra_connections[i][j][0] > 0
                                and i < num_nodes
                                and j < num_nodes
                            ):
                                edge_key = (
                                    f"e_n{i+1}_n{j+1}" if i < j else f"e_n{j+1}_n{i+1}"
                                )
                                app_deps[edge_key] = float(infra_connections[i][j][0])
                except (TypeError, IndexError) as e:
                    print(
                        f"Warning: Invalid infraConnections format for {node_name}: {e}. Skipping."
                    )
            else:
                print(
                    f"Warning: infraConnections is not a valid matrix for {node_name}. Skipping."
                )
            dependency_data[node_name] = app_deps

        app_h_N_final = {}
        app_e_N_final = {}

        for node in node_data:
            node_name = f"Node{node['nodeID']}"
            num_nodes = len(node_data)

            # Build feature vectors
            node_vectors = torch.tensor(
                [
                    [
                        node["cpu"],
                        node["memory"],
                        node["disk"],
                        node["reliabilityScore"],
                    ]
                    for node in node_data
                ],
                dtype=torch.float32,
            ).to(device)
            norm_nodes = torch.stack(
                [
                    normalize_vector(node_vectors[:, i])
                    for i in range(node_vectors.shape[1])
                ]
            ).T
            norm_nodes = torch.round(norm_nodes * 100) / 100

            # Compute h_N_0
            h_N_layers = [norm_nodes @ WN]

            # Process dependencies
            app_deps = dependency_data.get(node_name, {})
            dep_values = torch.tensor(
                [val for val in app_deps.values() if val > 0], dtype=torch.float32
            ).to(device)
            norm_deps = (
                normalize_vector(dep_values)
                if len(dep_values) > 0
                else torch.zeros(1, device=device)
            )
            e_N_dict = {}
            idx = 0
            for edge_key, val in app_deps.items():
                if val > 0:
                    e_N_dict[edge_key] = norm_deps[idx]
                    idx += 1
            sigmoid_e_N = {key: sigmoid(val) for key, val in e_N_dict.items()}

            # Node message passing
            for layer in range(num_layers - 1):
                h_N_current = h_N_layers[layer]
                UN_h_N = h_N_current @ UN
                UN_h_N = torch.round(UN_h_N * 100) / 100
                VN_h_N = h_N_current @ VN
                VN_h_N = torch.round(VN_h_N * 100) / 100
                h_N_next = h_N_current.clone()
                h_N_next = torch.round(h_N_next * 100) / 100

                for i in range(num_nodes):
                    neighbors = set()
                    for edge_key in e_N_dict.keys():
                        n1, n2 = map(int, edge_key.split("_n")[1:])
                        if n1 == i + 1:
                            neighbors.add(n2 - 1)
                        elif n2 == i + 1:
                            neighbors.add(n1 - 1)

                    neighbor_sum = torch.zeros(h_N_current.shape[1], device=device)
                    for j in neighbors:
                        if j >= num_nodes:
                            continue
                        edge_key = f"e_n{i+1}_n{j+1}" if i < j else f"e_n{j+1}_n{i+1}"
                        if edge_key in sigmoid_e_N:
                            sig_e_ij = sigmoid_e_N[edge_key]
                            sig_e_ij = torch.round(sig_e_ij * 100) / 100
                            nc_h_j = VN_h_N[j]
                            neighbor_sum += sig_e_ij * nc_h_j
                            neighbor_sum = torch.round(neighbor_sum * 100) / 100

                    un_h_i = UN_h_N[i]
                    aggr_result = un_h_i + neighbor_sum
                    norm_aggr = normalize_vector(aggr_result)
                    norm_aggr = torch.round(norm_aggr * 100) / 100
                    relu_result = relu(norm_aggr)
                    relu_result = torch.round(relu_result * 100) / 100
                    h_N_next[i] = h_N_current[i] + relu_result

                h_N_layers.append(h_N_next)

            # Edge message passing
            e_N_layers = [dict(e_N_dict)]
            for layer in range(num_layers - 1):
                e_N_current = e_N_layers[layer]
                h_N_current = h_N_layers[layer]
                e_N_next = e_N_current.copy()
                for edge_key in e_N_current.keys():
                    n1, n2 = map(int, edge_key.split("_n")[1:])
                    i, j = n1 - 1, n2 - 1
                    if i >= num_nodes or j >= num_nodes:
                        continue
                    e_ij = e_N_current[edge_key] * WNE
                    h_i, h_j = h_N_current[i], h_N_current[j]
                    AN_e_ij = e_ij @ AN
                    AN_e_ij = torch.round(AN_e_ij * 100) / 100
                    DS_e_ij = e_ij @ DS
                    DS_e_ij = torch.round(DS_e_ij * 100) / 100
                    BN_h_i = h_i @ BN
                    BN_h_i = torch.round(BN_h_i * 100) / 100
                    CN_h_j = h_j @ CN
                    CN_h_j = torch.round(CN_h_j * 100) / 100
                    aggr = AN_e_ij + DS_e_ij + BN_h_i + CN_h_j
                    aggr = torch.round(aggr * 100) / 100
                    norm_aggr = normalize_vector(aggr)
                    norm_aggr = torch.round(norm_aggr * 100) / 100
                    relu_result = relu(norm_aggr)
                    relu_result = torch.round(relu_result * 100) / 100
                    e_N_next[edge_key] = e_N_current[edge_key] + torch.mean(relu_result)
                e_N_layers.append(e_N_next)

            # Store final layer
            final_h_N_rounded = torch.round(h_N_layers[-1] * 100) / 100
            final_e_N_rounded = {
                key: round(float(val), 2) for key, val in e_N_layers[-1].items()
            }

            app_h_N_final[node_name] = final_h_N_rounded.tolist()
            app_e_N_final[node_name] = final_e_N_rounded

        computingnodes_embedding.append(app_h_N_final)
        computingnodes_edges_embedding.append(app_e_N_final)

    # Save GNN outputs to JSON files
    save_gnn_outputs(
        batch_idx,
        microservices_embedding,
        microservices_edges_embedding,
        computingnodes_embedding,
        computingnodes_edges_embedding,
    )

    # Save parameters to settings.json
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.cpu().tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    updated_parameters = parameters

    return (
        microservices_embedding,
        microservices_edges_embedding,
        computingnodes_embedding,
        computingnodes_edges_embedding,
        updated_parameters,
    )
