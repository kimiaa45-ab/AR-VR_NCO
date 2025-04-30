import torch
import json
import os
import torch
import yaml
import copy

# import numpy as np


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


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def normalize_vector(v, min_val=0.1, max_val=1.0):
    v_min, v_max = torch.min(v), torch.max(v)
    if v_max == v_min:
        return torch.full_like(v, min_val)
    return min_val + (max_val - min_val) * (v - v_min) / (v_max - v_min)


# Modify the componentConnections matrix
def modify_connections(matrix):
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise ValueError("Input must be a non-empty list of lists")
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("Input must be a square matrix")
    if not all(all(val in [0, 1] for val in row) for row in matrix):
        raise ValueError("Matrix must contain only 0 or 1")

    modified_matrix = copy.deepcopy(matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                modified_matrix[i][j] = 0
                continue
            if matrix[i][j] == 1:
                continue
            weight = 0
            for k in range(j + 1, n):
                if matrix[i][k] == 1:
                    distance = k - j
                    weight = 0.5**distance
                    if weight < 0.1:
                        weight = 0.1
                    break
            modified_matrix[i][j] = weight if weight > 0 else matrix[i][j]

    return modified_matrix


def matrix_to_multi_service_dict(
    matrix, num_services=charnum_service, component_names=None
):
    """
    Convert a connection matrix to a dictionary with multiple services, each containing edge weights.
    Edges are dynamically generated based on matrix size, including only weights 0.5 or 1.

    Args:
        matrix (list of lists): Square matrix where matrix[i][j] is the weight between components i and j.
        num_services (int, optional): Number of services to generate (default: 15).
        component_names (list, optional): List of component names. If None, uses 'c1', 'c2', etc.

    Returns:
        dict: Dictionary with service names (e.g., 'Service1') as keys and edge dictionaries as values.
              Each edge dictionary contains edges like 'e_cX_cY' with weights 0.5 or 1.

    Raises:
        ValueError: If matrix is not a valid square matrix or component_names length mismatches.
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise ValueError("Input must be a non-empty list of lists")
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("Input must be a square matrix")

    # Generate default component names if none provided
    if component_names is None:
        component_names = [f"c{i+1}" for i in range(n)]
    else:
        if len(component_names) != n:
            raise ValueError("Length of component_names must match matrix size")

    # Create edge dictionary for one service
    edge_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] in [
                0.5,
                1.0,
            ]:  # Only include weights 0.5 or 1, skip self-connections
                key = f"e_c{component_names[i]}_c{component_names[j]}"
                edge_dict[key] = float(matrix[i][j])

    # Replicate edge dictionary for each service
    result = {}
    for s in range(1, num_services + 1):
        service_name = f"Service{s}"
        result[service_name] = edge_dict.copy()

    return result


def msg_component_processing(
    X,
    parameters,
    device="cpu",
    settings_path="configs/setting.json",
):

    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    US, VS, WS, WSE, AS, BS, CS, DS = (
        parameters["US"],
        parameters["VS"],
        parameters["WS"],
        parameters["WSE"],
        parameters["AS"],
        parameters["BS"],
        parameters["CS"],
        parameters["DS"],
    )

    # print("Step 1: Reading and normalizing services.json")
    with open(f"data/processed/services_{X}.json", "r", encoding="utf-8") as f:
        component_data = json.load(f)
    # print(f"  - Number of services: {len(component_data)}")

    app_h_C_final = {}
    app_e_C_final = {}

    for service in component_data:
        service_name = f"Service{service['serviceID']}"
        components = service["components"]
        user_id = service["userID"]
        helper_id = service["helperID"]
        #  print(
        #      f"\nProcessing Service: {service_name} (User: {user_id}, Helper: {helper_id})"
        #  )
        num_components = len(components)
        # print(f"  - Number of components: {num_components}")

        # ساخت بردار ویژگی‌ها
        component_vectors = torch.tensor(
            [
                # [c["cpu"], c["memory"], c["dataSize"], c["disk"], c["reliabilityScore"]]
                [
                    c["characteristics"]["cpu"],
                    c["characteristics"]["memory"],
                    c["characteristics"]["dataSize"],
                    c["characteristics"]["disk"],
                    c["characteristics"]["reliabilityScore"],
                ]
                for c in components
            ],
            dtype=torch.float32,
        )
        norm_components = torch.stack(
            [
                normalize_vector(component_vectors[:, i])
                for i in range(component_vectors.shape[1])
            ]
        ).T

        # محاسبه h_C_0
        # WC = torch.rand(5, 128) * 2 - 1  # Random matrix between -1 and 1
        WC = WS
        h_C_layers = [norm_components @ WC]

        ##
        # Inside msg_component_processing, within the service loop
        # Load and process dependencies
        with open(
            f"data/processed/newcomponentsConnections_{X}.json", "r", encoding="utf-8"
        ) as f:
            full_matrix = json.load(f)

        # Validate matrix
        if not full_matrix or not all(isinstance(row, list) for row in full_matrix):
            raise ValueError("Dependency data must be a non-empty list of lists")
        n = len(full_matrix)
        if not all(len(row) == n for row in full_matrix):
            raise ValueError(
                f"Dependency data must be a square matrix, got {n}x{len(full_matrix[0])}"
            )
        if n < charnum_component:
            raise ValueError(
                f"Matrix size ({n}) is smaller than required ({charnum_component})"
            )

        # Truncate matrix to charnum_component x charnum_component
        dependency_data = [
            row[:charnum_component] for row in full_matrix[:charnum_component]
        ]
        a = len(dependency_data)

        # Validate matrix size
        if (a) != num_components:
            raise ValueError(
                f"Matrix size ({len(dependency_data)}) does not match number of components ({num_components})"
            )

        # Modify the matrix (if not already modified)
        modified_connections = dependency_data  # Use directly if already modified
        # If you need to apply modify_connections, uncomment:
        # modified_connections = modify_connections(dependency_data)

        # Convert to service dictionary
        component_names = [
            c.get("componentID", f"c{i+1}") for i, c in enumerate(components)
        ]
        service_dict = matrix_to_multi_service_dict(
            modified_connections, num_services=15, component_names=component_names
        )

        # Extract app_deps for the current service
        app_deps = service_dict.get(service_name, {})

        # Process dependencies
        dep_values = torch.tensor(
            [val for val in app_deps.values() if val > 0], dtype=torch.float32
        )
        norm_deps = (
            normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)
        )
        ###
        # # بارگذاری وابستگی‌ها
        # with open(
        #    f"data/processed/newcomponentsConnections_{X}.json", "r", encoding="utf-8"
        # ) as f:
        #    dependency_data = json.load(f)
        #    modified_connections = modify_connections(dependency_data)
        #   print(modified_connections)
        #  app_deps = modified_connections.get(service_name, {})
        # dep_values = torch.tensor(
        #    [val for val in app_deps.values() if val > 0], dtype=torch.float32
        # )
        # norm_deps = (
        #   normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)
        # )
        # print(dep_values)
        # Load dependencies
        # with open(
        #    f"data/processed/newcomponentsConnections_{X}.json", "r", encoding="utf-8"
        # ) as f:
        #   dependency_data = json.load(f)

        # dependency_data is a matrix (list of lists)
        # modified_connections = modify_connections(dependency_data)

        # Convert matrix to dictionary
        # component_names = [
        #  f"c{i+1}" for i in range(len(dependency_data))
        # ]  # Names: c1, c2, ..., c6
        # app_deps = matrix_to_dict(modified_connections, component_names=component_names)

        # Process dependencies
        # dep_values = torch.tensor(
        #  [val for val in app_deps.values() if val > 0], dtype=torch.float32
        # )
        # norm_deps = (
        # normalize_vector(dep_values) if len(dep_values) > 0 else torch.zeros(1)
        # )

        # WEC = torch.rand(128) * 2 - 1  # Random vector between -1 and 1
        WEC = WSE
        e_C_dict = {}
        idx = 0
        for edge_key, val in app_deps.items():
            if val > 0:
                e_C_dict[edge_key] = norm_deps[idx] * WEC
                idx += 1
        sigmoid_e_C = {key: sigmoid(val) for key, val in e_C_dict.items()}

        # MC = torch.rand(128, 128) * 2 - 1
        # NC = torch.rand(128, 128) * 2 - 1
        MC = US
        NC = VS

        # پیام‌رسانی کامپوننت‌ها
        for layer in range(num_components - 1):
            h_C_current = h_C_layers[layer]
            MC_h_C = h_C_current @ MC
            NC_h_C = h_C_current @ NC
            h_C_next = h_C_current.clone()

            for i in range(num_components):
                neighbors = set()
                for edge_key in e_C_dict.keys():
                    c1, c2 = map(int, edge_key.split("_c")[1:])
                    if c1 == i + 1:
                        neighbors.add(c2 - 1)
                    elif c2 == i + 1:
                        neighbors.add(c1 - 1)
                neighbors = list(neighbors)

                neighbor_sum = torch.zeros(1, dim)
                for j in neighbors:
                    edge_key = (
                        f"e_c{i + 1}_c{j + 1}" if i < j else f"e_c{j + 1}_c{i + 1}"
                    )
                    if edge_key in sigmoid_e_C:
                        sig_e_ij = sigmoid_e_C[edge_key]
                        nc_h_j = NC_h_C[j]
                        neighbor_sum += sig_e_ij * nc_h_j

                mc_h_i = MC_h_C[i]
                aggr_result = mc_h_i + neighbor_sum
                norm_aggr = normalize_vector(aggr_result)
                relu_result = relu(norm_aggr)
                h_C_next[i] = h_C_current[i] + relu_result

            h_C_layers.append(h_C_next)

        # پیام‌رسانی لبه‌ها
        # X = torch.rand(128, 128) * 2 - 1
        # Y = torch.rand(128, 128) * 2 - 1
        # Z = torch.rand(128, 128) * 2 - 1
        X1 = AS
        Y = BS
        Z = CS
        X2 = DS
        e_C_layers = [e_C_dict]

        # for layer in range(num_components - 1):
        #    e_C_current = e_C_layers[layer]
        #     h_C_current = h_C_layers[layer]
        #      e_C_next = e_C_current.copy()
        #       for edge_key in e_C_current.keys():
        #            c1, c2 = map(int, edge_key.split("_c")[1:])
        # i, j = c1 - 1, c2 - 1
        # e_ij = e_C_current[edge_key]
        #  e_ji = e_C_current[edge_key]
        #   h_i, h_j = h_C_current[i], h_C_current[j]
        #    X_e_ij = e_ij @ X
        #     #X_e_ji = e_ji @ X1
        # Y_h_i = h_i @ Y
        #  Z_h_j = h_j @ Z
        #   aggr = X_e_ij + X_e_ji+ Y_h_i + Z_h_j
        #    norm_aggr = normalize_vector(aggr)
        #     relu_result = relu(norm_aggr)
        #      e_C_next[edge_key] = e_ij + relu_result
        #   e_C_layers.append(e_C_next)

        for layer in range(num_layers):
            e_C_current = e_C_layers[layer]
            h_C_current = h_C_layers[layer]
            e_C_next = e_C_current.copy()

            for edge_key in e_C_current.keys():
                c1, c2 = map(int, edge_key.split("_c")[1:])
                i, j = c1 - 1, c2 - 1

                e_ij = e_C_current[edge_key]

                reverse_key = f"c{c2}_c{c1}"
                e_ji = e_C_current.get(reverse_key, torch.zeros_like(e_ij))

                h_i, h_j = h_C_current[i], h_C_current[j]

                X_e_ij = e_ij @ X1
                X_e_ji = e_ji @ X2
                Y_h_i = h_i @ Y
                Z_h_j = h_j @ Z

                aggr = X_e_ij + X_e_ji + Y_h_i + Z_h_j
                norm_aggr = normalize_vector(aggr)
                relu_result = relu(norm_aggr)
                e_C_next[edge_key] = e_ij + relu_result

            e_C_layers.append(e_C_next)

        # ذخیره لایه آخر
        final_h_C_rounded = (
            torch.round(h_C_layers[-1] * 100) / 100
        )  # Round to 2 decimals
        final_e_C_rounded = {
            key: torch.round(val * 100) / 100 for key, val in e_C_layers[-1].items()
        }

        app_h_C_final[service_name] = final_h_C_rounded.tolist()
        app_e_C_final[service_name] = {
            key: val.tolist() for key, val in final_e_C_rounded.items()
        }

        # نمایش لایه آخر
    # print(
    #    f"\nFinal Layer Results for {service_name} (User: {user_id}, Helper: {helper_id}):"
    #  )
    #        for i, h in enumerate(app_h_C_final[service_name]):
    #           formatted_h = [f"{x:.2f}" for x in (list(h) + [0] * charnum_s)[:charnum_s]]
    #          print(f"  h_C_final[Component {i + 1}]: {formatted_h}...")
    #     for edge_key, e in list(app_e_C_final[service_name].items())[:10]:
    #        formatted_e = [
    # f"{float(x):.2f}" for x in (list(e) + [0] * charnum_s)[:charnum_s]
    # f"{float(x):.2f}"for x in (e + [0] * charnum_s)[:charnum_s]
    #           f"{float(x):.2f}"
    #          for x in (sum(e, []) + [0] * charnum_s)[:charnum_s]
    #    ]
    #     print(f"  e_C_final[{edge_key}]: {formatted_e}...")
    # for i, h in enumerate(app_h_C_final[service_name]):
    #     formatted_h = [f"{x:.2f}" for x in h[:charnum_component]]
    #    print(f"  h_C_final[Component {i + 1}]: {formatted_h}...")
    # for edge_key, e in list(app_e_C_final[service_name].items())[:10]:
    #    formatted_e = [f"{x:.2f}" for x in e[:charnum_component]]
    #    print(f"  e_C_final[{edge_key}]: {formatted_e}...")

    # ذخیره نتایج
    with open(f"data/processed/msg_component_{X}.json", "w", encoding="utf-8") as f:
        json.dump(app_h_C_final, f, indent=4)
    with open(f"data/processed/msg_Cedge_{X}.json", "w", encoding="utf-8") as f:
        json.dump(app_e_C_final, f, indent=4)
    print("\nFinal embeddings saved for all services")


# if __name__ == "__main__":
#   msg_component_processing()
