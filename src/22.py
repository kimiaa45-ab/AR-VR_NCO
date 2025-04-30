import torch
import json
import os
import yaml
import copy
import numpy as np

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

# Determine device and set initial dim
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        default_dim = hidden_dim
    else:
        device = "cpu"
        default_dim = cpu_hidden_dim
else:
    device = device.lower()
    default_dim = hidden_dim if device == "cuda" else cpu_hidden_dim


# Convert infraconnection to the number of computing nodes
def truncate_computingnodes_edges(computingnodes_edges, charnum_node, device):
    truncated_edges = []
    for idx, edges in enumerate(computingnodes_edges):
        if isinstance(edges, dict):
            num_nodes = max([int(k.replace("Node", "")) for k in edges.keys()])
            edge_array = np.zeros((num_nodes, num_nodes, charnum_ne))
            for node_i in edges:
                i = int(node_i.replace("Node", "")) - 1
                for edge_key, value in edges[node_i].items():
                    j = int(edge_key.split("_n")[1]) - 1
                    edge_array[i, j, 0] = value
                    edge_array[j, i, 0] = value  # Symmetric
            edges = edge_array
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


def AR_model(
    services,
    nodes,
    connectivity,
    emc,
    ems,
    parameters,
    device="cpu",
    settings_path="configs/setting.json",
):
    """
    AR model to assign service components to nodes for a batch of instances in parallel.

    Args:
        services (list): List of [batch_size] lists of service ID strings.
        dag (list): List of [batch_size] componentConnections dictionaries.
        nodes (list): List of [batch_size] computing nodes data lists.
        connectivity (list): List of [batch_size] infraConnections matrices or dictionaries.
        emc (list): List of [batch_size] microservices embeddings dictionaries.
        Cedge (list): List of [batch_size] microservices edges embeddings.
        ems (list): List of [batch_size] dictionaries with node IDs as keys and embeddings as values.
        Sedge (list): List of [batch_size] computing nodes edges embeddings.
        parameters (dict): Parameters (WSQ, WNK).
        device (str): Device to run computations ('cuda' or 'cpu').
        settings_path (str): Path to save settings.json.

    Returns:
        tuple: (placements, updated_parameters), where placements is a list of [batch_size] placement dictionaries.
    """
    # Move parameters to device
    parameters = {
        k: (
            v.clone().detach().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, dtype=torch.float32, device=device)
        )
        for k, v in parameters.items()
    }
    WSQ, WNK = parameters["WSQ"], parameters["WNK"]

    batch_size = len(services)
    placements = []

    # Detect embedding dimension from ems
    sample_embedding = next(
        (
            ems[0][node_id]
            for node_id in ems[0]
            if isinstance(ems[0][node_id], (list, np.ndarray))
        ),
        [0.0] * default_dim,
    )
    dim = len(np.array(sample_embedding).flatten())
    if dim != default_dim:
        print(
            f"Warning: Using embedding dimension {dim} from ems, expected {default_dim}"
        )

    # Adjust WSK and WCQ to match embedding dimension
    WSK = torch.nn.Parameter(torch.randn(dim, dim, device=device))  # Initialize new WSK
    WCQ = torch.nn.Parameter(torch.randn(dim, dim, device=device))  # Initialize new WCQ
    parameters["WSQ"] = WSK
    parameters["WNK"] = WNK

    # Truncate connectivity
    connectivity = truncate_computingnodes_edges(connectivity, charnum_node, device)
    connectivity_matrix = torch.tensor(
        connectivity, dtype=torch.float32, device=device
    )  # Shape: (batch_size, charnum_node, charnum_node, charnum_ne)

    # Compute server embeddings (KS)
    ems_array = []
    for batch_idx in range(batch_size):
        instance = ems[batch_idx]
        instance_embeddings = []
        for node_dict in nodes[batch_idx]:
            node_id = f"Node{node_dict['nodeID']}"
            if node_id in instance:
                embedding = instance[node_id]
                if isinstance(embedding, (list, np.ndarray)):
                    embedding = np.array(embedding).flatten()
                    if len(embedding) != dim:
                        print(
                            f"Warning: Embedding for {node_id} in instance {batch_idx} has length {len(embedding)}, expected {dim}, padding/truncating"
                        )
                        embedding = np.pad(
                            embedding,
                            (0, max(0, dim - len(embedding))),
                            mode="constant",
                        )[:dim]
                else:
                    print(
                        f"Warning: Invalid embedding type for {node_id} in instance {batch_idx}: {type(embedding)}, using zeros"
                    )
                    embedding = [0.0] * dim
                instance_embeddings.append(embedding)
            else:
                print(
                    f"Warning: Embedding for {node_id} not found in instance {batch_idx}, using zeros"
                )
                instance_embeddings.append([0.0] * dim)
        ems_array.append(instance_embeddings)
    try:
        ems_array_np = np.array(
            ems_array, dtype=np.float32
        )  # Shape: (batch_size, charnum_node, dim)
        ems_tensor = torch.tensor(ems_array_np, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Error creating ems_tensor: {e}")
        print(f"ems_array shape: {[len(instance) for instance in ems_array]}")
        print(f"Sample embedding: {ems_array[0][0][:10]}...")
        raise
    if (
        ems_tensor.ndim != 3
        or ems_tensor.shape[0] != batch_size
        or ems_tensor.shape[1] != charnum_node
    ):
        print(
            f"Error: ems_tensor has shape {ems_tensor.shape}, expected ({batch_size}, {charnum_node}, {dim})"
        )
        raise ValueError("Invalid ems_tensor shape")
    KS = torch.bmm(
        ems_tensor, WSK.expand(batch_size, -1, -1)
    )  # Shape: (batch_size, charnum_node, dim)
    KS_min = torch.min(KS, dim=1, keepdim=True)[0]
    KS_max = torch.max(KS, dim=1, keepdim=True)[0]
    KS = (KS - KS_min) / (KS_max - KS_min + 1e-10)

    # Initialize resources
    initial_resources = [
        {
            str(node["nodeID"]): {
                "cpu": node["characteristics"]["cpu"],
                "memory": node["characteristics"]["memory"],
                "disk": node["characteristics"]["disk"],
            }
            for node in nodes[i]
        }
        for i in range(batch_size)
    ]
    S = [copy.deepcopy(res) for res in initial_resources]

    # Process services and components
    for service_idx in range(charnum_service):
        service_id = f"Service{service_idx + 1}"
        print(f"\nProcessing Service {service_id} for all instances")

        component_embeddings = [emc[i].get(service_id, []) for i in range(batch_size)]

        for comp_idx in range(charnum_component):
            component_id = f"S{service_idx + 1}_C{comp_idx + 1}"
            print(f"  - Processing Component: {component_id}")

            hc_i = torch.stack(
                [
                    (
                        torch.tensor(
                            component_embeddings[i][comp_idx]["embedding"],
                            dtype=torch.float32,
                            device=device,
                        )
                        if comp_idx < len(component_embeddings[i])
                        and "embedding" in component_embeddings[i][comp_idx]
                        else torch.zeros(dim, device=device)
                    )
                    for i in range(batch_size)
                ]
            )  # Shape: (batch_size, dim)

            qc = torch.matmul(hc_i, WCQ)
            qc_min = torch.min(qc, dim=1, keepdim=True)[0]
            qc_max = torch.max(qc, dim=1, keepdim=True)[0]
            qc = (qc - qc_min) / (qc_max - qc_min + 1e-10)

            inits = torch.bmm(qc.unsqueeze(1), KS.transpose(1, 2)).squeeze(1)
            connectivity_factor = torch.mean(connectivity_matrix, dim=(2, 3))
            inits = inits * (1 + connectivity_factor)

            # Check resource constraints using emc for component specs
            for batch_idx in range(batch_size):
                service_found = service_id in services[batch_idx]
                if not service_found or comp_idx >= len(
                    component_embeddings[batch_idx]
                ):
                    inits[batch_idx] = float("-inf")
                    continue
                c_specs = {
                    "cpu": component_embeddings[batch_idx][comp_idx].get(
                        "cpu", np.random.uniform(1, 10)
                    ),
                    "memory": component_embeddings[batch_idx][comp_idx].get(
                        "memory", np.random.uniform(1, 10)
                    ),
                    "disk": component_embeddings[batch_idx][comp_idx].get(
                        "disk", np.random.uniform(1, 10)
                    ),
                }
                for node_idx in range(charnum_node):
                    node_key = str(nodes[batch_idx][node_idx]["nodeID"])
                    if (
                        S[batch_idx][node_key]["cpu"] < c_specs["cpu"]
                        or S[batch_idx][node_key]["memory"] < c_specs["memory"]
                        or S[batch_idx][node_key]["disk"] < c_specs["disk"]
                    ):
                        inits[batch_idx, node_idx] = float("-inf")

            finite_inits = torch.where(
                torch.isinf(inits), torch.tensor(-1e10, device=device), inits
            )
            shifted_inits = (
                finite_inits - torch.max(finite_inits, dim=1, keepdim=True)[0]
            )
            eu = torch.exp(shifted_inits)
            zigma_inits = torch.sum(eu, dim=1, keepdim=True)

            p = torch.where(
                zigma_inits > 0, eu / (zigma_inits + 1e-10), torch.zeros_like(eu)
            )

            invalid_batches = (zigma_inits.squeeze() == 0).nonzero(as_tuple=True)[0]
            for batch_idx in invalid_batches:
                print(
                    f"    - Warning: No nodes available for {component_id} in instance {batch_idx}"
                )
                service_found = service_id in services[batch_idx]
                if not service_found or comp_idx >= len(
                    component_embeddings[batch_idx]
                ):
                    continue
                c_specs = {
                    "cpu": component_embeddings[batch_idx][comp_idx].get(
                        "cpu", np.random.uniform(1, 10)
                    ),
                    "memory": component_embeddings[batch_idx][comp_idx].get(
                        "memory", np.random.uniform(1, 10)
                    ),
                    "disk": component_embeddings[batch_idx][comp_idx].get(
                        "disk", np.random.uniform(1, 10)
                    ),
                }
                resource_scores = torch.tensor(
                    [
                        max(
                            S[batch_idx][str(nodes[batch_idx][j]["nodeID"])]["cpu"]
                            - c_specs["cpu"],
                            0,
                        )
                        + max(
                            S[batch_idx][str(nodes[batch_idx][j]["nodeID"])]["memory"]
                            - c_specs["memory"],
                            0,
                        )
                        + max(
                            S[batch_idx][str(nodes[batch_idx][j]["nodeID"])]["disk"]
                            - c_specs["disk"],
                            0,
                        )
                        for j in range(charnum_node)
                    ],
                    device=device,
                )
                selected_node_idx = torch.argmax(resource_scores)
                p[batch_idx] = torch.zeros(charnum_node, device=device)
                p[batch_idx, selected_node_idx] = 1.0

            selected_node_indices = torch.argmax(p, dim=1)
            selected_nodes = [
                str(nodes[batch_idx][selected_node_indices[batch_idx].item()]["nodeID"])
                for batch_idx in range(batch_size)
            ]

            for batch_idx in range(batch_size):
                service_found = service_id in services[batch_idx]
                if not service_found or comp_idx >= len(
                    component_embeddings[batch_idx]
                ):
                    continue
                c_specs = {
                    "cpu": component_embeddings[batch_idx][comp_idx].get(
                        "cpu", np.random.uniform(1, 10)
                    ),
                    "memory": component_embeddings[batch_idx][comp_idx].get(
                        "memory", np.random.uniform(1, 10)
                    ),
                    "disk": component_embeddings[batch_idx][comp_idx].get(
                        "disk", np.random.uniform(1, 10)
                    ),
                }
                selected_node = selected_nodes[batch_idx]
                S[batch_idx][selected_node]["cpu"] -= c_specs["cpu"]
                S[batch_idx][selected_node]["memory"] -= c_specs["memory"]
                S[batch_idx][selected_node]["disk"] -= c_specs["disk"]
                if batch_idx >= len(placements):
                    placements.append({})
                placements[batch_idx][component_id] = selected_node
                print(
                    f"    - Instance {batch_idx}: Selected Node {selected_node} for {component_id}"
                )

    # Save parameters
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.cpu().tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    return placements, parameters
