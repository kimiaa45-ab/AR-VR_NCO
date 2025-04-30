import json
import torch
import copy
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


def AR_pytorch(
    X,
    parameters,
    device="cpu",
    settings_path="configs/setting.json",
):

    # Move parameters to device
    parameters = {k: v.to(device) for k, v in parameters.items()}
    (
        WSQ,
        WNK,
    ) = (
        parameters["WSQ"],
        parameters["WNK"],
    )
    print("meghdar avalie", len(WSQ))
    # Load JSON files
    with open(f"data/processed/nodes_{X}.json", "r", encoding="utf-8") as f:
        nodes = json.load(f)
    with open(f"data/processed/services_{X}.json", "r", encoding="utf-8") as f:
        services = json.load(f)
    with open(f"data/processed/msg_component_{X}.json", "r", encoding="utf-8") as f:
        emc = json.load(f)
    with open(f"data/processed/msg_server_{X}.json", "r", encoding="utf-8") as f:
        ems = json.load(f)
    with open(
        f"data/processed/newInfraConnections_{X}.json", "r", encoding="utf-8"
    ) as f:
        connectivity = json.load(f)
    with open(
        f"data/processed/newcomponentsConnections_{X}.json", "r", encoding="utf-8"
    ) as f:
        dag = json.load(f)
    with open(f"data/processed/msg_Sedge_{X}.json", "r", encoding="utf-8") as f:
        sedge = json.load(f)
    with open(f"data/processed/msg_Cedge_{X}.json", "r", encoding="utf-8") as f:
        cedge = json.load(f)

    # Initialize matrices
    # WCQ = torch.rand(128, 128, dtype=torch.float32) * 2 - 1  # Random between -1 and 1
    # WSK = torch.rand(128, 128, dtype=torch.float32) * 2 - 1
    WCQ = WSQ
    print("kcqkwnk  lwn knd kn  dkl", len(WCQ))
    WSK = WNK

    # Prepare server embeddings (KS) and normalize
    KS = torch.tensor(ems, dtype=torch.float32) @ WSK  # Shape: (num_nodes, 128)
    KS = (KS - torch.min(KS, dim=1, keepdim=True)[0]) / (
        torch.max(KS, dim=1, keepdim=True)[0] - torch.min(KS, dim=1, keepdim=True)[0]
    )
    num_nodes = charnum_node

    # Convert connectivity to tensor
    connectivity_matrix = torch.tensor(
        connectivity, dtype=torch.float32
    )  # Shape: (43, 43)

    # Store initial node resources
    initial_resources = {
        str(node["nodeID"]): {
            "cpu": node["characteristics"]["cpu"],
            "memory": node["characteristics"]["memory"],
            "disk": node["characteristics"]["disk"],
        }
        for node in nodes
    }

    # منابع جاری که تجمعی تغییر می‌کنه
    S = copy.deepcopy(initial_resources)

    # Assignment dictionary
    placement = {}

    # Process each service
    for z, (service_id, service_components) in enumerate(emc.items(), 1):
        print(f"\nProcessing Service {z}: {service_id}")

        # استخراج شماره از "ServiceX"
        service_number = service_id.replace("Service", "")

        # پیدا کردن سرویس متناظر در services.json
        service = next(
            (s for s in services if str(s.get("serviceID")) == service_number), None
        )
        if not service:
            print(
                f"Warning: No service found for ID {service_number} (from {service_id}), skipping."
            )
            continue

        # Get DAG for this service
        # service_dag = dag.get(service_id, [])

        # Process each component
        for i, hc_i in enumerate(service_components, 1):
            component_id = f"S{service_number}_C{i}"
            # print(f"  - Processing Component: {component_id}")

            # Compute qc and normalize
            qc = torch.tensor(hc_i, dtype=torch.float32) @ WCQ  # Shape: (128,)
            qc = (qc - torch.min(qc)) / (
                torch.max(qc) - torch.min(qc)
            )  # Normalize to [0, 1]

            # Compute init_S with connectivity
            inits = []
            c_specs = service["components"][i - 1]  # Direct access to component dict

            # print(f"    - Component Demands: cpu={c_specs['cpu']}, memory={c_specs['memory']}, disk={c_specs['disk']}")
            # print(f"    - Current Node Resources: {S}")

            for j in range(num_nodes):
                node_key = str(nodes[j]["nodeID"])
                u = qc @ KS[j]  # Base compatibility score

                # Adjust score with connectivity
                if j < 43 and connectivity_matrix.shape[0] == 43:
                    connectivity_factor = torch.mean(connectivity_matrix[j])
                    u *= 1 + connectivity_factor

                # Check resources
                if (
                    S[node_key]["cpu"] < c_specs["characteristics"]["cpu"]
                    or S[node_key]["memory"] < c_specs["characteristics"]["memory"]
                    or S[node_key]["disk"] < c_specs["characteristics"]["disk"]
                ):
                    u = float("-inf")

                inits.append(u)

            inits = torch.tensor(inits, dtype=torch.float32)
            # print(f"    - init_S: {inits}")

            # Stable softmax
            finite_inits = torch.where(torch.isinf(inits), torch.tensor(-1e10), inits)
            shifted_inits = finite_inits - torch.max(finite_inits)
            eu = torch.exp(shifted_inits)
            zigma_inits = torch.sum(eu)

            if zigma_inits == 0:
                print(
                    f"    - Warning: No nodes available for {component_id}, using resource scores"
                )
                resource_scores = [
                    max(S[str(nodes[j]["nodeID"])]["cpu"] - c_specs["cpu"], 0)
                    + max(S[str(nodes[j]["nodeID"])]["memory"] - c_specs["memory"], 0)
                    + max(S[str(nodes[j]["nodeID"])]["disk"] - c_specs["disk"], 0)
                    for j in range(num_nodes)
                ]
                print(f"    - Resource Scores: {resource_scores}")
                selected_node_idx = torch.argmax(torch.tensor(resource_scores))
                p = torch.zeros(num_nodes)
            else:
                p = eu / zigma_inits
                selected_node_idx = torch.argmax(p)

            selected_node = str(nodes[selected_node_idx.item()]["nodeID"])
            print(
                f"    - Probabilities: {', '.join([f'{str(nodes[j]["nodeID"])}: {prob:.3f}' for j, prob in enumerate(p)])}"
            )
            print(f"    - Selected Node: {selected_node}")

            # Update node resources
            S[selected_node]["cpu"] -= c_specs["characteristics"]["cpu"]
            S[selected_node]["memory"] -= c_specs["characteristics"]["memory"]
            S[selected_node]["disk"] -= c_specs["characteristics"]["disk"]

            # Record placement
            placement[component_id] = selected_node

        # print(f"  - Resources after Service {service_id}:")
        # for n_key, res in S.items():
        # print(
        #    f"    {n_key}: cpu={res['cpu']:.2f}, memory={res['memory']:.2f}, disk={res['disk']:.2f}"
        # )

    # Save and display
    with open(f"data/processed/placement_{X}.json", "w", encoding="utf-8") as f:
        json.dump(placement, f, indent=4)

    print("\nFinal Placement:")
    for component, node in placement.items():
        print(f"  {component} -> {node}")


# print("\nFinal Node Resources (after all services):")
# for n_key, res in S.items():
#   print(
#      f"    {n_key}: cpu={res['cpu']:.2f}, memory={res['memory']:.2f}, disk={res['disk']:.2f}"
# )
