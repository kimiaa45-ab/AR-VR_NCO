import json
import numpy as np
import torch


def embedding_load(
    batch_idx,
    charnum_node,
    charnum_service,
    charnum_component,
    num_samples,
    embedding_dim=32,
    data_dir="data/processed",
    microservices_data_path="data/raw/microservices_data.json",
):
    """
    Load and preprocess embeddings for computing nodes and microservices components.

    Args:
        batch_idx (int): Batch index (e.g., 1 for batch_1).
        charnum_node (int): Number of nodes (from config.yaml).
        charnum_service (int): Number of services (from config.yaml).
        charnum_component (int): Number of components per service (from config.yaml).
        num_samples (int): Number of instances to process in parallel.
        embedding_dim (int): Expected embedding dimension (default: 32).
        data_dir (str): Directory containing JSON files.
        microservices_data_path (str): Path to microservices dataset with component specs.

    Returns:
        tuple: (ems, emc)
            - ems: List of [num_samples] dictionaries with node embeddings.
            - emc: List of [num_samples] dictionaries with component embeddings and specs.
    """
    # Load JSON files
    computingnodes_path = f"{data_dir}/computingnodes_embedding_batch_{batch_idx}.json"
    microservices_path = f"{data_dir}/microservices_embedding_batch_{batch_idx}.json"

    with open(computingnodes_path, "r") as f:
        computingnodes_embedding = json.load(f)
    with open(microservices_path, "r") as f:
        microservices_embedding = json.load(f)

    # Load microservices dataset for component specs (assumed format)
    try:
        with open(microservices_data_path, "r") as f:
            microservices_data = json.load(f)
    except FileNotFoundError:
        print("Warning: Microservices data not found, using dummy specs")
        microservices_data = None

    # Initialize ems and emc for num_samples instances
    ems = []
    emc = []

    # Since JSON files contain one batch, replicate or sample for num_samples
    for sample_idx in range(num_samples):
        # Process computing nodes embeddings (ems)
        nodes_dict = computingnodes_embedding[0]  # Single batch
        ems_instance = {}
        for node_id in [f"Node{i+1}" for i in range(charnum_node)]:
            if node_id in nodes_dict:
                embeddings = np.array(nodes_dict[node_id])  # Shape: (20, 32)
                if embeddings.shape[0] == 0:
                    print(f"Warning: No embeddings for {node_id}, using zeros")
                    embedding = np.zeros(embedding_dim)
                else:
                    # Average 20 samples to get a single embedding
                    embedding = np.mean(embeddings, axis=0)  # Shape: (32,)
                ems_instance[node_id] = embedding.tolist()
            else:
                print(f"Warning: {node_id} not found, using zeros")
                ems_instance[node_id] = [0.0] * embedding_dim
        ems.append(ems_instance)

        # Process microservices embeddings (emc)
        services_dict = microservices_embedding[0]  # Single batch
        emc_instance = {}
        for service_idx in range(charnum_service):
            service_id = f"Service{service_idx + 1}"
            components = []
            if service_id in services_dict:
                embeddings = np.array(services_dict[service_id])  # Shape: (5, 32)
                if embeddings.shape[0] == 0:
                    print(f"Warning: No embeddings for {service_id}, using zeros")
                    service_embedding = np.zeros(embedding_dim)
                else:
                    # Average 5 samples to get a single service embedding
                    service_embedding = np.mean(embeddings, axis=0)  # Shape: (32,)

                # Generate component embeddings
                for comp_idx in range(charnum_component):
                    # Simplified: Use service embedding for components
                    # TODO: Replace with component-level embeddings if available
                    component_embedding = service_embedding.tolist()

                    # Get resource specs from dataset or use dummies
                    if microservices_data and sample_idx < len(microservices_data):
                        try:
                            specs = microservices_data[sample_idx][service_id][comp_idx]
                            component_specs = {
                                "embedding": component_embedding,
                                "cpu": specs.get("cpu", np.random.uniform(1, 10)),
                                "memory": specs.get("memory", np.random.uniform(1, 10)),
                                "disk": specs.get("disk", np.random.uniform(1, 10)),
                            }
                        except (KeyError, IndexError):
                            print(
                                f"Warning: Specs for {service_id}_C{comp_idx+1} not found, using dummies"
                            )
                            component_specs = {
                                "embedding": component_embedding,
                                "cpu": np.random.uniform(1, 10),
                                "memory": np.random.uniform(1, 10),
                                "disk": np.random.uniform(1, 10),
                            }
                    else:
                        component_specs = {
                            "embedding": component_embedding,
                            "cpu": np.random.uniform(1, 10),
                            "memory": np.random.uniform(1, 10),
                            "disk": np.random.uniform(1, 10),
                        }
                    components.append(component_specs)
            else:
                print(f"Warning: {service_id} not found, using empty components")
                components = [
                    {
                        "embedding": [0.0] * embedding_dim,
                        "cpu": np.random.uniform(1, 10),
                        "memory": np.random.uniform(1, 10),
                        "disk": np.random.uniform(1, 10),
                    }
                    for _ in range(charnum_component)
                ]
            emc_instance[service_id] = components
        emc.append(emc_instance)

    return ems, emc
