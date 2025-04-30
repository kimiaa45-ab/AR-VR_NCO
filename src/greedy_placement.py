import numpy as np


def greedy_placement(
    microservice, computing_nodes, computingnode_edge, component_connections
):
    """
    Performs greedy placement of microservice components on computing nodes, considering
    resource constraints and communication costs.

    Args:
        microservice (dict): Microservice data containing services.
        computing_nodes (list): List of computing nodes with characteristics (dicts).
        computingnode_edge (list): Adjacency matrix of edge weights between nodes.
        component_connections (list): Matrix of connection weights between components.

    Returns:
        dict: Mapping of (serviceID, componentID) to nodeID.
    """
    # Validate inputs
    if not isinstance(microservice, list):
        raise ValueError(f"microservice must be a dict, got {type(microservice)}")
    if not isinstance(computing_nodes, list):
        raise ValueError(f"computing_nodes must be a list, got {type(computing_nodes)}")
    if not isinstance(computingnode_edge, list):
        raise ValueError(
            f"computingnode_edge must be a list, got {type(computingnode_edge)}"
        )
    if not isinstance(component_connections, list):
        raise ValueError(
            f"component_connections must be a list, got {type(component_connections)}"
        )

    # Validate computing_nodes structure
    for node in computing_nodes:
        if (
            not isinstance(node, dict)
            or "nodeID" not in node
            or "characteristics" not in node
        ):
            raise ValueError(f"Invalid node structure in computing_nodes: {node}")

    # Validate computingnode_edge and component_connections
    for edge_row in computingnode_edge:
        if not isinstance(edge_row, list):
            raise ValueError(
                f"Invalid edge structure in computingnode_edge: {edge_row}"
            )
    for conn_row in component_connections:
        if not isinstance(conn_row, list):
            raise ValueError(
                f"Invalid connection structure in component_connections: {conn_row}"
            )

    # Initialize available resources for each node
    node_resources = {
        node["nodeID"]: {
            "cpu": node["characteristics"]["cpu"],
            "memory": node["characteristics"]["memory"],
            "disk": node["characteristics"]["disk"],
        }
        for node in computing_nodes
    }

    # Placement dictionary: (serviceID, componentID) -> nodeID
    placements = {}
    placed_components = {}

    # Process each service and its components
    for service in microservice.get("services", []):
        service_id = service["serviceID"]
        components = service["components"]
        num_components = len(components)

        for component in components:
            component_id = component["componentID"]
            comp_requirements = component["characteristics"]
            comp_cpu = comp_requirements["cpu"]
            comp_memory = comp_requirements["memory"]
            comp_disk = comp_requirements["disk"]

            # Calculate a score for each node
            node_scores = []
            for node in computing_nodes:
                node_id = node["nodeID"]
                resources = node_resources[node_id]

                # Check resource availability
                if (
                    resources["cpu"] >= comp_cpu
                    and resources["memory"] >= comp_memory
                    and resources["disk"] >= comp_disk
                ):
                    # Resource score (normalized)
                    resource_score = (
                        resources["cpu"] / 1000
                        + resources["memory"]
                        + resources["disk"] / 10
                    )

                    # Communication cost
                    comm_cost = 0
                    num_connections = 0
                    if component_connections and component_id <= num_components:
                        for (s_id, c_id), placed_node_id in placed_components.items():
                            if s_id == service_id and c_id <= num_components:
                                try:
                                    conn_weight = component_connections[c_id - 1][
                                        component_id - 1
                                    ]
                                except IndexError:
                                    conn_weight = 0
                                if conn_weight > 0:
                                    try:
                                        edge_weight = computingnode_edge[
                                            placed_node_id - 1
                                        ][node_id - 1][
                                            0
                                        ]  # Use distance
                                    except IndexError:
                                        edge_weight = 0
                                    comm_cost += conn_weight * edge_weight
                                    num_connections += 1

                    # Normalize communication cost
                    if num_connections > 0:
                        comm_cost /= num_connections
                    else:
                        comm_cost = 0

                    # Combined score
                    total_score = 0.7 * resource_score - 0.3 * (comm_cost / 500)
                    node_scores.append((node_id, total_score))

            # Sort nodes by score (descending)
            node_scores.sort(key=lambda x: x[1], reverse=True)

            # Place component on the first suitable node
            placed = False
            for node_id, _ in node_scores:
                if (
                    node_resources[node_id]["cpu"] >= comp_cpu
                    and node_resources[node_id]["memory"] >= comp_memory
                    and node_resources[node_id]["disk"] >= comp_disk
                ):
                    placements[(service_id, component_id)] = node_id
                    placed_components[(service_id, component_id)] = node_id
                    node_resources[node_id]["cpu"] -= comp_cpu
                    node_resources[node_id]["memory"] -= comp_memory
                    node_resources[node_id]["disk"] -= comp_disk
                    placed = True
                    break

            if not placed:
                placements[(service_id, component_id)] = None

    return placements
