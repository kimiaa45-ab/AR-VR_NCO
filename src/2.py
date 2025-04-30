def gnn_model(
    microservices,
    microservices_edges,
    computingnodes,
    computingnodes_edges,
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

    # Initialize output lists
    microservices_embedding = []
    microservices_edges_embedding = []
    computingnodes_embedding = []
    computingnodes_edges_embedding = []

    # Process each sample
    for microservice, infraConnection in zip(computingnodes, computingnodes_edges):
        # Step 3: Extract servers

        nodes_data = []
        for server in computingnodes.get("servers", []):
            server_data = {
                "nodeID": server.get("nodeID", 0),
                "nodeTier": server.get("nodeTier", 0),  # Fixed key to match JSON
                "cpu": server["characteristics"]["cpu"],  # Use server instead of comp
                "memory": server["characteristics"]["memory"],
                "disk": server["characteristics"]["disk"],
                "reliabilityScore": server["characteristics"]["reliabilityScore"],
            }
            nodes_data.append(server_data)

        # print(f"  - Number of servers: {len(nodes_data)}")

        # Extract and convert componentinfraConnection to dictionary
        dependency_data = {}
        for server in nodes_data:
            service_name = f"server{server['nodeID']}"
            app_deps = {}
            # print(f"componentinfraConnection for {server_name}: {infraConnection[:100]}...")
            num_components = len(server["components"])
            if (
                infraConnection
                and isinstance(infraConnection, list)
                and all(isinstance(row, list) for row in infraConnection)
            ):
                try:
                    for i in range(len(infraConnection)):
                        for j in range(len(infraConnection[i])):
                            if (
                                infraConnection[i][j] > 0
                                and i < num_components
                                and j < num_components
                            ):
                                edge_key = (
                                    f"e_c{i+1}_c{j+1}" if i < j else f"e_c{j+1}_c{i+1}"
                                )
                                app_deps[edge_key] = float(
                                    infraConnection[i][j]
                                )  # Scalar value
                except (TypeError, IndexError) as e:
                    print(
                        f"Warning: Invalid componentinfraConnection format for {service_name}: {e}. Skipping."
                    )
            else:
                print(
                    f"Warning: componentinfraConnection is not a valid matrix for {service_name}. Skipping."
                )
            dependency_data[service_name] = app_deps

        app_h_C_final = {}
        app_e_C_final = {}

        for server in nodes_data:
            service_name = f"server{server['nodeID']}"
            components = server["components"]
            user_id = server["userID"]
            helper_id = server["helperID"]
            # print(f"\nProcessing server: {server_name} (User: {user_id}, Helper: {helper_id})")
            num_components = len(components)
            # print(f"  - Number of components: {num_components}")

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
            # h_C_layers = torch.round(h_C_layers * 100) / 100

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
                    e_C_dict[edge_key] = norm_deps[idx]  # Scalar edge weight
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

                for i in range(num_layers):
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
                            continue  # Skip invalid neighbors
                        edge_key = f"e_c{i+1}_c{j+1}" if i < j else f"e_c{j+1}_c{i+1}"
                        if edge_key in sigmoid_e_C:
                            sig_e_ij = sigmoid_e_C[edge_key]  # Scalar
                            sig_e_ij = torch.round(sig_e_ij * 100) / 100
                            nc_h_j = VS_h_C[j]  # Shape: [dim]
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
            e_C_layers = [dict(e_C_dict)]  # Initialize with scalar weights
            for layer in range(num_components - 1):
                e_C_current = e_C_layers[layer]
                h_C_current = h_C_layers[layer]
                e_C_next = e_C_current.copy()
                for edge_key in e_C_current.keys():
                    c1, c2 = map(int, edge_key.split("_c")[1:])
                    i, j = c1 - 1, c2 - 1
                    if i >= num_components or j >= num_components:
                        continue  # Skip invalid edges
                    e_ij = e_C_current[edge_key] * WSE  # Scalar * [charnum_se, dim]
                    h_i, h_j = h_C_current[i], h_C_current[j]
                    AS_e_ij = e_ij @ AS
                    AS_e_ij = torch.round(AS_e_ij * 100) / 100
                    DS_e_ij = e_ij @ DS
                    DS_e_ij = torch.round(DS_e_ij * 100) / 100
                    BS_h_i = h_i @ BS
                    BS_h_i = torch.round(BS_h_i * 100) / 100
                    CS_h_j = h_j @ CS  # Assuming DS_h_j is CS_h_j
                    CS_h_j = torch.round(CS_h_j * 100) / 100
                    aggr = AS_e_ij + DS_e_ij + BS_h_i + CS_h_j
                    aggr = torch.round(aggr * 100) / 100
                    norm_aggr = normalize_vector(aggr)
                    norm_aggr = torch.round(norm_aggr * 100) / 100
                    relu_result = relu(norm_aggr)
                    relu_result = torch.round(relu_result * 100) / 100
                    e_C_next[edge_key] = e_C_current[edge_key] + torch.mean(
                        relu_result
                    )  # Reduce to scalar
                    # e_C_next= torch.round(e_C_next * 100) / 100
                e_C_layers.append(e_C_next)

            # Store final layer
            final_h_C_rounded = torch.round(h_C_layers[-1] * 100) / 100
            final_h_C_rounded = torch.round(final_h_C_rounded * 100) / 100
            final_e_C_rounded = {
                key: round(float(val), 2) for key, val in e_C_layers[-1].items()
            }

            app_h_C_final[service_name] = final_h_C_rounded.tolist()
            app_e_C_final[service_name] = final_e_C_rounded

        computingnodess_embedding.append(app_h_C_final)
        computingnodes_edges_embedding.append(app_e_C_final)

    # Save parameters to settings.json for reuse in next run
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.cpu().tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    updated_parameters = parameters  # No modifications, just saved to JSON

    return (
        computingnodess_embedding,
        computingnodes_edges_embedding,
        computingnodes_embedding,
        computingnodes_edges_embedding,
        updated_parameters,
    )
