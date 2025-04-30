import yaml
import json
import os
import torch


def initialize_settings(
    config_path="configs/config.yaml",
    settings_path="configs/setting.json",
    force_init=False,
):
    """
    Initialize or load matrix parameters (US, UN, WS, WN, WSE, WNE, AS, BS, CS, DS, AN, BN, CN) and save to settings.json.

    Args:
        config_path (str): Path to config.yaml containing charnum_s, etc.
        settings_path (str): Path to save/load setting.json.
        force_init (bool): If True, regenerate random matrices even if setting.json exists.

    Returns:
        dict: Dictionary containing US, UN, WS, WN, WSE, WNE, AS, BS, CS, DS, AN, BN, CN as torch.Tensor.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract parameters directly
    charnum_s = config["model"]["charnum_s"]
    charnum_n = config["model"]["charnum_n"]
    charnum_se = config["model"]["charnum_se"]
    charnum_ne = config["model"]["charnum_ne"]
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

    # Validate parameters
    if not all(
        isinstance(x, int) and x > 0
        for x in [dim, charnum_s, charnum_n, charnum_se, charnum_ne]
    ):
        raise ValueError(
            "dim, charnum_s, charnum_n, charnum_se, charnum_ne must be positive integers"
        )

    # Required matrix keys
    matrix_keys = [
        "US",
        "VS",
        "UN",
        "VN",
        "WS",
        "WN",
        "WSE",
        "WNE",
        "AS",
        "BS",
        "CS",
        "DS",
        "AN",
        "BN",
        "CN",
        "WSQ",
        "WNK",
    ]

    # Check if settings file exists and force_init is False
    if os.path.exists(settings_path) and not force_init:
        with open(settings_path, "r") as f:
            settings = json.load(f)
        # Verify all required keys are present
        if all(key in settings for key in matrix_keys):
            # Convert lists back to tensors
            return {
                key: torch.tensor(settings[key], dtype=torch.float32)
                for key in matrix_keys
            }

    # Generate random matrices in [-1, 1], rounded to 2 decimal places
    def generate_random_matrix(size):
        # Generate random values in [0, 1], scale to [-1, 1]
        matrix = 2 * torch.rand(size, dtype=torch.float32) - 1
        # Round to 2 decimal places
        matrix = torch.round(matrix * 100) / 100
        return matrix

    # Initialize matrices
    parameters = {
        "US": generate_random_matrix((dim, dim)),
        "VS": generate_random_matrix((dim, dim)),
        "UN": generate_random_matrix((dim, dim)),
        "VN": generate_random_matrix((dim, dim)),
        "WS": generate_random_matrix((charnum_s, dim)),
        "WN": generate_random_matrix((charnum_n, dim)),
        "WSE": generate_random_matrix((charnum_se, dim)),
        "WNE": generate_random_matrix((charnum_ne, dim)),
        "AS": generate_random_matrix((dim, dim)),
        "BS": generate_random_matrix((dim, dim)),
        "CS": generate_random_matrix((dim, dim)),
        "DS": generate_random_matrix((dim, dim)),
        "AN": generate_random_matrix((dim, dim)),
        "BN": generate_random_matrix((dim, dim)),
        "CN": generate_random_matrix((dim, dim)),
        "WSQ": generate_random_matrix((dim, dim)),
        "WNK": generate_random_matrix((dim, dim)),
    }

    # Save to settings.json with exactly 2 decimal places
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    with open(settings_path, "w") as f:
        # Convert tensors to lists with 2 decimal place formatting
        formatted_parameters = {
            key: [[round(float(val), 2) for val in row] for row in param.tolist()]
            for key, param in parameters.items()
        }
        json.dump(formatted_parameters, f, indent=4)

    return parameters
