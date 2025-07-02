"""
FBD Logic for Function Block Diversification
"""
import json
import copy
import importlib.util
import sys
import os
import torch
import torch.nn as nn
import logging
from collections import defaultdict
from argparse import Namespace
import hashlib
import shutil

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    # Create fbd_log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure the root logger to redirect its output
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to prevent duplicate logs or console output
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Create a file handler for the root logger
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Return a specific logger for the component
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger

def load_fbd_settings(fbd_file_path):
    """
    Load FBD_TRACE, FBD_INFO, and TRANSPARENT_TO_CLIENT from a Python file.
    
    Args:
        fbd_file_path (str): Path to the FBD settings file (e.g., 'fbd_record/bloodmnist_plan_1.py')
        
    Returns:
        tuple: (FBD_TRACE, FBD_INFO, TRANSPARENT_TO_CLIENT) loaded from the file
    """
    # Convert relative path to absolute path
    abs_path = os.path.abspath(fbd_file_path)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("fbd_settings", abs_path)
    fbd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fbd_module)
    
    # Extract the settings
    fbd_trace = getattr(fbd_module, 'FBD_TRACE', {})
    fbd_info = getattr(fbd_module, 'FBD_INFO', {})
    transparent_to_client = getattr(fbd_module, 'TRANSPARENT_TO_CLIENT', True)  # Default to True for backward compatibility
    
    return fbd_trace, fbd_info, transparent_to_client

def load_shipping_plan(shipping_plan_path):
    """
    Load shipping plan from JSON file.
    
    Args:
        shipping_plan_path (str): Path to shipping plan JSON file
        
    Returns:
        dict: Shipping plan with round numbers as keys
    """
    with open(shipping_plan_path, 'r') as f:
        shipping_plan = json.load(f)
    
    # Convert string keys to integers for round numbers
    return {int(round_num): clients for round_num, clients in shipping_plan.items()}

def load_request_plan(request_plan_path):
    """
    Load request plan from JSON file.
    
    Args:
        request_plan_path (str): Path to request plan JSON file
        
    Returns:
        dict: Request plan with round numbers as keys
    """
    with open(request_plan_path, 'r') as f:
        request_plan = json.load(f)
    
    # Convert string keys to integers for round numbers
    return {int(round_num): clients for round_num, clients in request_plan.items()}

def save_optimizer_state_by_block(optimizer, model, fbd_trace, trainable_block_ids):
    """
    Extracts optimizer state and partitions it by FBD block ID for trainable blocks.
    The state for each parameter is keyed by its full name (e.g., 'layer1.0.conv1.weight').
    """
    param_groups_info = []
    if optimizer.param_groups:
        group = optimizer.param_groups[0]
        info = {k: v for k, v in group.items() if k != 'params'}
        param_groups_info.append(info)

    full_state = optimizer.state_dict()['state']
    
    # Create a reverse map from parameter object ID to its full name for efficient lookup.
    param_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Map the optimizer's internal parameter indices (which are the keys in full_state) to their full names.
    optim_idx_to_param_name_map = {}
    current_param_idx = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            param_name = param_id_to_name_map.get(id(p))
            if param_name:
                 optim_idx_to_param_name_map[current_param_idx] = param_name
            current_param_idx += 1

    partitioned_state = defaultdict(dict)
    for block_id in trainable_block_ids:
        model_part_prefix = fbd_trace[block_id]['model_part']
        for param_idx, param_name in optim_idx_to_param_name_map.items():
            if param_name.startswith(model_part_prefix) and param_idx in full_state:
                state_values = full_state[param_idx]
                cloned_state_vals = {}
                for k, v in state_values.items():
                    if isinstance(v, torch.Tensor):
                        cloned_state_vals[k] = v.detach().cpu().clone()
                    else:
                        cloned_state_vals[k] = v
                partitioned_state[block_id][param_name] = cloned_state_vals
                
    final_block_states = {}
    for block_id, state_data in partitioned_state.items():
        if state_data:
            final_block_states[block_id] = {
                'param_groups': param_groups_info,
                'state': state_data
            }
            
    return final_block_states

def build_optimizer_with_state(model, optimizer_states, trainable_params, device, default_lr=0.001):
    """
    Builds an Adam optimizer and correctly loads state for a specific set of trainable parameters.
    """
    # 1. Determine hyper-parameters from the first available state block
    param_groups_info = None
    if optimizer_states:
        for block_id, state_data in optimizer_states.items():
            if state_data and 'param_groups' in state_data:
                param_groups_info = state_data['param_groups']
                break
    
    # 2. Create a new optimizer with only the trainable parameters
    if param_groups_info:
        # Use hyper-parameters from the saved state
        optimizer = torch.optim.Adam(trainable_params, **param_groups_info[0])
    else:
        # Fallback to default if no state was provided
        optimizer = torch.optim.Adam(trainable_params, lr=default_lr)
        
    # 3. If there's state to load, map it correctly to the new optimizer
    if optimizer_states:
        # The optimizer's state_dict maps integer param IDs to their state.
        # We need to map the trainable parameters to their corresponding IDs in the new optimizer.
        param_to_id_map = {id(p): i for i, group in enumerate(optimizer.param_groups) for p in group['params']}
        
        # This will be the new state we load into the optimizer
        new_state_dict = defaultdict(dict)

        # Create a reverse map from parameter name to parameter object for easy lookup
        name_to_param_map = {name: p for name, p in model.named_parameters() if p.requires_grad}

        for block_id, state_data in optimizer_states.items():
            if not state_data or 'state' not in state_data:
                continue
            
            # The state_data['state'] is a dict mapping OLD param names to their state
            for param_name, state_values in state_data['state'].items():
                if param_name in name_to_param_map:
                    param_obj = name_to_param_map[param_name]
                    param_id = param_to_id_map.get(id(param_obj))
                    
                    if param_id is not None:
                        # Move state tensors to the correct device
                        for k, v in state_values.items():
                            if isinstance(v, torch.Tensor):
                                state_values[k] = v.to(device)
                        new_state_dict[param_id] = state_values

        if new_state_dict:
            final_optim_state = optimizer.state_dict()
            final_optim_state['state'] = new_state_dict
            optimizer.load_state_dict(final_optim_state)

    return optimizer

class FBDWarehouse:
    """
    Warehouse for storing and managing function block weights at the server.
    Organizes weights by FBD block IDs and enables flexible weight shipping/receiving.
    """
    
    def __init__(self, fbd_trace, model_template=None, log_file_path=None):
        """
        Initialize the warehouse.
        
        Args:
            fbd_trace (dict): FBD_TRACE dictionary mapping block IDs to model parts and colors
            model_template (nn.Module, optional): Template model to initialize weights from
            log_file_path (str, optional): Path to warehouse log file (default: warehouse.log)
        """
        self.fbd_trace = fbd_trace
        self.warehouse = {}  # Dictionary storing weights by block ID
        
        # Set up warehouse logging
        self._setup_warehouse_logger(log_file_path)
        
        # Initialize warehouse with random weights or from template model
        self._initialize_warehouse(model_template)
    
    def _setup_warehouse_logger(self, log_file_path=None):
        """
        Set up warehouse-specific logger.
        
        Args:
            log_file_path (str, optional): Path to log file. If None, defaults to 'warehouse.log'
        """
        # Create warehouse-specific logger
        self.warehouse_logger = logging.getLogger("FBDWarehouse")
        self.warehouse_logger.setLevel(logging.INFO)
        
        # Avoid adding handlers multiple times
        if not self.warehouse_logger.handlers:
            # Set default log file path
            if log_file_path is None:
                log_file_path = "warehouse.log"
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.warehouse_logger.addHandler(file_handler)
            
            # Prevent propagation to root logger to avoid duplicate logging
            self.warehouse_logger.propagate = False
            
            self.warehouse_logger.info(f"FBD Warehouse logging initialized - Log file: {log_file_path}")
            self.warehouse_logger.info(f"Warehouse initialized with {len(self.fbd_trace)} FBD blocks")
    
    def _initialize_warehouse(self, model_template=None):
        """
        Initialize warehouse with weights for all function blocks.
        
        Args:
            model_template (nn.Module, optional): Template model to copy weights from
        """
        if model_template is not None:
            # Initialize from template model
            model_state = model_template.state_dict()
            self.warehouse_logger.info(f"Initializing warehouse from template model with {len(model_state)} parameters")
            
            for block_id, block_info in self.fbd_trace.items():
                model_part = block_info['model_part']
                color = block_info['color']
                
                # Extract weights for this model part
                part_weights = {}
                for param_name, param_tensor in model_state.items():
                    if param_name.startswith(model_part + '.'):
                        part_weights[param_name] = param_tensor.clone()
                
                self.warehouse[block_id] = {
                    'weights': part_weights,
                    'optimizer_state': {}
                }
                self.warehouse_logger.info(f"Initialized block {block_id} (color: {color}, part: {model_part}) with {len(part_weights)} parameters")
        else:
            # Initialize with empty dictionaries - will be populated later
            self.warehouse_logger.info("Initializing warehouse with empty blocks - will be populated during training")
            for block_id in self.fbd_trace.keys():
                self.warehouse[block_id] = {
                    'weights': {},
                    'optimizer_state': {}
                }
                
        self.warehouse_logger.info(f"Warehouse initialization complete - {len(self.warehouse)} blocks ready")
    
    def store_weights(self, block_id, state_dict):
        """
        Store weights for a specific function block.
        
        Args:
            block_id (str): FBD block ID (e.g., "AFA79")
            state_dict (dict): State dictionary containing the weights
        """
        if block_id not in self.fbd_trace:
            error_msg = f"Unknown block ID: {block_id}"
            self.warehouse_logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get block info for logging
        block_info = self.fbd_trace[block_id]
        color = block_info['color']
        model_part = block_info['model_part']
        
        # Analyze weights quality (not all zeros or identical)
        total_norm = 0.0
        param_count = 0
        tensor_types = set()
        
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                param_count += 1
                tensor_types.add(str(v.dtype))
                
                # Convert integer tensors to float before computing norm
                if v.dtype in [torch.long, torch.int, torch.short, torch.uint8, torch.int32, torch.int64]:
                    # For integer tensors, convert to float temporarily for norm calculation
                    v_float = v.float()
                    total_norm += torch.norm(v_float).item()
                else:
                    # For floating point tensors, compute norm directly
                    total_norm += torch.norm(v).item()
        
        # Log detailed warehouse storage information
        self.warehouse_logger.info(f"Storing block {block_id} (color: {color}, part: {model_part})")
        self.warehouse_logger.info(f"  └─ Parameters: {param_count}, Total norm: {total_norm:.6f}")
        self.warehouse_logger.info(f"  └─ Tensor types: {', '.join(sorted(tensor_types))}")
        
        # Store the weights
        self.warehouse[block_id]['weights'] = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                   for k, v in state_dict.items()}
        
        self.warehouse_logger.info(f"Successfully stored block {block_id} in warehouse")
    
    def store_optimizer_state(self, block_id, optimizer_state):
        """
        Store optimizer state for a specific function block.
        
        Args:
            block_id (str): FBD block ID (e.g., "AFA79")
            optimizer_state (dict): Dictionary containing the optimizer state
        """
        if block_id not in self.fbd_trace:
            error_msg = f"Unknown block ID: {block_id}"
            self.warehouse_logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.warehouse[block_id]['optimizer_state'] = optimizer_state
        self.warehouse_logger.info(f"Stored optimizer state for block {block_id}")

    def store_optimizer_state_batch(self, optimizer_states_dict):
        """
        Store optimizer states for multiple function blocks at once.
        
        Args:
            optimizer_states_dict (dict): Dictionary mapping block IDs to their optimizer states
        """
        self.warehouse_logger.info(f"Starting batch storage of optimizer states for {len(optimizer_states_dict)} blocks")
        for block_id, state in optimizer_states_dict.items():
            try:
                self.store_optimizer_state(block_id, state)
            except Exception as e:
                self.warehouse_logger.error(f"Failed to store optimizer state for block {block_id}: {e}")
        self.warehouse_logger.info("Batch storage of optimizer states complete.")
    
    def store_weights_batch(self, weights_dict):
        """
        Store weights for multiple function blocks at once.
        
        Args:
            weights_dict (dict): Dictionary mapping block IDs to their state_dicts
        """
        self.warehouse_logger.info(f"Starting batch storage of {len(weights_dict)} blocks: {list(weights_dict.keys())}")
        
        success_count = 0
        for block_id, state_dict in weights_dict.items():
            try:
                self.store_weights(block_id, state_dict)
                success_count += 1
            except Exception as e:
                self.warehouse_logger.error(f"Failed to store block {block_id}: {e}")
                
        self.warehouse_logger.info(f"Batch storage complete: {success_count}/{len(weights_dict)} blocks stored successfully")
    
    def retrieve_weights(self, block_id):
        """
        Retrieve weights for a specific function block.
        
        Args:
            block_id (str): FBD block ID
            
        Returns:
            dict: State dictionary containing the weights
        """
        if block_id not in self.warehouse:
            error_msg = f"Block ID not found in warehouse: {block_id}"
            self.warehouse_logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get block info for logging
        block_info = self.fbd_trace[block_id]
        color = block_info['color']
        model_part = block_info['model_part']
        
        weights = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                   for k, v in self.warehouse[block_id]['weights'].items()}
        
        self.warehouse_logger.info(f"Retrieved block {block_id} (color: {color}, part: {model_part}) - {len(weights)} parameters")
        
        return weights
    
    def retrieve_optimizer_state(self, block_id):
        """
        Retrieve optimizer state for a specific function block.
        
        Args:
            block_id (str): FBD block ID
            
        Returns:
            dict: Optimizer state dictionary
        """
        if block_id not in self.warehouse:
            error_msg = f"Block ID not found in warehouse: {block_id}"
            self.warehouse_logger.error(error_msg)
            raise ValueError(error_msg)
        
        return self.warehouse[block_id].get('optimizer_state', {})
    
    def retrieve_weights_batch(self, block_ids):
        """
        Retrieve weights for multiple function blocks.
        
        Args:
            block_ids (list): List of FBD block IDs
            
        Returns:
            dict: Dictionary mapping block IDs to their state_dicts
        """
        return {block_id: self.retrieve_weights(block_id) for block_id in block_ids}
    
    def get_model_weights(self, model_color):
        """
        Reconstruct complete model weights for a specific model (color).
        
        Args:
            model_color (str): Model color (e.g., "M0", "M1", etc.)
            
        Returns:
            dict: Complete state dictionary for the model organized by model parts
        """
        self.warehouse_logger.info(f"Reconstructing complete model weights for color: {model_color}")
        
        full_state_dict = {}
        block_ids_for_color = [block_id for block_id, block_info in self.fbd_trace.items() if block_info['color'] == model_color]
        
        # Retrieve weights for each block and combine them
        for block_id in block_ids_for_color:
            block_weights = self.retrieve_weights(block_id)
            full_state_dict.update(block_weights)
            self.warehouse_logger.info(f"  └─ Added block {block_id} to state_dict")

        self.warehouse_logger.info(f"Model reconstruction for {model_color} complete. Total params: {len(full_state_dict)}")
        return full_state_dict
    
    def get_shipping_weights(self, shipping_list):
        """
        Prepare weights for shipping according to shipping plan.
        
        Args:
            shipping_list (list): List of block IDs to ship
            
        Returns:
            dict: A single state dictionary containing all weights to be shipped.
        """
        shipping_weights = {}
        
        for block_id in shipping_list:
            if block_id in self.fbd_trace:
                block_weights = self.retrieve_weights(block_id)
                shipping_weights.update(block_weights)
        
        return shipping_weights
    
    def get_shipping_optimizer_states(self, shipping_list):
        """
        Prepare optimizer states for shipping.
        
        Args:
            shipping_list (list): List of block IDs for which to ship optimizer states.
            
        Returns:
            dict: A dictionary mapping block_id to its optimizer state.
        """
        shipping_states = {}
        for block_id in shipping_list:
            if block_id in self.fbd_trace:
                shipping_states[block_id] = self.retrieve_optimizer_state(block_id)
        return shipping_states
    
    def warehouse_summary(self):
        """
        Get summary information about the warehouse contents.
        
        Returns:
            dict: Summary including block counts, model coverage, etc.
        """
        summary = {
            'total_blocks': len(self.warehouse),
            'models': defaultdict(list),
            'model_parts': defaultdict(list),
            'empty_blocks': []
        }
        
        for block_id, data in self.warehouse.items():
            if block_id in self.fbd_trace:
                block_info = self.fbd_trace[block_id]
                color = block_info['color']
                model_part = block_info['model_part']
                
                summary['models'][color].append(block_id)
                summary['model_parts'][model_part].append(block_id)
                
                if not data.get('weights'):
                    summary['empty_blocks'].append(block_id)
        
        return dict(summary)
    
    def save_warehouse(self, filepath):
        """
        Save warehouse state to file.
        
        Args:
            filepath (str): Path to save the warehouse
        """
        torch.save({
            'warehouse': self.warehouse,
            'fbd_trace': self.fbd_trace
        }, filepath)
    
    def load_warehouse(self, filepath):
        """
        Load warehouse state from file.
        
        Args:
            filepath (str): Path to load the warehouse from
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.warehouse = checkpoint['warehouse']
        self.fbd_trace = checkpoint['fbd_trace']

def generate_client_model_palettes(num_clients, fbd_file_path):
    """
    Generate model palettes for each client based on FBD settings from file.
    
    Args:
        num_clients (int): Number of clients in the federated learning setup
        fbd_file_path (str): Path to the FBD settings file
        
    Returns:
        dict: Dictionary where keys are client IDs and values are their model palettes
    """
    # Load FBD settings from file
    fbd_trace, fbd_info, transparent_to_client = load_fbd_settings(fbd_file_path)
    
    client_model_palettes = {}
    
    for cid in range(num_clients):
        if cid in fbd_info["clients"]:
            # Get the models (colors) this client has access to
            client_colors = fbd_info["clients"][cid]
            
            # Create the model palette for this client
            model_palette = {}
            for fbd_id, fbd_entry in fbd_trace.items():
                if fbd_entry["color"] in client_colors:
                    # Make a deep copy to avoid modifying the original
                    palette_entry = copy.deepcopy(fbd_entry)
                    
                    # Remove color information if not transparent to client
                    if not transparent_to_client:
                        palette_entry.pop("color", None)
                    
                    model_palette[fbd_id] = palette_entry
            
            client_model_palettes[cid] = model_palette
        else:
            # Default: client has access to all models if not specified in plan
            default_palette = copy.deepcopy(fbd_trace)
            
            # Remove color information from all entries if not transparent to client
            if not transparent_to_client:
                for fbd_id, fbd_entry in default_palette.items():
                    fbd_entry.pop("color", None)
            
            client_model_palettes[cid] = default_palette
    
    return client_model_palettes

def load_config(data_flag: str, model_flag: str) -> Namespace:
    """Load configuration for a given dataset and model."""
    config_path = f"config/{data_flag}/config.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    for config in configs:
        if config['model_flag'] == model_flag:
            return Namespace(**config)
    
    raise ValueError(f"Configuration for model {model_flag} not found in {config_path}")

def save_json(data: dict, filepath: str):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        filepath (str): The path to the output JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def handle_dataset_cache(dataset_name: str, cache_dir: str):
    """
    Manages the MedMNIST dataset cache.
    It checks for a dataset in a local cache directory and copies it to the
    `~/.medmnist/` directory for use by the training scripts. If the dataset

    is downloaded, it's copied back to the cache.
    """
    MEDMNIST_DIR = os.path.expanduser("~/.medmnist")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logging.info(f"Created cache directory: {cache_dir}")
    if not os.path.exists(MEDMNIST_DIR):
        os.makedirs(MEDMNIST_DIR)
        logging.info(f"Created MedMNIST directory: {MEDMNIST_DIR}")
        
    source_npz_path = os.path.join(cache_dir, f"{dataset_name}.npz")
    dest_npz_path = os.path.join(MEDMNIST_DIR, f"{dataset_name}.npz")

    # Before execution, check if the dataset is in the cache
    if os.path.exists(source_npz_path):
        if not os.path.exists(dest_npz_path) or \
           hashlib.md5(open(source_npz_path, 'rb').read()).hexdigest() != \
           hashlib.md5(open(dest_npz_path, 'rb').read()).hexdigest():
            logging.info(f"Copying cached '{dataset_name}' to {MEDMNIST_DIR}")
            shutil.copy(source_npz_path, dest_npz_path)
        else:
            logging.info(f"Dataset '{dataset_name}' already exists and is up to date in {MEDMNIST_DIR}")
    else:
        logging.info(f"Dataset '{dataset_name}' not found in cache. It will be downloaded.")

    # After execution, the training script is assumed to have downloaded the data if it was missing.
    # We copy it back to our cache for future runs.
    if os.path.exists(dest_npz_path) and not os.path.exists(source_npz_path):
        logging.info(f"Caching downloaded '{dataset_name}' to {cache_dir}")
        shutil.copy(dest_npz_path, source_npz_path)

def handle_weights_cache(model_name: str, cache_dir: str):
    """
    Manages caching of pretrained model weights.
    It syncs weights between a local project cache and the default torch hub cache.
    """
    try:
        from torchvision.models import ResNet18_Weights, ResNet50_Weights
    except ImportError:
        logging.warning("torchvision not found, cannot manage weights cache.")
        return

    model_weights_map = {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1,
    }

    if model_name not in model_weights_map:
        logging.warning(f"No pretrained weights mapping for '{model_name}'. Skipping cache handling.")
        return

    weights = model_weights_map[model_name]
    weights_filename = os.path.basename(weights.url)
    
    # Define paths
    hub_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    source_path = os.path.join(cache_dir, "torch_hub", "checkpoints", weights_filename)
    dest_path = os.path.join(hub_dir, weights_filename)

    # Ensure directories exist
    os.makedirs(os.path.dirname(source_path), exist_ok=True)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Sync from local cache to hub cache
    if os.path.exists(source_path) and not os.path.exists(dest_path):
        logging.info(f"Copying cached weights '{weights_filename}' to torch hub.")
        shutil.copy(source_path, dest_path)

    # After model loading, sync from hub cache to local cache
    # This part is tricky as it should be called after the download.
    # The calling function should handle this. For now, we add a function to call post-download
    
    def sync_weights_to_local_cache():
        if os.path.exists(dest_path) and not os.path.exists(source_path):
            logging.info(f"Caching downloaded weights '{weights_filename}' to local project cache.")
            shutil.copy(dest_path, source_path)
            
    # The initial sync is done. The caller should use the returned function
    # after the potential download.
    return sync_weights_to_local_cache 