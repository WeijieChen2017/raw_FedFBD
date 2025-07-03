#!/usr/bin/env python3
"""
FBD Plan Generation Script

This script generates the shipping, request, and update plans for FBD training
based on the configuration settings in a specified experiment's fbd_settings.json.
"""

import json
import os
import argparse
from collections import defaultdict

def load_experiment_settings(experiment_name):
    """Loads the fbd_settings.json for a given experiment."""
    config_path = os.path.join("config", experiment_name, "fbd_settings.json")
    print(f"DEBUG: Loading FBD settings from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Settings file not found for experiment '{experiment_name}' at: {config_path}")
    with open(config_path, 'r') as f:
        settings = json.load(f)
    print(f"DEBUG: Loaded settings for experiment: {experiment_name}")
    return settings

def generate_model_to_blocks_mapping(settings):
    """Generate mapping from model colors to their block IDs, ordered by layer."""
    FBD_TRACE = settings["FBD_TRACE"]
    FBD_INFO = settings["FBD_INFO"]
    MODEL_PART_ORDER = settings["MODEL_PART_ORDER"]
    
    part_rank = {p: i for i, p in enumerate(MODEL_PART_ORDER)}
    
    model_to_blocks = {mid: [] for mid in FBD_INFO["models"]}
    for bid, info in FBD_TRACE.items():
        model_to_blocks[info['color']].append(bid)
    
    for blist in model_to_blocks.values():
        blist.sort(key=lambda b: part_rank[FBD_TRACE[b]['model_part']])
    
    return model_to_blocks

def generate_plans(settings):
    """Generate shipping, request, and update plans for all rounds."""
    FBD_TRACE = settings["FBD_TRACE"]
    FBD_INFO = settings["FBD_INFO"]
    MODEL_PARTS = settings["MODEL_PARTS"]
    UPDATE_SCHEDULE = settings["UPDATE_SCHEDULE"]

    for stage in UPDATE_SCHEDULE:
        if not stage.get("model_part_to_update") and stage.get("model_part_index_to_update"):
            stage["model_part_to_update"] = [MODEL_PARTS[i] for i in stage["model_part_index_to_update"]]
    
    model_to_blocks = generate_model_to_blocks_mapping(settings)
    
    shipping_plan = defaultdict(dict)
    request_plan = defaultdict(dict)
    update_plan = defaultdict(dict)
    
    round_to_parts_to_update = {}
    current_round = 1
    for stage in UPDATE_SCHEDULE:
        parts_to_update = stage["model_part_to_update"]
        for _ in range(stage["num_rounds"]):
            round_to_parts_to_update[current_round] = parts_to_update
            current_round += 1
    
    total_rounds = len(round_to_parts_to_update)
    print(f"DEBUG: Generated {total_rounds} rounds from UPDATE_SCHEDULE")
    print(f"DEBUG: round_to_parts_to_update keys: {list(round_to_parts_to_update.keys())}")
    print(f"DEBUG: Available keys in FBD_INFO['training_plan']: {list(FBD_INFO['training_plan'].keys())}")
    
    # Check if outer_rounds_total exists in settings root
    if 'OUTER_ROUNDS_TOTAL' in settings:
        print(f"DEBUG: OUTER_ROUNDS_TOTAL from fbd_settings: {settings['OUTER_ROUNDS_TOTAL']}")
    else:
        print("DEBUG: OUTER_ROUNDS_TOTAL not found in settings")
    
    for outer_round in range(total_rounds):
        sched_idx = outer_round % FBD_INFO["training_plan"]["rounds"]
        schedule = FBD_INFO["training_plan"]["schedule"][str(sched_idx)]
        
        round_num = outer_round + 1
        parts_to_update_this_round = round_to_parts_to_update[round_num]
        
        for client, active_model in schedule.items():
            client_models = FBD_INFO["clients"][client]
            
            all_client_blocks = []
            for model_id in client_models:
                all_client_blocks.extend(model_to_blocks[model_id])
            
            shipping_plan[round_num][client] = all_client_blocks
            request_plan[round_num][client] = all_client_blocks
            
            update_plan[round_num][client] = generate_update_plan(
                FBD_TRACE, active_model, client_models, model_to_blocks, parts_to_update_this_round
            )
    
    return shipping_plan, request_plan, update_plan

def generate_update_plan(fbd_trace, active_model, client_models, model_to_blocks, parts_to_update):
    model_to_update = {}
    active_model_blocks = model_to_blocks[active_model]
    
    for block_id in active_model_blocks:
        model_part = fbd_trace[block_id]['model_part']
        status = "trainable" if model_part in parts_to_update else "frozen"
        model_to_update[model_part] = {"block_id": block_id, "status": status}
    
    regularizer_models = []
    for model_id in client_models:
        if model_id != active_model:
            regularizer_model = {}
            regularizer_blocks = model_to_blocks[model_id]
            
            for block_id in regularizer_blocks:
                model_part = fbd_trace[block_id]['model_part']
                regularizer_model[model_part] = block_id
            
            regularizer_models.append(regularizer_model)
    
    return {
        "model_to_update": model_to_update,
        "model_as_regularizer": regularizer_models
    }

def save_plans_to_json(experiment_name, shipping_plan, request_plan, update_plan):
    """Save all plans to JSON files in the specified experiment's config directory."""
    output_dir = os.path.join("config", experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plans = {
        "shipping_plan.json": {int(k): v for k, v in shipping_plan.items()},
        "request_plan.json": {int(k): v for k, v in request_plan.items()},
        "update_plan.json": {int(k): v for k, v in update_plan.items()}
    }
    
    for filename, plan_data in plans.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(plan_data, f, indent=2)
        print(f"  ✓ Saved {filepath}")

def main():
    """Main function to generate and save all FBD plans for a given experiment."""
    parser = argparse.ArgumentParser(description="Generate FBD plans for a specific experiment.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment (e.g., 'bloodmnist').")
    args = parser.parse_args()

    try:
        print(f"Generating FBD plans for experiment: {args.experiment_name}")
        
        settings = load_experiment_settings(args.experiment_name)
        
        shipping_plan, request_plan, update_plan = generate_plans(settings)
        
        save_plans_to_json(args.experiment_name, shipping_plan, request_plan, update_plan)
        
        print(f"\n✅ Successfully generated FBD plans for {args.experiment_name}!")
        
    except Exception as e:
        print(f"❌ Error generating plans: {e}")
        raise 

if __name__ == "__main__":
    main() 