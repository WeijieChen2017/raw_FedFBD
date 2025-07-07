#!/usr/bin/env python3
"""
Generate human-readable training plans for each model (M0-M5)
Creates txt files showing which client trains each model part in each round
"""

import json

def generate_human_readable_plans():
    """Generate human-readable training plans for M0-M5"""
    
    # Model color to name mapping
    model_mapping = {
        'M0': ['AFA79', 'BFF56', 'CAA77', 'DSC60', 'EJJ91', 'FXR03'],
        'M1': ['AKY64', 'BVD88', 'CGV29', 'DQM27', 'EVZ66', 'FPC91'],
        'M2': ['ALT34', 'BVP97', 'CNF57', 'DWJ41', 'EGO46', 'FBI78'],
        'M3': ['AOC39', 'BWW19', 'COO30', 'DHK75', 'EYT34', 'FGM06'],
        'M4': ['ASN90', 'BXG86', 'CPM83', 'DPU42', 'EVN11', 'FWC09'],
        'M5': ['AUV29', 'BYM04', 'CRZ52', 'DPX98', 'EVN36', 'FSY05']
    }
    
    # Block ID to model part mapping
    block_to_part = {
        # M0
        'AFA79': 'in_layer', 'BFF56': 'layer1', 'CAA77': 'layer2', 
        'DSC60': 'layer3', 'EJJ91': 'layer4', 'FXR03': 'out_layer',
        # M1
        'AKY64': 'in_layer', 'BVD88': 'layer1', 'CGV29': 'layer2',
        'DQM27': 'layer3', 'EVZ66': 'layer4', 'FPC91': 'out_layer',
        # M2
        'ALT34': 'in_layer', 'BVP97': 'layer1', 'CNF57': 'layer2',
        'DWJ41': 'layer3', 'EGO46': 'layer4', 'FBI78': 'out_layer',
        # M3
        'AOC39': 'in_layer', 'BWW19': 'layer1', 'COO30': 'layer2',
        'DHK75': 'layer3', 'EYT34': 'layer4', 'FGM06': 'out_layer',
        # M4
        'ASN90': 'in_layer', 'BXG86': 'layer1', 'CPM83': 'layer2',
        'DPU42': 'layer3', 'EVN11': 'layer4', 'FWC09': 'out_layer',
        # M5
        'AUV29': 'in_layer', 'BYM04': 'layer1', 'CRZ52': 'layer2',
        'DPX98': 'layer3', 'EVN36': 'layer4', 'FSY05': 'out_layer'
    }
    
    # Load update plan
    with open('config/organamnist_shuffle/update_plan.json', 'r') as f:
        update_plan = json.load(f)
    
    # Generate human-readable plans for each model
    for model_name, block_ids in model_mapping.items():
        print(f"Generating training plan for {model_name}...")
        
        # Create content for this model
        content = []
        
        # Header row with model parts
        header = "in_layer | layer1 | layer2 | layer3 | layer4 | out_layer"
        content.append(header)
        
        # Generate 30 rounds of training assignments
        for round_num in range(1, 31):
            round_str = str(round_num)
            
            if round_str not in update_plan:
                # If round doesn't exist, fill with "None"
                row = "None | None | None | None | None | None"
                content.append(row)
                continue
            
            round_plan = update_plan[round_str]
            
            # Find which client trains each part of this model
            part_assignments = {}
            
            for client_id, client_plan in round_plan.items():
                model_to_update = client_plan.get('model_to_update', {})
                
                # Check if this client is training any blocks of the current model
                for key, part_info in model_to_update.items():
                    block_id = part_info['block_id']
                    if block_id in block_ids:
                        # Get the actual model part from the block
                        model_part = block_to_part[block_id]
                        part_assignments[model_part] = f"C{client_id}"
            
            # Build the row for this round
            parts_order = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
            row_parts = []
            for part in parts_order:
                client = part_assignments.get(part, 'None')
                row_parts.append(client)
            
            row = " | ".join(row_parts)
            content.append(row)
        
        # Save to file
        filename = f"config/organamnist_shuffle/training_plan_human_{model_name}.txt"
        with open(filename, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"✅ Saved {filename}")
    
    print(f"\n🎉 Generated human-readable training plans for all models (M0-M5)!")
    print(f"📁 Files saved in: config/organamnist_shuffle/")

if __name__ == "__main__":
    generate_human_readable_plans()