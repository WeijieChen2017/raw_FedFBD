#!/usr/bin/env python3
"""
Generate client-side view of training plans showing which model each part comes from
Creates txt files showing what model parts each client trains in each round
"""

import json

def generate_client_view_plans():
    """Generate client-side training plans showing model sources for each part"""
    
    # Block ID to model mapping
    block_to_model = {
        # M0
        'AFA79': 'M0', 'BFF56': 'M0', 'CAA77': 'M0', 'DSC60': 'M0', 'EJJ91': 'M0', 'FXR03': 'M0',
        # M1
        'AKY64': 'M1', 'BVD88': 'M1', 'CGV29': 'M1', 'DQM27': 'M1', 'EVZ66': 'M1', 'FPC91': 'M1',
        # M2
        'ALT34': 'M2', 'BVP97': 'M2', 'CNF57': 'M2', 'DWJ41': 'M2', 'EGO46': 'M2', 'FBI78': 'M2',
        # M3
        'AOC39': 'M3', 'BWW19': 'M3', 'COO30': 'M3', 'DHK75': 'M3', 'EYT34': 'M3', 'FGM06': 'M3',
        # M4
        'ASN90': 'M4', 'BXG86': 'M4', 'CPM83': 'M4', 'DPU42': 'M4', 'EVN11': 'M4', 'FWC09': 'M4',
        # M5
        'AUV29': 'M5', 'BYM04': 'M5', 'CRZ52': 'M5', 'DPX98': 'M5', 'EVN36': 'M5', 'FSY05': 'M5'
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
    
    # Generate client-side plans for each client (C0-C5)
    for client_id in range(6):
        print(f"Generating client view for C{client_id}...")
        
        # Create content for this client
        content = []
        
        # Header row with model parts
        header = "in_layer | layer1 | layer2 | layer3 | layer4 | out_layer"
        content.append(header)
        
        # Generate 30 rounds of training assignments from this client's perspective
        for round_num in range(1, 31):
            round_str = str(round_num)
            
            if round_str not in update_plan:
                # If round doesn't exist, fill with "None"
                row = "None | None | None | None | None | None"
                content.append(row)
                continue
            
            round_plan = update_plan[round_str]
            client_plan = round_plan.get(str(client_id), {})
            model_to_update = client_plan.get('model_to_update', {})
            
            # Find what model parts this client is training and from which models
            part_sources = {}
            
            for key, part_info in model_to_update.items():
                block_id = part_info['block_id']
                model_name = block_to_model[block_id]
                model_part = block_to_part[block_id]
                
                # Handle multiple blocks with same model_part by keeping them separate
                if model_part in part_sources:
                    # If we already have this part, create a list or append to existing list
                    if isinstance(part_sources[model_part], list):
                        part_sources[model_part].append(model_name)
                    else:
                        part_sources[model_part] = [part_sources[model_part], model_name]
                else:
                    part_sources[model_part] = model_name
            
            # Build the row for this round showing model sources
            parts_order = ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']
            row_parts = []
            for part in parts_order:
                model_source = part_sources.get(part, 'None')
                # If there are multiple models for this part, show them all
                if isinstance(model_source, list):
                    model_source = '+'.join(sorted(model_source))
                row_parts.append(model_source)
            
            row = " | ".join(row_parts)
            content.append(row)
        
        # Save to file
        filename = f"config/organamnist_shuffle/client_view_C{client_id}.txt"
        with open(filename, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"✅ Saved {filename}")
    
    print(f"\n🎉 Generated client-side training plans for all clients (C0-C5)!")
    print(f"📁 Files saved in: config/organamnist_shuffle/")

if __name__ == "__main__":
    generate_client_view_plans()