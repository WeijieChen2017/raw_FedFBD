#!/usr/bin/env python3
"""
Check M0 model part distribution across clients in each round
"""

import json

def check_m0_distribution():
    """Check which client trains each M0 part in each round"""
    
    # M0 model blocks and their corresponding parts
    m0_blocks = {
        'AFA79': 'in_layer',
        'BFF56': 'layer1', 
        'CAA77': 'layer2',
        'DSC60': 'layer3',
        'EJJ91': 'layer4',
        'FXR03': 'out_layer'
    }
    
    # Load update plan
    with open('config/organamnist_shuffle/update_plan.json', 'r') as f:
        update_plan = json.load(f)
    
    print("M0 Model Part Distribution (Trainable Assignments Only)")
    print("=" * 80)
    print(f"{'Round':<6} {'in_layer':<10} {'layer1':<8} {'layer2':<8} {'layer3':<8} {'layer4':<8} {'out_layer':<10}")
    print("-" * 80)
    
    for round_num in range(1, min(31, len(update_plan) + 1)):
        round_str = str(round_num)
        if round_str not in update_plan:
            continue
            
        round_plan = update_plan[round_str]
        
        # Find which client trains each M0 part
        m0_assignments = {}
        
        for client_id, client_plan in round_plan.items():
            model_to_update = client_plan.get('model_to_update', {})
            
            # Check if this client is training any M0 parts
            for model_part, part_info in model_to_update.items():
                block_id = part_info['block_id']
                if block_id in m0_blocks:
                    m0_assignments[model_part] = client_id
        
        # Print the assignments for this round
        assignments = []
        for part in ['in_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'out_layer']:
            client = m0_assignments.get(part, 'None')
            assignments.append(f"C{client}" if client != 'None' else 'None')
        
        print(f"{round_num:<6} {assignments[0]:<10} {assignments[1]:<8} {assignments[2]:<8} {assignments[3]:<8} {assignments[4]:<8} {assignments[5]:<10}")
    
    # Summary: Check if M0 is always complete
    print("\n" + "=" * 80)
    print("VERIFICATION:")
    
    complete_rounds = 0
    incomplete_rounds = []
    
    for round_num in range(1, min(31, len(update_plan) + 1)):
        round_str = str(round_num)
        if round_str not in update_plan:
            continue
            
        round_plan = update_plan[round_str]
        
        # Count M0 parts assigned in this round
        m0_parts_found = set()
        
        for client_id, client_plan in round_plan.items():
            model_to_update = client_plan.get('model_to_update', {})
            
            for model_part, part_info in model_to_update.items():
                block_id = part_info['block_id']
                if block_id in m0_blocks:
                    m0_parts_found.add(model_part)
        
        if len(m0_parts_found) == 6:
            complete_rounds += 1
        else:
            incomplete_rounds.append(round_num)
    
    print(f"Complete M0 assignments: {complete_rounds}/{min(30, len(update_plan))} rounds")
    if incomplete_rounds:
        print(f"Incomplete rounds: {incomplete_rounds}")
    else:
        print("✅ All rounds have complete M0 model assignments")
    
    # Check for duplicates
    print(f"\nDUPLICATE CHECK:")
    for round_num in [1, 2, 3, 4, 5]:  # Check first 5 rounds
        round_str = str(round_num)
        if round_str not in update_plan:
            continue
            
        round_plan = update_plan[round_str]
        
        # Find all M0 assignments
        m0_client_assignments = []
        
        for client_id, client_plan in round_plan.items():
            model_to_update = client_plan.get('model_to_update', {})
            
            for model_part, part_info in model_to_update.items():
                block_id = part_info['block_id']
                if block_id in m0_blocks:
                    m0_client_assignments.append((model_part, client_id, block_id))
        
        # Check if all M0 parts are assigned to the same client (should be!)
        if m0_client_assignments:
            clients = set(assignment[1] for assignment in m0_client_assignments)
            if len(clients) == 1:
                print(f"  Round {round_num}: M0 complete model assigned to Client {list(clients)[0]} ✅")
            else:
                print(f"  Round {round_num}: M0 parts split across clients {clients} ❌")

if __name__ == "__main__":
    check_m0_distribution()