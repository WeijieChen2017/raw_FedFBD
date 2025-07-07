# please load siim_fbd_fold_0.json

import json
import os

with open('siim_fbd_fold_0.json', 'r') as f:
    data = json.load(f)

original_data_dir = "../../siim-101"
dup_data_dir = "siim-101"
os.makedirs(dup_data_dir, exist_ok=True)

commands = []

copy_data_keys = ['train', 'val', 'test']

for copy_data_key in copy_data_keys:
    for client in data[copy_data_key]:
        for item in data[copy_data_key][client]:
            # new image path is to add the dup_data_dir to the image path
            old_image_path = f"{original_data_dir}/{item['image']}"
            old_label_path = f"{original_data_dir}/{item['label']}"
            new_image_path = f"{dup_data_dir}/{item['image']}"
            new_label_path = f"{dup_data_dir}/{item['label']}"
            commands.append(f"cp {old_image_path} {new_image_path}")
            commands.append(f"cp {old_label_path} {new_label_path}")


# start a task report and wait for user to approve
print(f"We are about to copy the dataset using {len(commands)} commands.")
print(f"The first 10 commands are:")
for command in commands[:10]:
    print(command)
print()

# wait for user to approve
user_input = input("Press Enter to continue..., or type 'n' to exit")
if user_input == 'n':
    exit()
else:
    print("Copying dataset...")

for command in commands:
    # before copying, check if the the target folder exists
    target_folder = command.split(" ")[2]
    print(f"Checking if {target_folder} exists...")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        print(f"Created folder {target_folder}")
    os.system(command)