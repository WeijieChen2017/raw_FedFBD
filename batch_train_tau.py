import os

alpha = [0.1, 0.25, 0.5, 1.0]

for a in alpha:
    command = f"python3 generate_hetero_datasets.py --dataset dermamnist --alpha {a}"
    print("running: ", command)
    os.system(command)
    command = f"python fbd_main_tau.py --experiment dermamnist --model_flag resnet18 --reg none --FedAvg --save_affix _alpha_{a}"
    print("running: ", command)
    os.system(command)