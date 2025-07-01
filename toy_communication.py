import multiprocessing
import os
import json
import random
import time

num_client = 3
NUM_ROUNDS = 10
OUTPUT_DIR = "fbd_comm"

def client_task(client_id):
    """Client process that actively polls for round-based files."""
    print(f"Client {client_id}: Starting...")
    current_round = 0

    while True:
        # First, check for the shutdown signal
        shutdown_filepath = os.path.join(OUTPUT_DIR, f"last_round_client_{client_id}.json")
        if os.path.exists(shutdown_filepath):
            with open(shutdown_filepath, 'r') as f:
                data = json.load(f)
                if data.get("secret") == -1:
                    print(f"I am finished at client {client_id}")
                    break

        # If no shutdown, look for the file for the current round
        round_filepath = os.path.join(OUTPUT_DIR, f"goods_round_{current_round}_client_{client_id}.json")
        if os.path.exists(round_filepath):
            with open(round_filepath, 'r') as f:
                data = json.load(f)
                secret = data.get("secret")
                print(f"Client {client_id}, Round {current_round}: Found my goods! The secret is {secret}.")
            current_round += 1
        else:
            # Wait before polling again
            time.sleep(0.5)


if __name__ == "__main__":
    # Ensure the output directory exists and is clean for the run
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))
    else:
        os.makedirs(OUTPUT_DIR)
    
    print(f"Server: Starting {NUM_ROUNDS}-round simulation for {num_client} clients.")

    # Start client processes first
    processes = []
    for i in range(num_client):
        process = multiprocessing.Process(target=client_task, args=(i,))
        processes.append(process)
        process.start()

    # Server-side logic: create files for each round
    for r in range(NUM_ROUNDS):
        print(f"Server: --- Round {r} ---")
        for i in range(num_client):
            secret = random.randint(100, 999)
            data = {"secret": secret}
            filepath = os.path.join(OUTPUT_DIR, f"goods_round_{r}_client_{i}.json")
            with open(filepath, 'w') as f:
                json.dump(data, f)
        time.sleep(2)
    
    # After all rounds, send shutdown signal
    print("Server: All rounds complete. Sending shutdown signal.")
    for i in range(num_client):
        filepath = os.path.join(OUTPUT_DIR, f"last_round_client_{i}.json")
        with open(filepath, 'w') as f:
            json.dump({"secret": -1}, f)

    # Wait for all client processes to finish
    for process in processes:
        process.join()

    print("All clients have completed their tasks.")









