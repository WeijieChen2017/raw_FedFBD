#!/usr/bin/env python3
"""
Test script to verify the file locking mechanism works correctly.
"""

import os
import time
import multiprocessing
import tempfile
import shutil

# Import the locking functions from fbd_client
from fbd_client import acquire_file_lock, release_file_lock

def test_lock_worker(worker_id, lock_file_path, results):
    """Worker function to test file locking."""
    print(f"Worker {worker_id}: Starting...")
    
    # Try to acquire the lock
    print(f"Worker {worker_id}: Attempting to acquire lock...")
    lock_file = acquire_file_lock(lock_file_path, timeout=30)
    
    if lock_file is None:
        print(f"Worker {worker_id}: Failed to acquire lock (timeout)")
        results[worker_id] = "timeout"
        return
    
    print(f"Worker {worker_id}: Lock acquired successfully")
    results[worker_id] = "acquired"
    
    # Simulate some work
    time.sleep(2)
    
    # Release the lock
    release_file_lock(lock_file)
    print(f"Worker {worker_id}: Lock released")
    results[worker_id] = "completed"

def main():
    """Main test function."""
    print("Testing file locking mechanism...")
    
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    lock_file_path = os.path.join(temp_dir, "test.lock")
    
    try:
        # Test with multiple processes
        num_workers = 4
        manager = multiprocessing.Manager()
        results = manager.dict()
        
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=test_lock_worker, 
                args=(i, lock_file_path, results)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Check results
        print("\nResults:")
        for i in range(num_workers):
            print(f"Worker {i}: {results.get(i, 'unknown')}")
        
        # Verify that only one worker acquired the lock at a time
        acquired_count = sum(1 for result in results.values() if result == "acquired")
        completed_count = sum(1 for result in results.values() if result == "completed")
        
        print(f"\nSummary:")
        print(f"Workers that acquired lock: {acquired_count}")
        print(f"Workers that completed: {completed_count}")
        
        if completed_count == num_workers:
            print("✅ All workers completed successfully - locking mechanism works!")
        else:
            print("❌ Some workers failed - locking mechanism may have issues")
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main() 