#!/usr/bin/env python3
"""
Simple test script to verify the --fold argument parsing works.
"""
import argparse

def test_fold_argument():
    """Test that the fold argument is properly parsed"""
    parser = argparse.ArgumentParser(description="Test fold argument")
    parser.add_argument("--fold", type=int, default=None, help="Fold number for cross-validation (0-3). If not specified, uses original dataset loading.")
    parser.add_argument("--experiment_name", type=str, default="siim", help="Name of the experiment.")
    
    # Test different fold values
    test_cases = [
        ["--fold", "0"],
        ["--fold", "1"],  
        ["--fold", "2"],
        ["--fold", "3"],
        []  # No fold argument
    ]
    
    for i, test_args in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_args if test_args else 'No fold argument'}")
        try:
            args = parser.parse_args(test_args)
            print(f"  fold: {args.fold}")
            print(f"  experiment_name: {args.experiment_name}")
            
            # Test fold validation
            if args.fold is not None:
                if args.fold not in [0, 1, 2, 3]:
                    print(f"  ERROR: Invalid fold value {args.fold}")
                else:
                    print(f"  SUCCESS: Valid fold value {args.fold}")
            else:
                print(f"  SUCCESS: No fold specified (original loading)")
                
        except SystemExit:
            print(f"  ERROR: Failed to parse arguments")
            
    # Test invalid fold value
    print(f"\nTest case: Invalid fold value")
    try:
        args = parser.parse_args(["--fold", "5"])
        if args.fold not in [0, 1, 2, 3]:
            print(f"  ERROR: Invalid fold value {args.fold} should be rejected")
    except SystemExit:
        print(f"  SUCCESS: Invalid fold value rejected by parser")

if __name__ == "__main__":
    test_fold_argument() 