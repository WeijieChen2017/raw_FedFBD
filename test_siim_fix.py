#!/usr/bin/env python3
"""
Test script to verify SIIM INFO fixes
"""

print("Testing SIIM INFO fixes...")

try:
    # Test client import
    from fbd_client_siim import SIIM_INFO
    print("✓ fbd_client_siim imports successfully")
    print(f"  SIIM_INFO contains: {list(SIIM_INFO.keys())}")
    
    # Test server import  
    from fbd_server_siim import SIIM_INFO as server_siim_info
    print("✓ fbd_server_siim imports successfully")
    print(f"  Server SIIM_INFO contains: {list(server_siim_info.keys())}")
    
    # Test main import
    from fbd_main_siim import SIIM_INFO as main_siim_info
    print("✓ fbd_main_siim imports successfully")
    print(f"  Main SIIM_INFO contains: {list(main_siim_info.keys())}")
    
    # Test that SIIM info is properly defined
    siim_info = SIIM_INFO['siim']
    print(f"✓ SIIM info: task={siim_info['task']}, channels={siim_info['n_channels']}")
    
    print("\n🎉 All SIIM INFO fixes are working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()