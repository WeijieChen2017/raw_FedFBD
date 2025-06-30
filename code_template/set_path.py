import sys
import os

def add_base_path():
    # Add the project's base directory to the Python path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_path not in sys.path:
        sys.path.append(base_path)

add_base_path() 