#!/usr/bin/env python
import sys
import os
import pprint

def main():
    """Print information about the Python environment and path."""
    print("Python Version:", sys.version)
    print("\nPython Path:")
    pprint.pprint(sys.path)
    
    print("\nCurrent Working Directory:", os.getcwd())
    
    print("\nDirectory Listing:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"[dir] {item}/")
        else:
            print(f"[file] {item}")
    
    # Check if glitch_art directory exists
    if os.path.exists('glitch_art'):
        print("\nglitch_art/ Directory Listing:")
        for item in os.listdir('glitch_art'):
            if os.path.isdir(os.path.join('glitch_art', item)):
                print(f"[dir] {item}/")
            else:
                print(f"[file] {item}")
    
    # Check specific module import
    try:
        import glitch_art
        print("\nSuccessfully imported glitch_art")
        print("glitch_art path:", glitch_art.__file__)
    except ImportError as e:
        print("\nFailed to import glitch_art:", e)

if __name__ == "__main__":
    main() 