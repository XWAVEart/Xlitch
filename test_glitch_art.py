#!/usr/bin/env python3
"""
Simple test script to verify the glitch_art package is working correctly.
"""
import os
from PIL import Image
import sys

def test_imports():
    """Test that we can import from the glitch_art package."""
    print("Testing imports...")
    try:
        from glitch_art import load_image, resize_image_if_needed, generate_output_filename
        from glitch_art.core.pixel_attributes import PixelAttributes
        print("✅ Basic imports work!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pixel_attributes():
    """Test the PixelAttributes class."""
    print("Testing PixelAttributes...")
    try:
        from glitch_art.core.pixel_attributes import PixelAttributes
        
        # Test with a simple red pixel
        red_pixel = (255, 0, 0)
        brightness = PixelAttributes.brightness(red_pixel)
        hue = PixelAttributes.hue(red_pixel)
        
        print(f"  Red pixel brightness: {brightness:.2f}")
        print(f"  Red pixel hue: {hue:.2f}")
        
        # Test for expected values
        if brightness == 0.299 * 255 and abs(hue) < 30:
            print("✅ PixelAttributes calculations work!")
            return True
        else:
            print(f"❌ Unexpected values: brightness={brightness}, hue={hue}")
            return False
    except Exception as e:
        print(f"❌ Error testing PixelAttributes: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("Running glitch_art package tests...")
    
    tests = [
        test_imports,
        test_pixel_attributes,
    ]
    
    results = [test() for test in tests]
    
    if all(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 