#!/usr/bin/env python3
"""
Test script for the RunPod serverless handler.
This simulates RunPod's serverless request format to test the handler locally.
"""

import os
import json
from handler import handler

def test_translation(text, target_lang, source_lang=None):
    """Test the translation with the handler."""
    # Create a mock event in RunPod format
    event = {
        "id": "test-request-id",
        "input": {
            "text": text,
            "target_lang": target_lang
        }
    }
    
    # Add source_lang if provided
    if source_lang:
        event["input"]["source_lang"] = source_lang
    
    print(f"\n=== Testing Translation ===")
    print(f"Input: '{text}' ({source_lang or 'auto-detect'} → {target_lang})")
    
    # Call the handler function
    try:
        result = handler(event)
        
        # Print results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success!")
            print(f"Source language: {result['source_lang']} {'(detected)' if result.get('detected') else ''}")
            print(f"Translation: '{result['translated_text']}'")
            
        return result
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {"error": str(e)}

def main():
    """Run several test cases to verify the handler works correctly."""
    # Test 1: Auto-detect source language (Spanish to English)
    test_translation("Hola mundo, ¿cómo estás hoy?", "en")
    
    # Test 2: Specify source language (English to French)
    test_translation("Hello world, how are you today?", "fr", "en")
    
    # Test 3: Chinese to English
    test_translation("你好，世界", "en", "zh")
    
    # Test 4: Invalid target language
    test_translation("Hello world", "invalid_lang")
    
    # Test 5: Empty text
    test_translation("", "en")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 