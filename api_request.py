import requests

base_url = "http://localhost:23129"  # Update with your server's URL

def test_translate_endpoint(text, target_lang="en", source_lang=None):
    url = f"{base_url}/v1/lang/translate"
    payload = {
        "target_lang": target_lang,
        "text": text,
    }
    
    # Only include source_lang if it's provided
    if source_lang:
        payload["source_lang"] = source_lang

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"Translation Successful:\n{result}")
        return result
    else:
        print(f"Translation Failed. Status Code: {response.status_code}\nDetails: {response.text}")
        return None

# Test Case 1: Auto-detect language and translate to English
print("Testing auto-detection:")
test_translate_endpoint(text="Hola, ¿cómo estás?", target_lang="en")

# Test Case 2: Specify source language
print("\nTesting with specified source language:")
test_translate_endpoint(source_lang="zh", target_lang="en", text="你好")

# Test Case 3: List supported languages
def list_supported_languages():
    url = f"{base_url}/v1/lang/support"
    response = requests.get(url)
    if response.status_code == 200:
        languages = response.json()
        print(f"Supported languages: {len(languages)} languages")
        for lang in languages[:5]:  # Show just the first 5 for brevity
            print(f"  - {lang['code']}: {lang['name']} ({lang['full_code']})")
        print("  ... and more")
    else:
        print(f"Failed to get languages. Status Code: {response.status_code}")

print("\nGetting supported languages:")
list_supported_languages()

