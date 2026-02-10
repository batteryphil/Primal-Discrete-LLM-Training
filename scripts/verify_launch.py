import requests
import sys

REPO = "batteryphil/Trinity-1.58bit-Prime-Harmonic-LLM-Evolution"
TAG = "v1.0.0"
FILE = "trinity_1.58bit_packed.bin"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{FILE}"

print(f"Checking Release Artifact: {URL}...")
try:
    response = requests.head(URL, allow_redirects=True)
    if response.status_code == 200:
        print("✅ SUCCESS: Binary is publicly downloadable.")
        print(f"   Size: {response.headers.get('content-length', 'Unknown')} bytes")
    else:
        print(f"❌ FAILURE: Received status code {response.status_code}")
        print("   Note: Ensure the release is 'Published' and not just a 'Draft'.")
except Exception as e:
    print(f"❌ ERROR: Could not connect to GitHub: {e}")
