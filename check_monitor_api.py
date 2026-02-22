import requests
import json

def check_monitor():
    try:
        print("[*] Checking Trinity Peak API...")
        response = requests.get("http://127.0.0.1:8000/api/stats")
        if response.status_code == 200:
            print("[SUCCESS] Monitor Backend is LIVE.")
            print(f"[*] Response: {str(response.json())[:200]}...")
        else:
            print(f"[FAILED] Monitor returned status: {response.status_code}")
    except Exception as e:
        print(f"[FAILED] Could not connect: {e}")

if __name__ == "__main__":
    check_monitor()
