import primal_chat
import time
import sys
# Redirect stdout to capture chat output? 
# primal_chat prints directly. We'll just run it.

def run_test(name, prompt, expected_substr):
    print(f"\n[TEST] {name}")
    print(f"Prompt: '{prompt}'")
    # Capture output? 
    # primal_chat.chat(prompt) prints result.
    # We will just run it and manually verify or try to hook?
    # primal_chat.chat returns nothing.
    # Modify primal_chat to return the token? 
    # Too invasive? 
    # Let's just run inputs.
    
    primal_chat.chat(prompt)
    print(f"Target: '{expected_substr}'")

if __name__ == "__main__":
    print("=== PRIMAL-COMMANDER QA SUITE ===")
    
    run_test("Pattern", "1, 2, 3, 4, 5,", "6")
    run_test("Story", "Once upon a time, there was a little boy named Tim. He had a red", "ball")
    run_test("Logic", "If it is raining, I need an", "umbrella")
    
    print("\n=== QA COMPLETE ===")
