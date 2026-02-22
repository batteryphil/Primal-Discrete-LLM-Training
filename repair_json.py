import json
import os
import re

def repair_json(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return
    
    print(f"Attempting to repair and sanitize {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()
        
        # Strategy: Find all JSON blocks and merge them
        blocks = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        all_items = []
        
        for b in blocks:
            try:
                # Wrap each block in [] and parse
                # Using a custom decoder to handle NaN if it exists in the raw string
                # actually json.load handles it, but we want to turn it into None
                block_content = "[" + b + "]"
                # Replace raw NaN with null before parsing
                block_content = block_content.replace('NaN', 'null').replace('Infinity', 'null')
                items = json.loads(block_content)
                if isinstance(items, list):
                    all_items.extend(items)
            except Exception as e:
                continue
        
        if all_items:
            # Sort by step
            try:
                all_items.sort(key=lambda x: x.get('step', 0))
            except: pass
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_items, f)
            print(f"Successfully repaired {filename} with {len(all_items)} entries.")
        else:
            print(f"No valid JSON blocks found in {filename}.")
            
    except Exception as e:
        print(f"Critical error repairing {filename}: {e}")

if __name__ == "__main__":
    repair_json('stats_project_real.json')
    repair_json('samples_project_real.json')
    repair_json('perplexity_project_real.json')
