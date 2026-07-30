import ast
import os
from collections import defaultdict

def scan_for_duplicates():
    func_registry = defaultdict(list)
    var_registry = defaultdict(list)
    
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and environments
        if any(part.startswith('.') for part in root.split(os.sep)): 
            continue
            
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=filepath)
                        
                        for node in tree.body:
                            # Catch top-level functions
                            if isinstance(node, ast.FunctionDef):
                                func_registry[node.name].append(filepath)
                            
                            # Catch top-level constants (all-caps variables)
                            elif isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name) and target.id.isupper():
                                        var_registry[target.id].append(filepath)
                except Exception as e:
                    print(f"Could not parse {filepath}: {e}")

    print("--- DUPLICATED FUNCTIONS ---")
    for name, paths in func_registry.items():
        unique_paths = set(paths)
        if len(unique_paths) > 1:
            print(f"'{name}' found in: {', '.join(unique_paths)}")

    print("\n--- DUPLICATED CONSTANTS ---")
    for name, paths in var_registry.items():
        unique_paths = set(paths)
        if len(unique_paths) > 1:
            print(f"'{name}' found in: {', '.join(unique_paths)}")

if __name__ == "__main__":
    scan_for_duplicates()
