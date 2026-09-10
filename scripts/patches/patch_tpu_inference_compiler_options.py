import re
import sys
import importlib.util
from pathlib import Path

def patch_compiler_options():
    # Attempt to find the package path. The file could be in `tpu_commons` or `tpu_inference`.
    base_dir = None
    for pkg in ["tpu_commons", "tpu_inference"]:
        spec = importlib.util.find_spec(pkg)
        if spec is not None and spec.origin is not None:
            base_dir = Path(spec.origin).parent
            break
    else:
        print("Could not find the target package in either tpu_commons or tpu_inference.")
        sys.exit(1)

    print(f"Found target package at: {base_dir}")
    
    # Regex to remove compiler_options=...
    # This safely matches both flat dictionaries {...} and function calls get_step_fn_compiler_options()
    pattern = r'compiler_options\s*=\s*(?:get_step_fn_compiler_options\(\)|\{[^}]+\})\s*,?'
    
    total_patched = 0
    for target_file in base_dir.rglob('*.py'):
        if target_file.name == 'decode_loop.py':
            print(f"Skipping {target_file} to preserve top-level compiler_options.")
            continue
            
        content = target_file.read_text()
        
        new_content, count = re.subn(pattern, '', content)
        
        if count > 0:
            target_file.write_text(new_content)
            print(f"Successfully removed {count} instance(s) of `compiler_options` from {target_file}.")
            total_patched += count
            
    if total_patched == 0:
        print("No `compiler_options` found in any files to patch. It might have already been patched.")
        for target_file in base_dir.rglob('*.py'):
            if 'compiler_options=' in target_file.read_text():
                print(f"WARNING: 'compiler_options=' still found in {target_file} but didn't match the regex pattern!")

if __name__ == "__main__":
    patch_compiler_options()
