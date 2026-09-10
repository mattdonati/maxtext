import os
import sys

def patch_math_verify_pool():
    filepath = '/app/src/maxtext/trainers/post_train/rl/math_verify_pool.py'
    if not os.path.exists(filepath):
        try:
            import maxtext.trainers.post_train.rl.math_verify_pool as mvp
            filepath = mvp.__file__
        except ImportError:
            print("Could not import maxtext to find math_verify_pool.py")
            return
            
    if not os.path.exists(filepath):
        print(f"Could not find math_verify_pool.py at {filepath}")
        return
    
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace parsing_timeout=None with parsing_timeout=5.0
    content = content.replace("parsing_timeout=None", "parsing_timeout=5.0")
    # Replace timeout_seconds=None with timeout_seconds=5.0
    content = content.replace("timeout_seconds=None", "timeout_seconds=5.0")
    
    # Truncate strings to prevent instant OOM spikes before the timeout can even trigger
    truncate_logic = """
    # Truncate to the last 2000 chars to save the boxed answer but prevent OOM
    predictions = [p[-2000:] if len(p) > 2000 else p for p in predictions]
    extracted_predictions = list("""
    content = content.replace("    extracted_predictions = list(", truncate_logic)
    
    with open(filepath, 'w') as f:
        f.write(content)
        
    print(f"Successfully patched {filepath} to add 5.0s timeouts to SymPy parsing.")

if __name__ == "__main__":
    patch_math_verify_pool()
