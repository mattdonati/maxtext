#!/usr/usr/bin/env python3
import pathlib
import sys

def patch_train_rl():
    possible_paths = [
        pathlib.Path("src/maxtext/trainers/post_train/rl/train_rl.py"),
        pathlib.Path("/app/src/maxtext/trainers/post_train/rl/train_rl.py"),
    ]
    
    target_path = None
    for path in possible_paths:
        if path.exists():
            target_path = path
            break
            
    if not target_path:
        print("[-] Error: train_rl.py not found.")
        return
 
    content = target_path.read_text()
    
    patch_code = """
import multiprocessing
_original_run = multiprocessing.Process.run
def _patched_run(self, *args, **kwargs):
    try:
        import jax
        if hasattr(jax, '_src') and hasattr(jax._src, 'monitoring'):
            jax._src.monitoring._event_listeners.clear()
    except Exception:
        pass
    return _original_run(self, *args, **kwargs)
multiprocessing.Process.run = _patched_run
"""
    
    if "multiprocessing.Process.run = _patched_run" not in content:
        import_marker = "from absl import app"
        content = content.replace(import_marker, import_marker + "\n" + patch_code)
        target_path.write_text(content)
        print("[+] Multiprocessing JAX monitoring patch injected into train_rl.py!")
    else:
        print("[-] Warning: train_rl.py might already be patched!")

if __name__ == "__main__":
    patch_train_rl()
