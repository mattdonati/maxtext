#!/usr/bin/env python3
import pathlib
import sys
import importlib.util

def apply_patch():
    # Find tunix installation path dynamically
    spec = importlib.util.find_spec("tunix")
    if spec is None or spec.origin is None:
        print("[-] Error: tunix module not found in the Python path.")
        # Maybe it's installed via source in a different directory?
        possible_paths = [
            pathlib.Path("/app/tunix/tunix/rl/agentic/pipeline/rollout_orchestrator.py"),
            pathlib.Path("/usr/local/lib/python3.10/site-packages/tunix/rl/agentic/pipeline/rollout_orchestrator.py"),
        ]
        target_path = None
        for p in possible_paths:
            if p.exists():
                target_path = p
                break
        if not target_path:
            print("[-] Could not locate rollout_orchestrator.py")
            sys.exit(1)
    else:
        tunix_dir = pathlib.Path(spec.origin).parent
        target_path = tunix_dir / "rl" / "agentic" / "pipeline" / "rollout_orchestrator.py"
    
    if not target_path.exists():
        print(f"[-] Error: {target_path} not found.")
        sys.exit(1)

    print(f"[+] Found rollout_orchestrator.py at: {target_path}")
    content = target_path.read_text()

    old_code = (
        "      # Parallel execution for the group\n"
        "      self._rollout_sync_lock.acquire_rollout()\n"
        "      try:"
    )
    
    new_code = (
        "      # Parallel execution for the group\n"
        "      loop = asyncio.get_running_loop()\n"
        "      await loop.run_in_executor(None, self._rollout_sync_lock.acquire_rollout)\n"
        "      try:"
    )

    if old_code in content:
        content = content.replace(old_code, new_code)
        target_path.write_text(content)
        print("[+] Deadlock patch applied successfully!")
    elif new_code in content:
        print("[+] Patch is already applied.")
    else:
        print("[-] Warning: Failed to find exact target code block! Attempting fallback replace.")
        
        # We need to make sure we don't accidentally replace something else, but it's a unique string.
        old_code_fallback = "self._rollout_sync_lock.acquire_rollout()"
        new_code_fallback = "loop = asyncio.get_running_loop()\n      await loop.run_in_executor(None, self._rollout_sync_lock.acquire_rollout)"
        
        if old_code_fallback in content:
            content = content.replace(old_code_fallback, new_code_fallback)
            target_path.write_text(content)
            print("[+] Applied fallback patch successfully!")
        else:
            print("[-] Error: Could not find the lock acquisition call in the file.")
            sys.exit(1)

if __name__ == "__main__":
    apply_patch()
