#!/usr/bin/env python3
import pathlib
import sys

def apply_comprehensive_patch():
    # Define potential paths to types.py
    possible_paths = [
        pathlib.Path("src/maxtext/configs/types.py"),
        pathlib.Path("/app/src/maxtext/configs/types.py"),
    ]
    
    target_path = None
    for path in possible_paths:
        if path.exists():
            target_path = path
            break
            
    if not target_path:
        print("[-] Error: types.py not found in any of the expected locations.")
        sys.exit(1)

    print(f"[+] Found config schema file at: {target_path}")
    content = target_path.read_text()

    # 1. Add SplashAttention to the RLConfig Mixin inheritance list
    old_inheritance = "class RLConfig(\n    LogitsAndLoss,"
    new_inheritance = "class RLConfig(\n    SplashAttention,\n    LogitsAndLoss,"

    # 2. Add MRoPE configuration parameters into the RLConfig class body
    old_fields = '  num_epoch: int = Field(1, ge=1, description="Number of epochs to train for.")'
    new_fields = (
        '  use_mrope: bool = Field(False, description="Enable Multi-dimensional RoPE")\n'
        '  mrope_section: list[int] | None = Field(None, description="MRoPE section config")\n'
        '  reuse_example_batch: int = Field(0, description="For performance testing, repeatedly uses the same batch.")\n'
        '  num_epoch: int = Field(1, ge=1, description="Number of epochs to train for.")'
    )

    if old_inheritance in content:
        content = content.replace(old_inheritance, new_inheritance)
    else:
        print("[-] Warning: Failed to replace inheritance! (Might already be patched)")

    if old_fields in content:
        content = content.replace(old_fields, new_fields)
    else:
        print("[-] Warning: Failed to replace fields! (Might already be patched)")

    target_path.write_text(content)
    print("[+] Comprehensive config patch applied successfully!")

if __name__ == "__main__":
    apply_comprehensive_patch()
