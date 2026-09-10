import importlib.util
import os
import sys

p = "/usr/local/lib/python3.12/site-packages/tunix/rl/trainer.py"
if not os.path.exists(p):
    print(f"Warning: {p} not found. Skipping patch.")
    sys.exit(0)

print(f"Target file: {p}")
with open(p, "r", encoding="utf-8") as f:
    c = f.read()

target = "    assert self._buffered_train_metrics is not None"
replacement = (
    "    if self._buffered_train_metrics is None:\n"
    "      from tunix.sft.peft_trainer import MetricsBuffer\n"
    "      self._buffered_train_metrics = MetricsBuffer(step=getattr(self, '_train_steps', 0), losses=[0.0])\n"
)

if target in c:
    c = c.replace(target, replacement)
    with open(p, "w", encoding="utf-8") as f:
        f.write(c)
    print("Hotfix successfully applied to trainer.py.")
else:
    print("Target not found in trainer.py.")
