"""Hotfix to correct compute_logps_micro_batch_size in Tunix AgenticGrpoLearner."""

import importlib.util
import os
import sys

print("Applying hotfix to tunix/rl/agentic/agentic_grpo_learner.py...")

# Find the file without importing it to avoid dependency issues during setup
spec = importlib.util.find_spec("tunix.rl.agentic.agentic_grpo_learner")
if spec and spec.origin:
    p = spec.origin
else:
    # Fallback to standard pip installation path
    p = "/usr/local/lib/python3.12/site-packages/tunix/rl/agentic/agentic_grpo_learner.py"

if not os.path.exists(p):
    print(f"Warning: {p} not found. Skipping patch.")
    sys.exit(0)

print(f"Target file: {p}")

with open(p, "r", encoding="utf-8") as f:
    c = f.read()

target1 = (
    "    actor_mesh = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR]\n"
    "    have_actor_mesh = actor_mesh is not None and not actor_mesh.empty\n"
    "    rollout_per_token_logps = None"
)

replacement1 = (
    "    actor_mesh = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR]\n"
    "    have_actor_mesh = actor_mesh is not None and not actor_mesh.empty\n"
    "\n"
    "    configured_compute_logps = self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size\n"
    "    compute_logps_micro_batch_size = (\n"
    "        configured_compute_logps * self.algo_config.num_generations\n"
    "        if configured_compute_logps\n"
    "        else len(trajectories)\n"
    "    )\n"
    "\n"
    "    rollout_per_token_logps = None"
)

target2 = "micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,"
replacement2 = "micro_batch_size=compute_logps_micro_batch_size,"

modified = False
if target1 in c:
    c = c.replace(target1, replacement1)
    modified = True
else:
    if "compute_logps_micro_batch_size =" in c:
        print("Hotfix part 1 already applied.")
    else:
        print("Warning: target 1 string not found.")

if target2 in c:
    c = c.replace(target2, replacement2)
    modified = True
else:
    if "micro_batch_size=compute_logps_micro_batch_size," in c:
        print("Hotfix part 2 already applied.")
    else:
        print("Warning: target 2 string not found.")

if modified:
    with open(p, "w", encoding="utf-8") as f:
        f.write(c)
    print("Hotfix successfully applied to agentic_grpo_learner.py.")
else:
    print("No changes made to agentic_grpo_learner.py.")
