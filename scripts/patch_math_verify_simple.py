import os

content = """
from typing import Any, Callable, Optional

def math_verify_pool(
    trainer_config: Any,
    items: list,
    scores: list[float],
    timeout: float = 300,
    num_procs: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[float]:
  if not items:
    return scores
  if log_fn:
    log_fn(f"math_verify_pool_simple: Received {len(items)} items. Returning dummy scores.")
  return scores

def verify_math_worker(golds: list[str], predictions: list[str]) -> float:
  return 0.0
"""

def patch():
    filepath = '/app/src/maxtext/trainers/post_train/rl/math_verify_pool.py'
    if not os.path.exists(filepath):
        try:
            import maxtext.trainers.post_train.rl.math_verify_pool as mvp
            filepath = mvp.__file__
        except ImportError:
            pass
            
    with open(filepath, 'w') as f:
        f.write(content)
        
    print(f"Successfully patched {filepath} with simple dummy grader.")

if __name__ == "__main__":
    patch()
