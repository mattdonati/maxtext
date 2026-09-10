import multiprocessing

# Apply the JAX monitoring monkey-patch to prevent GCS write conflicts from child processes
_original_run = multiprocessing.Process.run

def _patched_run(self, *args, **kwargs):
    try:
        import jax
        if hasattr(jax, '_src') and hasattr(jax._src, 'monitoring') and hasattr(jax._src.monitoring, '_event_listeners'):
            jax._src.monitoring._event_listeners.clear()
    except Exception:
        pass
    return _original_run(self, *args, **kwargs)

multiprocessing.Process.run = _patched_run

# Delegate to the original MaxText RL entrypoint
from maxtext.trainers.post_train.rl import train_rl
from absl import app

if __name__ == "__main__":
    app.run(train_rl.main)
