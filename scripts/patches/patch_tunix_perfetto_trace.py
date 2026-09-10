import os
import sys

def main():
    file_path = "src/maxtext/trainers/post_train/rl/train_rl.py"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        sys.exit(1)
        
    with open(file_path, "r") as f:
        content = f.read()
        
    old_code = """      perf_config = perf_metrics.PerfMetricsConfig()
      perf_config.custom_export_fn = perf_export.PerfMetricsExport.create_metrics_export_fn(cluster_config)"""
      
    new_code = """      perf_config = perf_metrics.PerfMetricsConfig()
      cluster_config.training_config.perf_metrics_options = perf_metrics.PerfMetricsOptions(
          enable_trace_writer=True,
          trace_dir=tensorboard_dir
      )
      perf_config.custom_export_fn = perf_export.PerfMetricsExport.create_metrics_export_fn(cluster_config)"""
      
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, "w") as f:
            f.write(content)
        print("Successfully patched train_rl.py to enable Perfetto tracing.")
    else:
        print("Could not find the target code in train_rl.py to patch.")

if __name__ == "__main__":
    main()
