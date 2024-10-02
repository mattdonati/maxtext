

export LIBTPU_INIT_ARGS="--xla_tpu_enable_host_aware_passes=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"


python3 MaxText/train.py MaxText/configs/base.yml run_name=runner_$(date +%Y-%m-%d-%H-%M-%S) base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset steps=10 enable_checkpointing=false dataset_type=synthetic gradient_clipping_threshold=0