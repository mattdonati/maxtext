echo "Running 8b_bf16.sh"
# 16B parameter model.
# This config will work out of the box for any number of v5e-256 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5e/16b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"


# Stop execution if any command exits with error
set -e

export PLATFORM="gce"
export PER_DEVICE_BATCH_SIZE=4
export GLOBAL_PARAMETER_SCALE=8
export INT8_TRAINING=true
# Shards: 0 will use ttf when int8 is true (bf16 when false), else ttt (setting to 1 is basicaly non-local)
export local_aqt_shards_mlp1=0 
export local_aqt_shards_mlp2=0
export local_aqt_shards_query_proj=0
export local_aqt_shards_key_proj=0
export local_aqt_shards_value_proj=0
export local_aqt_shards_attention_out_proj=0

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

# Train
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=5 per_device_batch_size=$PER_DEVICE_BATCH_SIZE enable_checkpointing=false\
    enable_profiler=true remat_policy=full global_parameter_scale=$GLOBAL_PARAMETER_SCALE\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic attention='flash' gcs_metrics=true int8_training=$INT8_TRAINING\
    local_aqt_shards_mlp1=$local_aqt_shards_mlp1\
    local_aqt_shards_mlp2=$local_aqt_shards_mlp2\
    local_aqt_shards_query_proj=$local_aqt_shards_query_proj\
    local_aqt_shards_key_proj=$local_aqt_shards_key_proj\
    local_aqt_shards_value_proj=$local_aqt_shards_value_proj\
    local_aqt_shards_attention_out_proj=$local_aqt_shards_attention_out_proj\