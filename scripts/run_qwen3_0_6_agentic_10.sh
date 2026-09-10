#!/bin/bash
set -e

# export WORKLOAD_NAME="qwen3-0805-aserve-full"
# rpa kernel errored 

export PROJECT_ID="${PROJECT_ID:?PROJECT_ID must be set}"
export CLUSTER_NAME="${CLUSTER_NAME:?CLUSTER_NAME must be set}"
export ZONE="${ZONE:-us-central1-c}"
export BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://${PROJECT_ID}-maxtext-logs/}"
export MAXTEXT_CKPT_PATH="${MAXTEXT_CKPT_PATH:?MAXTEXT_CKPT_PATH must be set (e.g. gs://${PROJECT_ID}-bucket/checkpoints/qwen3-0.6b-pathways/0/items)}"
export WORKLOAD_NAME="${WORKLOAD_NAME:-qwen3-0807-rpa}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-gcr.io/${PROJECT_ID}/maxtext-rl-ironwood:latest}"
export TPU_TYPE="${TPU_TYPE:-tpu7x-16}"
export NUM_SLICES="${NUM_SLICES:-1}"

# XLA Flags (Truncated for brevity, keep yours intact)
export XLA_FLAGS="--xla_tpu_dvfs_p_state=7 \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_num_sparse_cores_for_gather_offloading=1 \
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_use_tc_device_shape_on_sc=True \
--xla_sc_disable_megacore_partitioning=True \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
--xla_enable_async_all_gather=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
--xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
--xla_tpu_enable_sparse_core_collective_aggregator=true \
--xla_tpu_enable_latency_hiding_layer_scheduler=true \
--xla_tpu_scheduler_percent_shared_memory_limit=150 \
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
--xla_tpu_pcie_bandwidth_multiplier=0.03 \
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
--xla_tpu_enable_3d_reduce_scatter_decomposer=false"

# ==============================================================================
# 1. FIX APPLIED: Use single quotes to protect the JSON from bash stripping
# ==============================================================================
export MAXTEXT_COMMAND="HF_TOKEN='' \
SKIP_JAX_PRECOMPILE=1 \
JAX_RANDOM_WEIGHTS=1 \
NEW_MODEL_DESIGN=1 \
TPU_MIN_LOG_LEVEL=0 \
TF_CPP_MIN_LOG_LEVEL=0 \
TPU_STDERR_LOG_LEVEL=0 \
JAX_PLATFORMS=proxy,cpu \
NUM_PRECOMPILE_WORKERS=1 \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
ENABLE_PATHWAYS_PERSISTENCE=1 \
GRR_RPA_DECODE_BLOCKS=\"1,16384,1,4096\" \
PYTHONPATH=/app/src \
python3 -m maxtext.trainers.post_train.rl.train_rl \
model_name=qwen3-0.6b \
tokenizer_path=Qwen/Qwen3-0.6B \
run_name=$WORKLOAD_NAME \
checkpoint_storage_use_ocdbt=False \
async_scheduling=true \
base_output_directory=$BASE_OUTPUT_DIRECTORY \
chips_per_vm=8 \
num_batches=1 \
num_test_batches=0 \
rl.use_agentic_rollout=true \
rl.off_policy_steps=1 \
rl.num_generations=8 \
rl.grpo_beta=0.05 \
rl.grpo_epsilon=0.2 \
rl.epsilon_high=null \
gradient_clipping_threshold=1.0 \
decode_sampling_temperature=0.8 \
decode_sampling_top_k=50 \
decode_sampling_nucleus_p=0.95 \
dataset_name=nvidia/OpenMathInstruct-2 \
remat_policy=save_dot_except_mlp \
attention=flash \
hf_train_files=hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet \
train_split=train_1M \
max_target_length=24576 \
max_prefill_predict_length=16384 \
learning_rate=1e-6 \
batch_size=15 \
train_micro_batch_size=1 \
rollout_micro_batch_size=1 \
rollout_data_parallelism=8 \
rollout_tensor_parallelism=1 \
enable_dp_attention=false \
hbm_utilization_vllm=0.6 \
max_num_seqs=8 \
max_num_batched_tokens=24832 \
scan_layers=True \
allow_split_physical_axes=True \
enable_tunix_perf_metrics=True \
checkpoint_period=100 \
max_num_checkpoints_to_keep=1000 \
enable_checkpointing=false \
load_parameters_path=$MAXTEXT_CKPT_PATH \
rl.max_concurrency=8 \
rl.return_logprobs=False \
rl.use_rollout_logps=False \
rl.kv_cache_metrics=True \
rl.disable_log_stats=True \
vllm_additional_config='{\"enable_continue_decode\":true,\"max_decode_steps\":128}'"

# rl.rollout_vllm_server_mode_submission_threshold=8 \
# rl.rollout_vllm_server_mode_submission_timeout_s=5 \

# ==============================================================================
# 2. CONFIGURATION: SET RESOURCE LIMITS
# ==============================================================================
export JAX_TPU_CPU="56"
export JAX_TPU_MEM="400G"
export PROXY_CPU="16"
export PROXY_MEM="128G"
export RM_CPU="16"
export RM_MEM="128G"

echo ">>> Pre-building custom Docker image with local workspace..."
export CUSTOM_IMAGE="gcr.io/${PROJECT_ID}/${WORKLOAD_NAME}-$(date +%s)"
cat <<EOF > Dockerfile.tmp
FROM ${DOCKER_IMAGE}
WORKDIR /app
COPY . /app
EOF
docker build -t ${CUSTOM_IMAGE} -f Dockerfile.tmp .
docker push ${CUSTOM_IMAGE}
rm Dockerfile.tmp

# ==============================================================================
# 3. DEFINE YOUR XPK COMMAND AS A BASH ARRAY (Removes the need for `eval`)
# ==============================================================================
export RAW_MANIFEST="raw_manifest.yaml"
export MODIFIED_MANIFEST="modified_manifest.yaml"

XPK_CMD=(
  xpk workload create-pathways
  --cluster="${CLUSTER_NAME}"
  --project="${PROJECT_ID}"
  --tpu-type="${TPU_TYPE}"
  --zone="${ZONE}"
  --num-slices="${NUM_SLICES}"
  --base-docker-image="${CUSTOM_IMAGE}"
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260730-jax_0.10.2"
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260730-jax_0.10.2"
  --workload="${WORKLOAD_NAME}"
  --custom-pathways-proxy-server-args="${XLA_FLAGS}"
  --custom-pathways-server-args=""
  --command="pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git --no-deps; python3 scripts/patches/patch_rl_config.py; python3 scripts/patches/patch_tpu_inference_compiler_options.py; python3 scripts/patches/patch_rollout_orchestrator_deadlock.py; python3 scripts/patches/patch_tunix_perfetto_trace.py; ${MAXTEXT_COMMAND}"
)

echo ">>> Running XPK in dry-run mode to generate the raw manifest..."
"${XPK_CMD[@]}" --dry-run --output-manifest-file="${RAW_MANIFEST}"

echo ">>> Modifying container resource limits in the manifest..."
# ==============================================================================
# 4. STREAMLINED PYTHON MODIFIER
# ==============================================================================
python3 - << 'EOF'
import yaml, os

# 1. Define all targets cleanly in one dictionary
targets = {
    'jax-tpu': (os.environ['JAX_TPU_CPU'], os.environ['JAX_TPU_MEM'], os.environ.get('CUSTOM_IMAGE')),
    'pathways-proxy': (os.environ['PROXY_CPU'], os.environ['PROXY_MEM'], None),
    'pathways-rm': (os.environ['RM_CPU'], os.environ['RM_MEM'], None)
}

docs = list(yaml.safe_load_all(open(os.environ['RAW_MANIFEST'])))

# 2. Iterate and apply changes
for doc in docs:
    if doc and doc.get('kind') == 'JobSet':
        for rjob in doc.get('spec', {}).get('replicatedJobs', []):
            pod_spec = rjob.get('template', {}).get('spec', {}).get('template', {}).get('spec', {})
            
            # Combine main and init containers to loop them together
            all_containers = pod_spec.get('containers', []) + pod_spec.get('initContainers', [])
            
            for c in all_containers:
                if c.get('name') in targets:
                    cpu, mem, img = targets[c['name']]
                    
                    # Safely fetch or create 'resources', then 'limits' and 'requests'
                    res = c.setdefault('resources', {})
                    res.setdefault('limits', {}).update({'cpu': cpu, 'memory': mem})
                    res.setdefault('requests', {}).update({'cpu': cpu, 'memory': mem})
                    
                    if img: c['image'] = img

with open(os.environ['MODIFIED_MANIFEST'], 'w') as f:
    yaml.safe_dump_all(docs, f, sort_keys=False)
EOF


echo ">>> Successfully updated manifest. Submitting workload to cluster..."
kubectl delete jobset $WORKLOAD_NAME || true
kubectl apply -f $MODIFIED_MANIFEST
