# Pathways cluster configuration

export CLUSTER_NAME="${CLUSTER_NAME:-my-pathways-cluster}"
export CLUSTER_NODEPOOL_COUNT="1"
export TPU_TYPE="tpu7x-32"
# export PW_CPU_MACHINE_TYPE="c4-highcpu-144"
PW_CPU_MACHINE_TYPE="c4-standard-96"
# export DEFAULT_CPU_MACHINE_TYPE=n2-standard-16
export REGION="${REGION:-us-central1}"
export PROJECT="${PROJECT_ID:?PROJECT_ID must be set}"
export BUCKET="${BUCKET:-gs://${PROJECT}-bucket}"
export ZONE="${ZONE:-us-central1-c}"
export BASE_OUTPUT_DIRECTORY="${BUCKET}/maxtext_rl"
export CLOUD_IMAGE_NAME="gcr.io/${PROJECT}/maxtextrl:latest"
export DOCKER_IMAGE="gcr.io/${PROJECT}/maxtextrl:latest"
export WORKLOAD_NAME="rl-workload"
export HF_TOKEN=""
export MAXTEXT_CKPT_PATH="${BUCKET}/checkpoints/qwen3-0.6b-pathways/0/items"


XLA_FLAGS="--xla_tpu_dvfs_p_state=7 \
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

xpk cluster create-pathways --num-slices=1 --tpu-type=${TPU_TYPE} --flex --project=${PROJECT} --zone=${ZONE} --cluster=${CLUSTER_NAME} --pathways-gce-machine-type=${PW_CPU_MACHINE_TYPE}  


xpk workload create-pathways \
--cluster=${CLUSTER_NAME} \
--project=${PROJECT} \
--zone=${ZONE} \
--max-restarts=0 \
--tpu-type=${TPU_TYPE} \
--num-slices=1 \
--docker-image="${DOCKER_IMAGE}" \
--workload="${WORKLOAD_NAME}" \
--custom-pathways-proxy-server-args="${XLA_FLAGS}" \
--command="sleep infinity"



# Cluster update to get larger nodes 
# update cpu node pool: 
# c4-standard-144

gcloud container node-pools update ${NODE_POOL_NAME:-cpu-np} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --machine-type=c4-highmem-96