# Steps
get baseline of runs with smaller batch size for smaller topology 
enable profiles & and get the metrics (e.g. rollout time, etc. )
  maybe use perfetto to get the percentage time for everything 
enable the agentic learner and check improvement 


# checkout 

## tunix async document notes 

## tunix perf metrics

## build for image
https://github.com/AI-Hypercomputer/maxtext/pull/4184/files#diff-de563ce0a02335898f4a4e1bca2fc074614122bfce30b980d7065142806e4defR45

## TPU-GPU RL Parity



DESTINATION_IMAGE="gcr.io/${PROJECT_ID}/rl-image:clean-v4"
export CLUSTER_NAME="${CLUSTER_NAME}"
export PROJECT_ID="${PROJECT_ID}"
export ZONE="${ZONE:-us-central1-c}"


gcloud container images add-tag \
  ${SOURCE_IMAGE} \
  ${DESTINATION_IMAGE}

export DOCKER_IMAGE="${DESTINATION_IMAGE}"
export WORKLOAD_NAME="pathways-headless"
export CLUSTER_NAME="${CLUSTER_NAME}"
export PROJECT_ID="${PROJECT_ID}"
export ZONE="${ZONE:-us-central1-c}"
export TPU_TYPE="tpu7x-16"
export NUM_SLICES=1

source ~/VENVS/ubench/bin/activate

kubectl set image deployment/dev-pod vllm-tpu=${DOCKER_IMAGE}

# gs://${BUCKET_NAME}/checkpoints/qwen3-0.6b-pathways/0/items

xpk cluster create-pathways \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-8 \
  --num-slices=4 \
  --spot

xpk cluster create-pathways \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-16 \
  --num-slices=1 \
  --spot

xpk workload create-pathways \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --tpu-type="${TPU_TYPE}" \
  --zone="${ZONE}" \
  --num-slices=1 \
  --base-docker-image="${DOCKER_IMAGE}" \
  --server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20260623-jax_0.10.1" \
  --proxy-server-image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260623-jax_0.10.1" \
  --workload="${WORKLOAD_NAME}" \
  --custom-pathways-proxy-server-args="${XLA_FLAGS}" \
  --custom-pathways-server-args="" \
  --command="sleep infinity"


# start the cluster in headless mode 
export WORKLOAD_NAME="headless-pathways"
xpk workload create-pathways \
  --headless \
  --tpu-type=${TPU_TYPE} \
  --workload=${WORKLOAD_NAME} \
  --num-slices=${NUM_SLICES} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --cluster=${CLUSTER_NAME}


# TensorBoard export
# tensorboard_upload.sh \
#   --experiment_name="rl-qwen-16k" \
#   --root_dir="gs://${BUCKET_NAME}/maxtext-logs/tensorboard/"


# Build dependency image 
cd maxtext/
bash src/dependencies/scripts/docker_build_dependency_image.sh WORKFLOW=post-training DEVICE=tpu

# 1. Define the full Artifact Registry path
export CLOUD_IMAGE_NAME="gcr.io/${PROJECT_ID}/maxtext-rl:latest"

# 2. Upload the image 
# upload_maxtext_docker_image CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
# don't need the maxtext env activated when doing this way. 
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# update cpu node pool: 
# c4-standard-144

gcloud container node-pools update ${NODE_POOL_NAME:-cpu-np} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION:-us-central1} \
    --machine-type=c4-highmem-96


# patched settings that I removed 
rollout_vllm_init_with_random_weights=True 
  don't want to init vllm with model weights when you will be overwriting them anyway 
torch.backends.cuda.enable_memory_efficient_sdpa(True)
compute_logps_micro_batch_size=1
  defaults to batch size in trajectories if not set explicitly 

# additional settings: 
decode_sampling_top_k=0 # set to 0 to go to -1 for vllm which means no topk. 

# docker images 
## copy image from one project to another 
# Set your destination variables (example using GCR)
export DESTINATION_IMAGE="gcr.io/${PROJECT_ID}/maxtext-rl-ironwood:latest"

# Add the tag to the new location
gcloud container images add-tag \
  ${SOURCE_IMAGE} \
  ${DESTINATION_IMAGE}
# The add-tag command performs a read-only operation on the source project.
# It reads the source image data and makes an independent tag in your project (${PROJECT_ID}).

export BUCKET="gs://${BUCKET_NAME}/"
gcloud storage buckets add-iam-policy-binding ${BUCKET} \
    --member="user:${COLLEAGUE_EMAIL}" \
    --role="roles/storage.objectAdmin" \
    --project=${PROJECT_ID}


Here is exactly how that timeout setting operates in the codebase logic.

The goal in a single-turn setup using the queue is to force the system to wait until exactly X prompts (your batch size) have arrived before sending them to the TPU. You do not want the queue to send a partial, inefficient batch just because a few milliseconds passed.

In vllm_async_driver.py, the code checks if it should flush the queue early using this specific condition:

```python
if (self._submission_timeout_s > 0 and 
    time.perf_counter() - self._submission_window_start >= self._submission_timeout_s):
    return True # Flush the partial batch early
```
Based on this logic, there are two ways to prevent the queue from flushing early:

1. Set it to 0.0 If you set rl.rollout_vllm_server_mode_submission_timeout_s=0.0, the condition self._submission_timeout_s > 0 evaluates to False. The timeout feature is completely disabled. The queue will wait indefinitely until the exact number of prompts defined by submission_threshold is reached.

2. Set it extremely high If you set it to 3600.0 (1 hour), the condition > 0 is met, but the time delta will not trigger for an hour. The queue will effectively wait indefinitely for the exact number of prompts to arrive, achieving the exact same result.