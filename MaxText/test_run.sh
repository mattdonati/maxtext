#!/bin/bash



# We also run pre-training, this is similar to the finetuning command except we don't pass any checkpoint directory to load parameters from
python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=${HOME}/ckpt dataset_type=synthetic \
hardware=cpu tokenizer_path=assets/tokenizer_llama3.tiktoken per_device_batch_size=1 \
run_name=test_run steps=10 enable_checkpointing=True max_target_length=256 \
base_emb_dim=128 \
base_num_query_heads=1 \
base_num_kv_heads=1 \
base_mlp_dim=128 \
base_num_decoder_layers=1 \
head_dim=128 \
