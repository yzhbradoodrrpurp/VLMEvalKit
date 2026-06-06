#!/usr/bin/env bash

export CUDA_HOME="${CUDA_HOME:-/root/miniconda3/envs/verl/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export FLA_DISABLE_BACKEND_DISPATCH="${FLA_DISABLE_BACKEND_DISPATCH:-1}"

export LMUData="${LMUData:-/root/autodl-fs/LMUData}"
export PRED_FORMAT="${PRED_FORMAT:-tsv}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
