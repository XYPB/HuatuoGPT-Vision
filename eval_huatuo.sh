#!/bin/bash

set -e
set -x
source activate base
conda activate vllm
# python eval_huatuo.py --num_samples -1
# python eval_huatuo.py --num_samples -1 --dataset PMC-VQA
# python eval_huatuo.py --num_samples -1 --dataset VQA-RAD
# python eval_huatuo.py --num_samples -1 --dataset OmniMedVQA
# python eval_huatuo.py --num_samples -1 --dataset PVQA
# python eval_huatuo.py --num_samples -1 --dataset SLAKE
# export CUDA_VISIBLE_DEVICES=2,3; python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region

export CUDA_VISIBLE_DEVICES=0
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region --bbox_coord
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region --side_by_side
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region --skip_region

# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region_yn --skip_region
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region_yn --side_by_side
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region_yn --bbox_coord

python eval_huatuo.py --num_samples -1 --dataset medsynth_no_region
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region_yn_hard --skip_region
# python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region_yn_hard --side_by_side