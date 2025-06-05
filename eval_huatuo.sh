#!/bin/bash

conda activate vllm
# python eval_huatuo.py --num_samples -1
# python eval_huatuo.py --num_samples -1 --dataset PMC-VQA
# python eval_huatuo.py --num_samples -1 --dataset VQA-RAD
# python eval_huatuo.py --num_samples -1 --dataset OmniMedVQA
# python eval_huatuo.py --num_samples -1 --dataset PVQA
# python eval_huatuo.py --num_samples -1 --dataset SLAKE
python eval_huatuo.py --num_samples -1 --dataset MeCoVQA_region