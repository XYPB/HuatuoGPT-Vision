import torch
import json
import os
import datetime
from copy import deepcopy
from cli import HuatuoChatbot
from tqdm import tqdm

import argparse

from preprocess_eval_datasets import (
    parse_pmc_vqa_to_multi_choice_conversations,
    parse_mecovqa_json_to_conversations,
    parse_rad_vqa_json_to_conversations,
    parse_omnimedvqa_jsons,
    parse_pvqa_to_conversations,
    parse_slake_json_to_conversations
)

parser = argparse.ArgumentParser(description="Evaluate VLLM models on MeCoVQA dataset")
parser.add_argument("--dataset", type=str, default="MeCoVQA", help="Dataset to evaluate on (default: MeCoVQA)")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (default: 10)")
parser.add_argument("--temperature", type=float, default=0, help="Temperature for model inference (default: 0)")

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)


def save_outputs_to_json(outputs, filename, output_dir="./runs/output", model_info=None):
    """
    Save model outputs to a JSON file.
    
    Args:
        outputs (list): List of model outputs
        filename (str): Name of the output JSON file
        output_dir (str): Directory to save the outputs
        model_info (dict, optional): Additional model information to include
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the output file
    output_path = os.path.join(output_dir, filename)
    
    # Create a results dictionary with metadata
    results = {
        "outputs": outputs,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(outputs)
    }
    
    # Add model info if provided
    if model_info:
        results.update({"model_info": model_info})
    
    # Save the outputs
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Outputs saved to {output_path}")
    return output_path


def eval_huatuogpt(conversations, gts):
    bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7B")
    outputs = []

    for idx, messages in tqdm(enumerate(conversations), total=len(conversations), desc="Evaluating HuatuoGPT"):
        image_path = messages[1]['content'][1]['image']

        system_prompt = messages[0]["content"][0]["text"]
        question = messages[1]['content'][0]['text']
        message = f"INSTRUCTION: {system_prompt}\n\nQUESTION: {question}\n\nANSWER:"

        with torch.inference_mode():
            response = bot.inference(
                message, [image_path]
            )
            response = response[0].strip()

            output = {
                "id": image_path,
                "input": messages[1]["content"][0]['text'],
                "output": response,
                "gt": gts[idx] if idx < len(gts) else None
            }
            outputs.append(output)
    return outputs


if __name__ == "__main__":
    args = parser.parse_args()
    # Create a timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    size = "full" if args.num_samples <= 0 else args.num_samples
    output_dir = os.path.join("./runs/output", f"eval_{timestamp}_{args.dataset}_{size}_huatuogpt")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == "PMC-VQA":
        data_path = "./data/PMC-VQA/test_2.csv"
        _, _, conversations, gts = parse_pmc_vqa_to_multi_choice_conversations(data_path)
    elif args.dataset == "MeCoVQA":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Complex_VQA_test.json'
        conversations, gts = parse_mecovqa_json_to_conversations(data_path)
    elif args.dataset == "VQA-RAD":
        data_path = './data/VQA_RAD/VQA_RAD Dataset Public.json'
        conversations, gts = parse_rad_vqa_json_to_conversations(data_path)
    elif args.dataset == "OmniMedVQA":
        data_path = './data/OmniMedVQA/QA_information/Open-access/'
        conversations, gts = parse_omnimedvqa_jsons(data_path)
    elif args.dataset == "PVQA":
        data_path = './data/pvqa/qas/test_vqa.pkl'
        conversations, gts = parse_pvqa_to_conversations(data_path)
    elif args.dataset == "SLAKE":
        data_path = './data/SLAKE/test.json'
        conversations, gts = parse_slake_json_to_conversations(data_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are: PMC-VQA, MeCoVQA, VQA-RAD, OmniMedVQA.")
    
    # Number of samples to evaluate
    num_samples = args.num_samples if args.num_samples > 0 else len(conversations)
    # Save evaluation configuration

    model_config = {
        "name": "HuatuoGPT-Vision-7B",
        "processing": "Sequential"
    }
    config = {
        "timestamp": timestamp,
        "dataset": data_path,
        "num_samples": num_samples,
        "models": model_config
    }
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Evaluate HuatuoGPT
    print(f"Evaluating HuatuoGPT on {num_samples} samples...")
    HuatuoGPT_outputs = eval_huatuogpt(deepcopy(conversations)[:num_samples], gts[:num_samples])
    HuatuoGPT_model_info = {
        "model_name": "google/HuatuoGPT-4b-it",
        "model_type": "Image-Text-to-Text",
        "batch_size": "N/A (Sequential processing)"
    }
    HuatuoGPT_output_path = save_outputs_to_json(
        HuatuoGPT_outputs, 
        "HuatuoGPT_outputs.json", 
        output_dir,
        HuatuoGPT_model_info
    )
    
    print(f"\nAll outputs saved to: {output_dir}")

