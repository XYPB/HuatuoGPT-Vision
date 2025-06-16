import torch
import json
import os
import datetime
from copy import deepcopy
from cli import HuatuoChatbot
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

import argparse

from preprocess_eval_datasets import (
    parse_pmc_vqa_to_multi_choice_conversations,
    parse_mecovqa_json_to_conversations,
    parse_rad_vqa_json_to_conversations,
    parse_omnimedvqa_jsons,
    parse_pvqa_to_conversations,
    parse_slake_json_to_conversations,
    parse_mecovqa_region_json_to_conversations
)

parser = argparse.ArgumentParser(description="Evaluate VLLM models on MeCoVQA dataset")
parser.add_argument("--dataset", type=str, default="MeCoVQA", help="Dataset to evaluate on (default: MeCoVQA)")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (default: 10)")
parser.add_argument("--temperature", type=float, default=0, help="Temperature for model inference (default: 0)")
parser.add_argument("--bbox_coord", action='store_true', help="Use bounding box coordinates for models that support it (default: False)")
parser.add_argument("--side_by_side", action='store_true', help="Use side-by-side mask visualization for models that support it (default: False)")
parser.add_argument("--skip_region", action='store_true', help="Skip region highlighting in the image (default: False)")

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

def highlight_region(image, mask, alpha=0.25):
    """
    Highlight a specific mask in the image by drawing a transparent overlay around it.
    """
    # Both image and mask should be PIL Images
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2_mask = np.array(mask)

    # Ensure mask is binary
    if cv2_mask.ndim == 3:
        cv2_mask = cv2_mask[:, :, 0]
    cv2_mask = (cv2_mask > 0).astype(np.uint8) * 255
    # Create a colored overlay
    color = (0, 0, 255)
    overlay = np.zeros_like(cv2_image, dtype=np.uint8)
    overlay[cv2_mask > 0] = color
    # Blend the overlay with the original image
    highlighted_image = cv2.addWeighted(cv2_image, 1 - alpha, overlay, alpha, 0)
    return Image.fromarray(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))

def highlight_region_bbox(image, mask, width=5, color=(0, 0, 255)):
    """
    Highlight a bounding box in the image by drawing a rectangle around it.
    """
    # Convert image to numpy array
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert mask to bbox coordinates
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    x1 = np.min(np.where(mask > 0)[1])
    x2 = np.max(np.where(mask > 0)[1])
    y1 = np.min(np.where(mask > 0)[0])
    y2 = np.max(np.where(mask > 0)[0])

    # Draw the rectangle on the image
    cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, width)

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def mask_side_by_side(image, mask):
    """
    Combine the original image and the mask side by side.
    """
    # Convert image and mask to numpy arrays
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2_mask = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)

    # Ensure mask is binary
    # if cv2_mask.ndim == 3:
    #     cv2_mask = cv2_mask[:, :, 0]
    cv2_mask = (cv2_mask > 0).astype(np.uint8) * 255

    # Create a side-by-side image
    combined_image = np.hstack((cv2_image, cv2_mask))
    
    return Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

def mask_as_bbox(mask):
    image_width, image_height = mask.size
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    x1 = np.min(np.where(mask > 0)[1])
    x2 = np.max(np.where(mask > 0)[1])
    y1 = np.min(np.where(mask > 0)[0])
    y2 = np.max(np.where(mask > 0)[0])
    bbox = (x1, y1, x2, y2)
    return normalize_coordinates(bbox, image_width, image_height)

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


def eval_huatuogpt(conversations, gts, use_region_bbox=False, side_by_side=False, skip_region=False):
    bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7B", device="cuda")
    outputs = []

    for idx, messages in tqdm(enumerate(conversations), total=len(conversations), desc="Evaluating HuatuoGPT"):
        image_path = messages[1]['content'][1]['image']
        image = Image.open(image_path).convert('RGB')

        bbox = None
        if len(messages[1]['content']) > 2 and messages[1]['content'][2]['type'] == 'region':
            region_mask_path = messages[1]['content'][2]['region']
            region_mask = Image.open(region_mask_path).convert('L')
            if skip_region:
                bbox = None
            elif use_region_bbox:
                bbox = mask_as_bbox(region_mask)
            elif side_by_side:
                image = mask_side_by_side(image, region_mask)
            else:
                image = highlight_region_bbox(image, region_mask)
        image = image.resize((448, 448))  # Resize to 448x448 for HuatuoGPT

        system_prompt = messages[0]["content"][0]["text"]
        question = messages[1]['content'][0]['text']

        if bbox:
            question += f"\nRegion coordinates: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            system_prompt += "\nRegion of interest are provided as coordinates in the format x1,y1,x2,y2, where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner. The coordinates are normalized to a scale of 0 to 1000, where 1000 corresponds to the full width or height of the image."
        elif side_by_side:
            system_prompt += "\nThe image is shown side by side with the mask. Please answer based on the image and the mask."

        message = f"INSTRUCTION: {system_prompt}\n\nQUESTION: {question}\n\nANSWER:"

        with torch.inference_mode():
            response = bot.inference(
                message, [image]
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
    elif args.dataset == "MeCoVQA_region":
        data_path = 'data/MeCoVQA/test/MeCoVQA_Region_VQA_test.json'
        conversations, gts = parse_mecovqa_region_json_to_conversations(data_path)
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
    HuatuoGPT_outputs = eval_huatuogpt(deepcopy(conversations)[:num_samples], gts[:num_samples], use_region_bbox=args.bbox_coord, side_by_side=args.side_by_side, skip_region=args.skip_region)
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

