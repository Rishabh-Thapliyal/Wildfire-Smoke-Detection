import torch
import re
import os
from tqdm import tqdm
import warnings
from PIL import Image, ImageDraw
from IPython.display import display
import requests
from PIL import Image
import numpy as np
import time
import pickle

import logging

import logging
from datetime import datetime
import sys

def setup_logger():
    """Configure a comprehensive logger that overwrites files and includes timestamps"""
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Configure basic settings
    logger.setLevel(logging.DEBUG)  # Set to lowest level
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (overwrites existing file)
    file_handler = logging.FileHandler('logs_val_3.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO+ in console
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Initial log message
    logger.info(f"=== New session started at {datetime.now()} ===")
    
    return logger

# Initialize logger
logger = setup_logger()


logger.info(torch.__version__)

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore")

os.sys.path.append("/expanse/lustre/projects/ddp464/mhnguyen/data")

model_dict = { "ov_chat" :"llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
"ov" : "llava-hf/llava-onevision-qwen2-7b-ov-hf",
"si" : "llava-hf/llava-onevision-qwen2-7b-si-hf"
}

# =========== select the model =========

model_name = list(model_dict.keys())[0]

# ========================================

# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_dict[model_name],
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
# ).to(device = 'cuda')

# processor = AutoProcessor.from_pretrained(model_dict[model_name])


def draw_grid_on_image(image, grid_size=(3, 3)):
    """
    Draws a grid on the image.

    :param image: PIL Image object
    :param grid_size: Tuple (rows, cols) indicating the grid dimensions
    :return: PIL Image object with grid lines drawn
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    grid_width = width // grid_size[1]
    grid_height = height // grid_size[0]

    # Draw vertical lines
    for i in range(1, grid_size[1]):
        x = i * grid_width
        draw.line((x, 0, x, height), fill="red", width=2)

    # Draw horizontal lines
    for i in range(1, grid_size[0]):
        y = i * grid_height
        draw.line((0, y, width, y), fill="red", width=2)

    return image


def parse_result(output_text):
    """
    Parses the model's output to extract "Yes" or "No".

    :param output_text: The model's output text
    :return: "Yes" or "No" based on the output
    """
    # Use regex to find "Yes" or "No" in the output
    match = re.search(r'\b(Yes|No)\b', output_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "No"  # Default to "No" if no match is found


def parse_assistant_response(output_text):
    """
    Parses the model's output to extract only the assistant's response.

    :param output_text: The model's output text
    :return: The assistant's response (starting from "Result: Yes" or "Result: No")
    """
    # Find the start of the assistant's response
    assistant_start = output_text.find("assistant")
    if assistant_start == -1:
        return output_text  # If "assistant" is not found, return the entire output

    # Extract the assistant's response
    assistant_response = output_text[assistant_start:].split("assistant")[-1].strip()
    return assistant_response


rows, cols = 4, 4

prompt = f'''
        Is there a smoke plume in the image? 
        Answer with "Result: Yes" or "Result: No" at the beginning of your response.
        If yes, in which grid do you see the smoke plume? 
        The image is divided into a {rows}x{cols} grid. 
        Identify the grid by its row and column (e.g., Row 1, Column 2).
        How confident are you on a scale of 0 to 10?
        Note: Ignore any text present in the image.
        '''

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{prompt}"},
            {"type": "image"},
        ],
    },
]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Path to the directory containing images
# image_dir = "data/20160722_FIRE_mw-e-mobo-c"
with open('val_fires_final.txt', 'r') as file:
    val_fire_names = [line.strip() for line in file if line.strip()]

fire_scores = {}
fires_covered = 0

start_time = time.time()

len_fires = {}
for fire in tqdm(os.listdir("/expanse/lustre/projects/ddp464/mhnguyen/data"), desc='processing images...'):

    if fire in val_fire_names:
        # print(fire)
        
        image_dir = f"/expanse/lustre/projects/ddp464/mhnguyen/data/{fire}"

        # Get all files and sort them alphabetically (start with negative sample and then positives...)
        image_files = sorted([f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        len_fires[fire] = {"length": len(image_files),
                           "starting_image": image_files[0],
                           "last_image": image_files[-1],
                           "last_image_path": os.path.join(image_dir, image_files[-1])}

with open("fire_len.pickle", "wb") as file:  # 'wb' = write in binary mode
    pickle.dump(len_fires, file)

        # if len(image_files)!=81:
        #     logger.info(f"=============== {fire}, {len(image_files)} ===============")
        #     logger.info(f"starting image: {image_files[0]}, last image: {image_files[-1]}")
        #     image_path = os.path.join(image_dir, image_files[-1])
        #     logger.info(f"=============== {image_path} ===============")
        #     with Image.open(image_path) as img:
                
        #         # gridded_image = draw_grid_on_image(img, grid_size=(rows, cols))
        #         display(img)

#         tp = 0
#         fp = 0
#         fn = 0
#         time_to_detect = 0
#         predictions = list()

#         for image_name in image_files:
#             image_path = os.path.join(image_dir, image_name)

#             # Determine target_label based on image name
#             if '-' in image_name:
#                 target_label = 0
#             elif '+' in image_name:
#                 target_label = 1
#             else:
#                 target_label = -1  # default case if neither - nor + is found

#             try:
#                 # Open the image using PIL
#                 with Image.open(image_path) as img:

#                     gridded_image = draw_grid_on_image(img, grid_size=(rows, cols))
#                     # display(gridded_image)

#                     inputs = processor(images=gridded_image, text=prompt, return_tensors='pt').to('cuda', torch.float32)
#                     output = model.generate(**inputs, max_new_tokens=200, do_sample=False )
#                     output_text = processor.decode(output[0][2:], skip_special_tokens=True)
#                     assistant_response = parse_assistant_response(output_text)
#                     result = parse_result(assistant_response)

#                     if result == 'Yes' and target_label == 1:
#                         tp += 1
#                         if tp == 1:
#                             idx = image_name.find("+")
#                             time_to_detect = int(image_name[idx:-4])
#                             # display(gridded_image)

#                     elif result == 'Yes' and target_label == 0:
#                         fp += 1

#                     elif result == 'No' and target_label == 1:
#                         fn += 1

#                     if result == "Yes":
#                         predictions.append(1)
#                     else:
#                         predictions.append(0)

#             except Exception as e:
#                 logger.info(f"Error processing {image_name}: {str(e)}")

#         if tp + fp + fn > 0:
#             fire_scores[fire] = {
#                             "precision": round(tp / (tp + fp), 2) if (tp + fp) != 0 else 0.0,
#                             "recall": round(tp / (tp + fn), 2) if (tp + fn) != 0 else 0.0,
#                             "f1_score": round(tp / (tp + 0.5 * (fp + fn)), 2) if (tp + 0.5 * (fp + fn)) != 0 else 0.0,
#                             "time_to_detect": time_to_detect,
#                             "true_positives": tp,
#                             "false_positives": fp,
#                             "false_negatives": fn,
#                             "predictions": predictions
#                                 }
#         fires_covered += 1
#         if fires_covered % 5 == 0:
#             logger.info(f"======== Count of fires: {fires_covered} =======")
#             avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
#             avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
#             avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
#             avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])

#             logger.info(f"Average Precision: {avg_precision}")
#             logger.info(f"Average Recall: {avg_recall}")
#             logger.info(f"Average F1 score: {avg_f1_score}")
#             logger.info(f"Average time to detect: {avg_time_to_detect}")
#             logger.info(f"Time taken: {time.time() - start_time} secs")
#             # break

# avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
# avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
# avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
# avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])

# logger.info(f"Overall Average Precision: {avg_precision}")
# logger.info(f"Overall Average Recall: {avg_recall}")
# logger.info(f"Overall Average F1 score: {avg_f1_score}")
# logger.info(f"Overall Average time to detect: {avg_time_to_detect}")

# # Save to a pickle file
# with open("fire_scores.pickle", "wb") as file:  # 'wb' = write in binary mode
#     pickle.dump(fire_scores, file)

# logger.info("============= File saved ================")
