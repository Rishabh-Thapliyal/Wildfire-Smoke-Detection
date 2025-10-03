import torch
import re
import os
from tqdm import tqdm
import warnings
from PIL import Image, ImageDraw
# from IPython.display import display
import requests
from PIL import Image
import numpy as np
import time
import pickle

import logging

import logging
from datetime import datetime
import sys
import gc


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
    file_handler = logging.FileHandler('logs_test2_qwen2_5_final.log', mode='w', encoding='utf-8')
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


# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore")

os.sys.path.append("/expanse/lustre/projects/ddp464/mhnguyen/data")


# =========== select the model =========
# Set memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
gc.collect()

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

qwen2_model_path = "/expanse/lustre/projects/ddp464/rthapliyal/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen2_model_path,
#     torch_dtype="auto",
#     device_map="auto",
    torch_dtype=torch.bfloat16,  # Use bfloat16 instead of "auto"
    device_map={"": "cuda:0"},  # Force all model components to cuda:0
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda")

processor = AutoProcessor.from_pretrained(qwen2_model_path)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,  # Use bfloat16 instead of "auto"
#     device_map={"": "cuda:0"},  # Force all model components to cuda:0
#     low_cpu_mem_usage=True,
# ).to("cuda")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     load_in_8bit = True,
# #     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

model.eval()
logger.info(f"Model device: {next(model.parameters()).device}")

# Clear cache after model loading
torch.cuda.empty_cache()
gc.collect()

# ========================================

def parse_llm_fire_response(llm_output):
    """
    Parses LLM output with robust fallback logic:
    1. First tries structured formats (with/without bold)
    2. Then looks for "result:" + yes/no
    3. Finally checks for standalone yes/no keywords
    """
    if not isinstance(llm_output, str):
        return None

    # Main pattern (handles bold/plain, case-insensitive)
    main_pattern = (
        r"(?:\*\*)?Result(?:\*\*)?\s*[:=]\s*(?:\*\*)?(Yes|No)(?:\*\*)?\s*"
        r"(?:\*\*)?Confidence(?:\*\*)?\s*[:=]\s*(?:\*\*)?(\d+)(?:\*\*)?\s*"
        r"(?:\*\*)?Description(?:\*\*)?\s*[:=]\s*(?:\*\*)?(.*?)(?:\*\*)?$"
    )
    match = re.search(main_pattern, llm_output, re.IGNORECASE | re.DOTALL)
    
    if match:
        return {
            'Result': match.group(1).strip().capitalize(),
            'Confidence': int(match.group(2)),
            'Description': match.group(3).strip()
        }
    
    # Fallback 1: Look for "result:" + yes/no
    result_match = re.search(
        r"(?:result|RESULT)\s*[:=]\s*(yes|no)", 
        llm_output, 
        re.IGNORECASE
    )
    if result_match:
        return {
            'Result': result_match.group(1).capitalize(),            
        }
    
    # Fallback 2: Find standalone yes/no
    standalone_yesno = re.search(
        r"\b(yes|no)\b", 
        llm_output, 
        re.IGNORECASE
    )
    if standalone_yesno:
        return {
            'Result': standalone_yesno.group(1).capitalize(),
        }
    
    return None

prompt = f'''
Analyze this image for wildfire smoke presence. Respond strictly in this format:

Result: [Yes/No]
Confidence: [0-10]
Description: [2-3 sentence explanation of visual evidence]

Rules:
1. Focus only on visual smoke indicators (ignore text/logos)
2. "Yes" requires clear visible smoke plumes
3. Confidence 8+ needs unambiguous evidence
4. Keep descriptions factual and concise
'''


# Path to the directory containing images
# image_dir = "data/20160722_FIRE_mw-e-mobo-c"
with open('test_fires_final.txt', 'r') as file:
    val_fire_names = [line.strip() for line in file if line.strip()]

fire_scores = {}
fires_covered = 0

start_time = time.time()
print(torch.version.cuda) 

for fire in tqdm(os.listdir("/expanse/lustre/projects/ddp464/mhnguyen/data"), desc='processing images...'):

    if fire in val_fire_names:
        # print(fire)
        logger.info(f"=============== {fire} ===============")
        image_dir = f"/expanse/lustre/projects/ddp464/mhnguyen/data/{fire}"

        # Get all files and sort them alphabetically (start with negative sample and then positives...)
        image_files = sorted([f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        time_to_detect = 0
        predictions = list()
        inference_time = list()

        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)

            # Determine target_label based on image name
            if '-' in image_name:
                target_label = 0
            elif '+' in image_name:
                target_label = 1
            else:
                target_label = -1  # default case if neither - nor + is found

            try:
                messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"{image_path}",
                                },
                                {"type": "text", "text": f'''{prompt}
                                '''},
                            ],
                        }
                    ]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
                logger.info(f"input device: {next(model.parameters()).device}")

                # Inference: Generation of the output
                t1 = time.time()
                try:
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                except RuntimeError as e:
                    print(f"CUDA Error: {e}")
                    print("Input IDs:", inputs.input_ids)
                    print("Attention Mask:", inputs.attention_mask)
                    raise
#                 generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
#                 print(output_text)
                parsed_result = parse_llm_fire_response(output_text[0])
                logger.info(parsed_result)
                if parsed_result:
                    # logger.info(f"Result: {parsed_result['Result']}")
                    # logger.info(f"Confidence: {parsed_result['Confidence']}")
                    # logger.info(f"Description: {parsed_result['Description']}")
                    if parsed_result['Result'].lower() == "yes":
                        result = "Yes"
                    else:
                        result = "No"
                else:
                    logger.info("Failed to parse LLM output.  Output was not in the expected format.")
                    result = "No"
                t2 = time.time()
                inference_time.append(t2-t1)
            
                if result == 'Yes' and target_label == 1:
                    tp += 1
                    if tp == 1:
                        idx = image_name.find("+")
                        time_to_detect = int(image_name[idx:-4])

                elif result == 'Yes' and target_label == 0:
                    fp += 1

                elif result == 'No' and target_label == 1:
                    fn += 1
                else:
                    tn += 1

                if result == "Yes":
                    predictions.append(1)
                else:
                    predictions.append(0)
                    
            except Exception as e:
                logger.info(f"Error processing {image_name}: {str(e)}")
                torch.cuda.empty_cache()
                gc.collect()

        fire_scores[fire] = {
                        "accuracy" : round((tp+tn)/(tp+tn+fp+fn), 2),
                        "precision": round(tp / (tp + fp), 2) if (tp + fp) != 0 else 0.0,
                        "recall": round(tp / (tp + fn), 2) if (tp + fn) != 0 else 0.0,
                        "f1_score": round(tp / (tp + 0.5 * (fp + fn)), 2) if (tp + 0.5 * (fp + fn)) != 0 else 0.0,
                        "time_to_detect": time_to_detect,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "true_negatives" : tn,
                        "predictions": predictions,
                        "inference_time" : inference_time
                            }
        
        fires_covered += 1
        if fires_covered % 5 == 0:
            logger.info(f"======== Count of fires: {fires_covered} =======")
            avg_accuracy = np.mean([v['accuracy'] for k,v in fire_scores.items()])
            avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
            avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
            avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
            avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])
            
            logger.info(f"Average Accuracy: {avg_accuracy}")
            logger.info(f"Average Precision: {avg_precision}")
            logger.info(f"Average Recall: {avg_recall}")
            logger.info(f"Average F1 score: {avg_f1_score}")
            logger.info(f"Average time to detect: {avg_time_to_detect}")
            logger.info(f"Time taken: {time.time() - start_time} secs")
#         break

avg_accuracy = np.mean([v['accuracy'] for k,v in fire_scores.items()])
avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])

logger.info(f"Average Accuracy: {avg_accuracy}")
logger.info(f"Overall Average Precision: {avg_precision}")
logger.info(f"Overall Average Recall: {avg_recall}")
logger.info(f"Overall Average F1 score: {avg_f1_score}")
logger.info(f"Overall Average time to detect: {avg_time_to_detect}")
logger.info(f"Time taken: {time.time() - start_time} secs")

# Save to a pickle file
with open("fire_scores_qwen2_5_final_test2.pickle", "wb") as file:  # 'wb' = write in binary mode
    pickle.dump(fire_scores, file)

logger.info("============= File saved ================")
torch.cuda.empty_cache()
gc.collect()
