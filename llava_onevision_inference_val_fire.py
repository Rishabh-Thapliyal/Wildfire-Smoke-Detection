import re
import os
from tqdm import tqdm
import warnings
# from IPython.display import display
import numpy as np
import base64
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
    file_handler = logging.FileHandler('./logs/logs_val_llava_onevision.log', mode='w', encoding='utf-8')
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


# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore")

os.sys.path.append("/expanse/lustre/projects/ddp464/mhnguyen/data")


# =========== select the model =========

import openai
client = openai.OpenAI(
    api_key="sk-PP2Y9C-TkPw_ZBRPnDQryQ",
    base_url="https://llm.nrp-nautilus.io"
)

prompt = '''
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

def encode_image(image_path):
    """Convert local image to base64 data URI"""
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

with open('val_fires_final.txt', 'r') as file:
    val_fire_names = [line.strip() for line in file if line.strip()]

fire_scores = {}
fires_covered = 0

start_time = time.time()

for fire in tqdm(os.listdir("/expanse/lustre/projects/ddp464/mhnguyen/data"), desc='processing images...'):

    if fire in val_fire_names:
        
        logger.info(f"=============== {fire} ===============")
        image_dir = f"/expanse/lustre/projects/ddp464/mhnguyen/data/{fire}"

        # Get all files and sort them alphabetically (start with negative sample and then positives...)
        image_files = sorted([f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        tp = 0
        fp = 0
        fn = 0
        time_to_detect = 0
        predictions = list()

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

                response = client.chat.completions.create(
                            model="llava-onevision",
                            messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": f"{encode_image(image_path)}"},
                                    {"type": "text", "text": f"{prompt}"}
                                ]
                            }
                        ],
                            max_tokens=300
                        )

                llm_output = response.choices[0].message.content
                logger.info(llm_output)
                parsed_result = parse_llm_fire_response(llm_output)
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

                if result == 'Yes' and target_label == 1:
                    tp += 1
                    if tp == 1:
                        idx = image_name.find("+")
                        time_to_detect = int(image_name[idx:-4])

                elif result == 'Yes' and target_label == 0:
                    fp += 1

                elif result == 'No' and target_label == 1:
                    fn += 1

                if result == "Yes":
                    predictions.append(1)
                else:
                    predictions.append(0)
                    
            except Exception as e:
                logger.info(f"Error processing {image_name}: {str(e)}")


        fire_scores[fire] = {
                        "precision": round(tp / (tp + fp), 2) if (tp + fp) != 0 else 0.0,
                        "recall": round(tp / (tp + fn), 2) if (tp + fn) != 0 else 0.0,
                        "f1_score": round(tp / (tp + 0.5 * (fp + fn)), 2) if (tp + 0.5 * (fp + fn)) != 0 else 0.0,
                        "time_to_detect": time_to_detect,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "predictions": predictions
                            }
        
        fires_covered += 1
        logger.info(f"---------- Fires covered : {fires_covered} ------------------")
        if fires_covered % 5 == 0:
            logger.info(f"======== Count of fires: {fires_covered} =======")
            avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
            avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
            avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
            avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])

            logger.info(f"Average Precision: {avg_precision}")
            logger.info(f"Average Recall: {avg_recall}")
            logger.info(f"Average F1 score: {avg_f1_score}")
            logger.info(f"Average time to detect: {avg_time_to_detect}")
            logger.info(f"Time taken: {time.time() - start_time} secs")
            # break

avg_precision = np.mean([v['precision'] for k,v in fire_scores.items()])
avg_recall = np.mean([v['recall'] for k,v in fire_scores.items()])
avg_f1_score = np.mean([v['f1_score'] for k,v in fire_scores.items()])
avg_time_to_detect = np.mean([v['time_to_detect'] for k,v in fire_scores.items()])

logger.info(f"Overall Average Precision: {avg_precision}")
logger.info(f"Overall Average Recall: {avg_recall}")
logger.info(f"Overall Average F1 score: {avg_f1_score}")
logger.info(f"Overall Average time to detect: {avg_time_to_detect}")

# Save to a pickle file
with open("fire_scores_llava_onevision.pickle", "wb") as file:  # 'wb' = write in binary mode
    pickle.dump(fire_scores, file)

logger.info("============= File saved ================")
