import openai
import os
client = openai.OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url="https://llm.nrp-nautilus.io"
)

os.sys.path.append("/expanse/lustre/projects/ddp464/mhnguyen/data")



palisades_image_dict = {
    0: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274241_%2B00000.jpg",
    1: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274301_%2B00060.jpg",
    2: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274362_%2B00121.jpg",
    3: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274422_%2B00181.jpg",
    6: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274601_%2B00360.jpg",
    12: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736274961_%2B00720.jpg",
    24: "http://legacy-www.hpwren.ucsd.edu/FIgLib/HPWREN-FIgLib-Data/20250107_PalisadesFire_69bravo-e-mobo-c/1736275681_%2B01440.jpg",   
}

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
img = palisades_image_dict[24]

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'''{prompt}'''},
                {"type": "image_url",
                 "image_url":
                 {
                    "url": f"{img}",
                    "format": "image/jpeg"}
                 },
            ],
        }
    ],
)

print(response.choices[0].message.content)


messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": f"{img}"},
                    {"type": "text", "text": f"{prompt}"}
                ]
            }
        ]


response = client.chat.completions.create(
            model="llava-onevision",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
print("=================================")
print(response.choices[0].message.content)


completion = client.chat.completions.create(
    model="gemma3",
    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": f"{img}"},
            {"type": "text", "text": f"{prompt}"}
        ]
    }
]
)

print("=================================")
print(completion.choices[0].message.content)



# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])


with open('val_fires_final.txt', 'r') as file:
    val_fire_names = [line.strip() for line in file if line.strip()]

fire_scores = {}
fires_covered = 0
import time
from tqdm import tqdm
import os
import base64
start_time = time.time()

def encode_image(image_path):
    """Convert local image to base64 data URI"""
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"


for fire in tqdm(os.listdir("/expanse/lustre/projects/ddp464/mhnguyen/data"), desc='processing images...'):

    if fire in val_fire_names:
        # print(fire)
        print(f"=============== {fire} ===============")
        image_dir = f"/expanse/lustre/projects/ddp464/mhnguyen/data/{fire}"

        # Get all files and sort them alphabetically (start with negative sample and then positives...)
        image_files = sorted([f for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            print(image_path)
            
            completion = client.chat.completions.create(
                    model="gemma3",
                    messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": f"{encode_image(image_path)}"},
                            {"type": "text", "text": f"{prompt}"}
                        ]
                    }
                ]
                )

            print("=================================")
            print(completion.choices[0].message.content)



