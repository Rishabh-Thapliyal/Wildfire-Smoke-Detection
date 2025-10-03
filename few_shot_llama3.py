import openai

client = openai.OpenAI(
    api_key="sk-PP2Y9C-TkPw_ZBRPnDQryQ",
    base_url="https://llm.nrp-nautilus.io"
)
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

response1 = '''
Result: No
Confidence: 8
Description: The image shows a clear blue sky with some wispy clouds, but there are no visible smoke plumes or haze that would indicate the presence of wildfire smoke. The overall atmosphere appears serene and free of any signs of smoke or pollution.
'''

response2 = '''
Result: Yes
Confidence: 7
Description: The image shows a hazy, blue-gray layer of smoke in the distance, partially obscuring the mountain range. The smoke appears to be rolling in from the right side of the image, with a few wispy tendrils extending upwards into the sky. The overall effect is one of a subtle, yet noticeable haze that reduces the visibility of the landscape.
'''


response = client.chat.completions.create(
    model="llama3",
    messages=[
        # System message (optional) to set behavior
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes images and answers questions about them."
        },
        
        # Example 1
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{palisades_image_dict[0]}"}}
            ]
        },
        {
            "role": "assistant",
            "content": f"{response1}"
        },
        
        # Example 2
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{palisades_image_dict[12]}"}}
            ]
        },
        {
            "role": "assistant",
            "content": f"{response2}"
        },
        
        # Example 3
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"{palisades_image_dict[24]}"}}
            ]
        }
    ],
)

print(response.choices[0].message.content)