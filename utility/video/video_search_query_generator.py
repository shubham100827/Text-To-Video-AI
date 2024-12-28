import os
import json
import re
from datetime import datetime
from utility.utils import log_response,LOG_TYPE_GPT
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


prompt = """# Instructions

Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

For example, if the caption is 'The cheetah is the fastest land animal, capable of running at speeds up to 75 mph', the keywords should include 'cheetah running', 'fastest animal', and '75 mph'. Similarly, for 'The Great Wall of China is one of the most iconic landmarks in the world', the keywords should be 'Great Wall of China', 'iconic landmark', and 'China landmark'.

Important Guidelines:

Use only English in your text queries.
Each search string must depict something visual.
The depictions have to be extremely visually concrete, like rainy street, or cat sleeping.
'emotional moment' <= BAD, because it doesn't depict something visually.
'crying child' <= GOOD, because it depicts something visual.
The list must always contain the most relevant and appropriate query searches.
['Car', 'Car driving', 'Car racing', 'Car parked'] <= BAD, because it's 4 strings.
['Fast car'] <= GOOD, because it's 1 string.
['Un chien', 'une voiture rapide', 'une maison rouge'] <= BAD, because the text query is NOT in English.

Note: Your response should be the response only and no extra text or data.
"""


def fix_json(json_str):
    # Replace typographical apostrophes with straight quotes
    json_str = json_str.replace("’", "'")
    # Replace any incorrect quotes (e.g., mixed single and double quotes)
    json_str = json_str.replace("“", "\"").replace("”", "\"").replace("‘", "\"").replace("’", "\"")
    # Add escaping for quotes within the strings
    json_str = json_str.replace('"you didn"t"', '"you didn\'t"')
    return json_str


def load_model():
    # Log in to Hugging Face # to do: put the tokens somewhere
    HF_API_KEY = os.getenv('HF_KEY')
    login(token=HF_API_KEY)

    # Model and quantization configuration
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    # Define the text-generation pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return text_generator, tokenizer

def getVideoSearchQueriesTimed(script,captions_timed):
    end = captions_timed[-1][0][1]
    try:

        out = [[[0,0],""]]
        while out[-1][0][1] != end:
            content = call_model(script,captions_timed).replace("'",'"')
            try:
                out = json.loads(content)
            except Exception as e:
                print("content: \n", content, "\n\n")
                print(e)
                content = fix_json(content.replace("```json", "").replace("```", ""))
                out = json.loads(content)
        return out
    except Exception as e:
        print("error in response",e)

    return None


def call_model(script,captions_timed):
    user_content = """Script: {} Timed Captions:{}""".format(script,"".join(map(str,captions_timed)))
    print("Content", user_content)
    text_generator, tokenizer = load_model()

    prompt = [{"role": "system", "content": prompt}, {"role": "user", "content": user_content}]
    sequences = text_generator(prompt, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    content = sequences[0]['generated_text'][2]['content']
    text = content.replace("\n", "").strip()
    text = re.sub('\s+', ' ', text)
    log_response(LOG_TYPE_GPT,script,text)
   
    print("Text", text)
    return text


def merge_empty_intervals(segments):
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            # Find consecutive None intervals
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1

            # Merge consecutive None intervals with the previous valid URL
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])

            i = j
        else:
            merged.append([interval, url])
            i += 1

    return merged
