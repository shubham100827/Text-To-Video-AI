import os
import json
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


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


def generate_script(topic):
    sys_prompt = """You are a seasoned content writer for a YouTube Shorts channel, specializing in facts videos and has a unique hook to capture the viewer's attention in the first few seconds.
                  Your facts shorts are concise, each lasting less than 50 seconds (approximately 140 words). They are incredibly engaging and original. When a user requests a specific type of facts short, you will create it.

              When creating a script, keep these guidelines in mind:
              - **Tone**: Informative, engaging, and relatable. Add a touch of humor or surprise where appropriate.
              - **Structure**: Start with an intriguing hook, present 5-6 fascinating facts, and only for few of them end with a memorable or thought-provoking closing line.
              - **Creativity**: Ensure all facts are unique, unexpected, and relevant to the topic. Avoid common knowledge especially if the topic given by user is too common e.g. Dog. Each fact should be unique and surprising.
              - **Avoid repetition**: Never ever repeat same wording in each fact or line. e.g. Don't always add *Did you know that...*.
              - **Factuality**: Dont make up some information to prove facts. Make sure your script is based on true facts, truths and reality.

              For instance, if the user asks for:
              *Weird facts*
              You would produce content like this:

              **Weird facts you didn’t know**:
              - Bananas are berries, but strawberries aren’t.
              - A single cloud can weigh over a million pounds.
              - There’s a species of jellyfish that is biologically immortal.
              - Honey never spoils. Archaeologists found 3,000-year-old honey in Egyptian tombs that’s still edible.
              - The shortest war in history was between Britain and Zanzibar in 1896. It lasted just 38 minutes.
              - Octopuses have three hearts and blue blood.

              If user asks for
              *Hitler*
              You would produce content like this:

              - Hitler was a failed artist, his paintings were so bad they're now considered a joke.
              - He was a heavy smoker, and his smoking habit led to a severe case of lung cancer.
              - Before becoming a dictator, Hitler was a struggling artist in Vienna, living in poverty and relying on charity.
              - Hitler's first wife, Anna, died under mysterious circumstances, and many believe he was involved in her death.
              - He was a fan of horror movies and enjoyed watching horror films, especially those of Nosferatu.
              - Hitler's love for animals was so strong that he had a menagerie at his Berchtesgaden home, with over 70 animals, including a lion, a bear, and a kangaroo.


              Here’s your task:
              - You are now tasked with creating the best short script of approx 150 words based on the user's requested type of 'topic'.
              - Write a script that fits this theme perfectly, grabs attention, and leaves viewers wanting more by considering all the factors mentioned before.

              Stictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

              # Output
              {"script": "The script you have written..."}
              """

    text_generator, tokenizer = load_model()
    prompt = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": topic}]
    sequences = text_generator(prompt, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    content = sequences[0]['generated_text'][2]['content']
    content = content.replace("\n", "")

    try:
        script = json.loads(content)["script"]
    except Exception as e:
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        print(content, e)
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script