import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def analyze_image(image_path_or_url, query, previous_messages=None):
    model_id = "microsoft/Phi-3-vision-128k-instruct"

    print("Loading model and processor...")
    with tqdm(total=2, desc="Loading components") as pbar:
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation='eager'
        )
        pbar.update(1)
        
        # Initialize processor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        pbar.update(1)

    # Load image with progress bar for URL downloads
    if image_path_or_url.startswith("http"):
        print("Downloading image...")
        response = requests.get(image_path_or_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            content = b""
            for data in response.iter_content(chunk_size=1024):
                content += data
                pbar.update(len(data))
            image = Image.open(io.BytesIO(content))
    else:
        image = Image.open(image_path_or_url)

    # Prepare messages
    if previous_messages is None:
        messages = [{"role": "user", "content": f"<|image_1|>\n{query}"}]
    else:
        messages = previous_messages + [{"role": "user", "content": query}]

    print("Processing input...")
    # Process input
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs and move to CUDA with progress bar
    with tqdm(total=2, desc="Processing") as pbar:
        inputs = processor(prompt, [image], return_tensors="pt")
        pbar.update(1)
        
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        pbar.update(1)

    # Generate response
    print("Generating response...")
    generation_args = {
        "max_new_tokens": 500,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
    )

    # Process and return response
    print("Decoding response...")
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


# Example usage
if __name__ == "__main__":
    try:
        # Example with local file
        image_path = "../test-images/person1001_bacteria_2932.jpeg"
        query = "What is shown in this image?"

        print("\nStarting image analysis...")
        print("-" * 50)
        response = analyze_image(image_path, query)
        print("-" * 50)
        print("\nResponse:")
        print(response)
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")