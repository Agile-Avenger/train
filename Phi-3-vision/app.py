from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor


def analyze_image(image_path_or_url, query, previous_messages=None):
    model_id = "microsoft/Phi-3-vision-128k-instruct"

    # Initialize model and processor for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",  # Changed to CPU
        trust_remote_code=True,
        torch_dtype="auto",
        # Removed flash attention as it's GPU-only
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load image
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)

    # Prepare messages
    if previous_messages is None:
        messages = [{"role": "user", "content": f"<|image_1|>\n{query}"}]
    else:
        messages = previous_messages + [{"role": "user", "content": query}]

    # Process input
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Changed to CPU
    inputs = processor(prompt, [image], return_tensors="pt")

    # Generate response with memory-efficient settings
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    # Process and return response
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


# Example usage
if __name__ == "__main__":
    try:
        # Example with URL
        url = "../test-images/person1001_bacteria_2932.jpeg"
        query = "What is shown in this image?"

        print("Processing image... This may take a while on CPU.")
        response = analyze_image(url, query)
        print("\nResponse:")
        print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
