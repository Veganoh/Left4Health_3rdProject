import os
import json
from openai import OpenAI

client = OpenAI(api_key="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")
from tenacity import retry, stop_after_attempt, wait_exponential

# Set your OpenAI API key here


# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def query_openai(image_name):
    prompt = f"What skin disease is depicted in the image named '{image_name}'?"
    response = client.completions.create(engine="davinci",
    prompt=prompt,
    max_tokens=20)
    return response.choices[0].text.strip()


# Function to batch query OpenAI API
def batch_query_openai(image_names, model="gpt-3.5-turbo"):
    prompts = [f"What skin disease is depicted in the image named '{name}'?" for name in image_names]
    responses = client.completions.create(model=model,
    prompt=prompts,
    max_tokens=20)
    return [resp['choices'][0]['text'].strip() for resp in responses.choices]


# Main function to process folders and images
def process_folders(root_folder, batch_size=10):
    results = {}
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            image_names = []
            for image_name in os.listdir(folder_path):
                prediction_file = f"{image_name}_results.json"
                if not os.path.exists(prediction_file):
                    image_names.append(image_name)
                    if len(image_names) == batch_size:
                        # Batch querying OpenAI API
                        predictions = batch_query_openai(image_names)
                        for i, prediction in enumerate(predictions):
                            # Assuming image names are the actual skin disease names
                            disease_name = os.path.splitext(image_names[i])[0]
                            if folder_name not in results:
                                results[folder_name] = []
                            result = {
                                "image_name": image_names[i],
                                "disease_name": disease_name,
                                "prediction": prediction
                            }
                            output_file = f"{image_names[i]}_results.json"
                            f = open(output_file, "w")
                            json.dump(result, f, indent=4)
                            print(f"Results for {image_names[i]} saved to {output_file}")
                        image_names = []
    return results


# Main function
def main():
    root_folder = "../../../Dataset/Dermnet"
    results = process_folders(root_folder)


if __name__ == "__main__":
    main()

