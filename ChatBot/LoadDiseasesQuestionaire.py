import csv
import os
import json
import re
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="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")
from tenacity import retry, stop_after_attempt, wait_exponential


# Set your OpenAI API key here


# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def query_openai(image_name):
    prompt = f"What skin disease is depicted in the image named '{image_name}'?"
    response = client.chat.completions.create(engine="gpt-3.5-turbo",
    prompt=prompt,
    max_tokens=20)
    return response.choices[0].text.strip()


batch_size = 20

# Function to batch query OpenAI API
def batch_query_openai(df, model="gpt-3.5-turbo"):
    prompts = []
    disease = 'melanoma'

    completions = []
    for i in range(0, len(df), batch_size):
        # Extract a batch of rows
        batch_df = df.iloc[i:i + batch_size]
        for index, row in batch_df.iterrows():
            question = row[0]
            answer = row[1]
            prompt = f"Given the question '{question}' and answer '{answer}' about the disease '{disease}' can you generate more examples of questions and answers about the same topic?"
            prompts.append({'role': 'user','content': prompt})
            responses = client.chat.completions.create(model=model, messages=prompts,
                                                       temperature=0)
            for resp in responses.choices:
                completions.append({'Prompt': prompt, 'Completion': resp.message.content.strip().split('\n\n')})

    # Write completions to a CSV file
    with open('completions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for completion in completions:
            for pair in completion['Completion']:
                question, answer = pair.split('\n', 1)
                question = question.split(': ')[1].strip()
                answer = answer.split(': ')[1].strip()
                writer.writerow({'Question': question, 'Answer': answer})

    return True


# Main function to process folders and images
def process_diseases(df, batch_size=1):
    # Iterate through all lines in the DataFrame
    batch_query_openai(df)
    # Extract completions from the responses
    print('Done!')
    return True


# Main function
def main():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('dataset.csv', sep='|')
    df_filtered = df[df['processed'] != 1]
    results = process_diseases(df)
    # Add a new column 'processed' with value 1
    df['processed'] = 1

    # Write the updated DataFrame back to the original CSV file
    df.to_csv('dataset.csv', index=False)


if __name__ == "__main__":
    main()

