from openai import OpenAI

client = OpenAI(api_key='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')
import csv
import os

# Set your OpenAI API key securely


def get_intent(question, n=1):
    try:
        prompts = []
        prompt = f"Question: {question}\n Intent (select ONLY one IN THE LIST): Causes, Diagnosis, Treatments, Symptoms, Generic, please return just one word of the list, otherwise my script will fail\n"
        prompts.append({'role': 'user', 'content': prompt})
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=prompts,
        max_tokens=100,
        n=n,
        stop=["\n"])

        if len(response.choices) > 0:
            return response.choices[0].message.content.strip()
    except Exception as e:
        return "OOS"

    return "OOS"

def main():
    # Open the input CSV file for reading
    with open('dataset_full.csv', 'r') as input_file:
        csv_reader = csv.reader(input_file)
        next(csv_reader)  # Skip header row if exists

        # Open the output CSV file for writing
        with open('dataset_with_intents.csv', 'w', newline='') as output_file:
            fieldnames = ['Question', 'Answer', 'Disease', 'Intent']
            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            # Iterate over each row in the input CSV file
            for row in csv_reader:
                question = row[0]  # Assuming the question is in the first column
                intent = get_intent(question)

                # Write the row along with the predicted intent into the output CSV file
                csv_writer.writerow({'Question': row[0], 'Answer': row[1], 'Disease': row[2], 'Intent': intent})


if __name__ == "__main__":
    main()