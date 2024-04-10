from openai import OpenAI

client = OpenAI(api_key='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')
import csv
import os

# Set your OpenAI API key securely


def generate_questions(prompt, n=1):
    prompts = []
    prompts.append({'role': 'user', 'content': prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=prompts,
    max_tokens=100,
    n=n,
    stop=["\n"])
    return [resp.message.content.strip() for resp in response.choices]


def append_to_csv(file_path, questions, answer="This question is out of scope for disease classification.",
                  disease="OOS"):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for question in questions:
            writer.writerow([question, answer, disease])


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(current_dir, 'dataset_full.csv')

    # Ensure the CSV has the correct headers if it's a new file
    if not os.path.exists(dataset_file_path):
        with open(dataset_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Answer", "Disease"])

    # Generate and append questions
    for _ in range(100):  # 1000 questions, 10 at a time
        out_of_scope_questions = generate_questions(
            "Give me a question that is out of scope for disease classification.", n=10)
        append_to_csv(dataset_file_path, out_of_scope_questions)


if __name__ == "__main__":
    main()