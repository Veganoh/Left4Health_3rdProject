from transformers import BertTokenizer, BertForQuestionAnswering
import torch


def answer_question(question, context):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)

    # Perform inference
    outputs = model(**inputs)

    # Get the most likely answer
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
    answerr = tokenizer.decode(answer_tokens)

    return answerr


# Example usage
if __name__ == "__main__":
    question = "I have a big mole that keeps growing"
    context = "You're concerned about a skin disease."
    answer = answer_question(question, context)
    print("Answer:", answer)
