from openai import OpenAI

client = OpenAI(api_key='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')
import csv
import os


def replace_attribute(json_array, old_attribute, new_attribute):
    for obj in json_array:
        if old_attribute in obj:
            obj[new_attribute] = obj.pop(old_attribute)
    return json_array


def generate_answer_with_intent(messages, disease_intent):
    messages.insert(0,{
        "role": "system",
        "content": f"You will be helping a patient with skin disease queries. Patient is trying to find out about "
                   f"{disease_intent} skin disease, as far as we know. Do not answer patient query about other things."
      })

    return generate_answer(messages)


def generate_answer_without_intent(messages):
    # this grounds the chat gpt into skin disease recognition of the agreed diseases
    messages.insert(0,{
        "role": "system",
        "content": "You will be helping a patient with skin disease queries. Patient is trying to find out about a "
                   "skin disease, as far as we know, we dont know specifically which one but suspect its either "
                   "melanoma, dermatitis, psoriasis, urticaria or lupus."
                   " Do not answer patient query about other things."
      })

    return generate_answer(messages)


def generate_answer(messages):
    messages = replace_attribute(messages, 'text', 'content')

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages,
                                              max_tokens=500,
                                              n=1)

    responses = ""
    print(response)
    for resp in response.choices:
        print(resp.message.content)
        responses = responses + resp.message.content + " "

    return {"role": "assistant", "text": responses}