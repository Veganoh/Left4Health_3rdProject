from openai import OpenAI

client = OpenAI(api_key='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')
import csv
import os


def replace_attribute(json_array, old_attribute, new_attribute):
    for obj in json_array:
        if old_attribute in obj:
            obj[new_attribute] = obj.pop(old_attribute)
    return json_array


def generate_answer(messages):
    messages.insert(0,{
        "role": "system",
        "content": "You will be helping a patient with skin disease queries. Patient is trying to find out about Melanoma skin disease, as far as we know. Do not answer patient query about other things."
      })

    messages = replace_attribute(messages,'text', 'content')

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