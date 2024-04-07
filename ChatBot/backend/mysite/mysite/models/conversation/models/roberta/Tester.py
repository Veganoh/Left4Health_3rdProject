list_of_queries_valid = [{'query':'What are the symptoms of melanoma?'},
{'query':'Are there any treatments for melanoma?'},
{'query':'How can I get melanoma?'},
{'query':'Is melanoma a serious disease?'},
{'query':'Can I die of melanoma?'},
{'query':'What are the symptoms of psoriasis?'},
{'query':'Are there any treatments for psoriasis?'},
{'query':'How can I get psoriasis?'},
{'query': 'What are the symptoms of urticaria?'},
{'query': 'Are there any treatments for urticaria?'},
{'query': 'How can I get urticaria?'},
{'query': 'What are the symptoms of lupus?'},
{'query': 'Are there any treatments for lupus?'},
{'query': 'How can I get lupus?'},
{'query': 'What are the symptoms of dermatitis?'},
{'query': 'Are there any treatments for dermatitis?'},
{'query': 'How can I get dermatitis?'}]



melanoma_queries = [
    {"query": "I've noticed a new mole on my skin that looks irregular. Could it be melanoma?"},
    {"query": "Is it normal for a mole to change color and size? I'm concerned it might be melanoma."},
    {"query": "I have a family history of melanoma. Should I be more vigilant about checking my skin for any changes?"},
    {"query": "What are the early signs and symptoms of melanoma that I should watch out for?"},
    {"query": "Does excessive sun exposure increase my risk of developing melanoma?"},
    {"query": "I have a mole that's been itching and bleeding. Should I be worried about melanoma?"},
    {"query": "How often should I get my moles checked by a dermatologist for signs of melanoma?"},
    {"query": "Are there any specific factors that put me at higher risk for melanoma?"},
    {"query": "If melanoma is diagnosed early, what are the treatment options available?"},
    {"query": "Can you explain the ABCDE rule for identifying melanoma lesions on the skin?"}
]

urticaria_queries = [
    {"query": "I suddenly developed red, itchy welts on my skin. Could it be urticaria?"},
    {"query": "Is it common for stress to trigger urticaria outbreaks?"},
    {"query": "How can I differentiate between urticaria and other skin conditions like eczema or psoriasis?"},
    {"query": "Are there any specific foods that commonly cause urticaria reactions?"},
    {"query": "What over-the-counter medications can I use to alleviate the itching and swelling associated with urticaria?"},
    {"query": "I've heard about chronic urticaria. What distinguishes it from acute urticaria?"},
    {"query": "Should I avoid certain environmental factors like heat or cold to prevent urticaria outbreaks?"},
    {"query": "Is there a connection between urticaria and allergies?"},
    {"query": "How long do urticaria outbreaks typically last, and is there any way to shorten their duration?"},
    {"query": "Are there any lifestyle changes or dietary adjustments I can make to manage my urticaria symptoms better?"}
]

lupus_queries = [
    {"query": "I've been experiencing joint pain and swelling. Could it be a symptom of lupus?"},
    {"query": "What are the common early signs and symptoms of lupus that I should be aware of?"},
    {"query": "Is lupus hereditary? I have family members with autoimmune diseases."},
    {"query": "Can lupus affect different organs in the body, and if so, how does it manifest?"},
    {"query": "Are there specific triggers or factors that can exacerbate lupus symptoms?"},
    {"query": "I've heard about a lupus rash. How does it appear, and is it always present in lupus patients?"},
    {"query": "How is lupus diagnosed, and what tests are usually involved in the diagnostic process?"},
    {"query": "What treatment options are available for managing lupus symptoms, and what are their side effects?"},
    {"query": "Can lupus affect pregnancy, and what precautions should I take if I have lupus and want to conceive?"},
    {"query": "Is there a cure for lupus, or is it a lifelong condition that requires ongoing management?"}
]

dermatitis_queries = [
    {"query": "I have developed a red, itchy rash on my skin. Could it be dermatitis?"},
    {"query": "Is dermatitis contagious? Can I spread it to others by direct contact?"},
    {"query": "What are the common triggers for dermatitis outbreaks, and how can I avoid them?"},
    {"query": "Are there different types of dermatitis, and how do they differ in symptoms and treatment?"},
    {"query": "What over-the-counter creams or lotions are effective in relieving the itching and inflammation associated with dermatitis?"},
    {"query": "Can stress exacerbate dermatitis symptoms, and if so, what can I do to manage stress effectively?"},
    {"query": "Should I avoid certain fabrics or materials in clothing to prevent irritation and flare-ups of dermatitis?"},
    {"query": "How long do dermatitis flare-ups typically last, and is there anything I can do to shorten their duration?"},
    {"query": "Is it possible for dermatitis to lead to more serious skin conditions if left untreated?"},
    {"query": "Are there any dietary changes or supplements that can help improve dermatitis symptoms?"}
]

psoriasis_queries = [
    {"query": "I have patches of thick, red skin with silvery scales. Could this be psoriasis?"},
    {"query": "Is psoriasis contagious? Can I transmit it to others through skin-to-skin contact?"},
    {"query": "What are the common triggers for psoriasis flare-ups, and how can I avoid them?"},
    {"query": "Are there different types of psoriasis, and do they vary in terms of symptoms and treatment approaches?"},
    {"query": "What topical treatments are effective in managing the itching and scaling associated with psoriasis?"},
    {"query": "Can psoriasis affect nails and joints, and if so, what are the symptoms and treatment options?"},
    {"query": "Should I avoid certain lifestyle factors like smoking or alcohol consumption to prevent worsening of psoriasis symptoms?"},
    {"query": "How does psoriasis affect mental health, and are there resources available for coping with the emotional impact of the condition?"},
    {"query": "What role does genetics play in psoriasis, and are there any genetic tests available to assess the risk of developing the condition?"},
    {"query": "Are there any alternative or complementary therapies that can help alleviate psoriasis symptoms, in addition to traditional medical treatments?"}
]

list_of_queries_valid = (list_of_queries_valid + melanoma_queries + urticaria_queries + psoriasis_queries + dermatitis_queries + lupus_queries)

list_of_queries_invalid = [{'query':'Will I die?'},
                           {'query':'What is the meaning of life?'},
                           {'query':'How to fry an egg?'}]



def get_analytics_queries():
    return{'list_of_queries_valid':list_of_queries_valid,
           'list_of_queries_invalid':list_of_queries_invalid}
