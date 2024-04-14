import os
import pickle

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
import pandas as pd
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator




def load_pickle(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    return None

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two levels up
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
dataset_file_path = os.path.join(two_levels_up, 'dataset_with_intents.csv')
# Load the dataset
df = pd.read_csv(dataset_file_path)
dataset = df.drop_duplicates()

docs = []
for index, doc in dataset.iterrows():
    contentDoc = doc["Question"] + '[SEP]' + doc["Answer"]
    titleDoc = doc["Disease"] + ' ' + doc["Intent"]
    docs.append(
        Document(content=contentDoc, meta={"title": titleDoc, "abstract": titleDoc})
    )


document_store_file = 'document_store.pkl'
document_store = load_pickle(document_store_file)
if document_store is None:
    document_store = InMemoryDocumentStore()

    document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
    # Using pre trained embedding model bge-small-en-v1.5
    # General embedding model https://github.com/FlagOpen/FlagEmbedding BGE Embedding
    # Pre trained using retroMAE, trained on wikipedia and book corpus
    # used with Huggingface sentence transformers
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cpu")
    )
    document_writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("document_splitter", document_splitter)
    indexing_pipeline.add_component("document_embedder", document_embedder)
    indexing_pipeline.add_component("document_writer", document_writer)

    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    indexing_pipeline.run({"document_splitter": {"documents": docs}})
    with open(document_store_file, 'wb') as f:
        pickle.dump(document_store, f)

# uses transformers to embbedd the query string to find the matches
#not using gpu because mac has issues with it
text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cpu")
)

# retrieves based on vector similarity
embedding_retriever = InMemoryEmbeddingRetriever(document_store)

# this is a BM25 algorithm implementation, works by calculating a relevance score for each document in the collection
# concerning a specific query. The algorithm considers the frequency of query terms in the document,
# the length of the document, and the average document length in the entire collection.
bm25_retriever = InMemoryBM25Retriever(document_store)

# cross encoder model BGE Reranker from https://github.com/FlagOpen/FlagEmbedding
# Cross-encoder will perform full-attention over the input pair, which is more accurate than embedding model
# but more time-consuming than embedding model. Therefore, it can be used to re-rank the top-k documents
# returned by embedding model
# used together with Huggingface similarity transformer
ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")


# joins all the results into one


#    A component that joins multiple list of Documents into a single list.
#    It supports different joins modes:
#    - concatenate: Keeps the highest scored Document in case of duplicates.
#    - merge: Merge a calculate a weighted sum of the scores of duplicate Documents.
#    - reciprocal_rank_fusion: Merge and assign scores based on reciprocal rank fusion.
#  We are using concatenate

document_joiner = DocumentJoiner()

hybrid_retrieval = Pipeline()
hybrid_retrieval.add_component("text_embedder", text_embedder)
hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
hybrid_retrieval.add_component("document_joiner", document_joiner)
hybrid_retrieval.add_component("ranker", ranker)


hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "document_joiner")
hybrid_retrieval.connect("embedding_retriever", "document_joiner")
hybrid_retrieval.connect("document_joiner", "ranker")


def generate_response_haystack(query, intent, document_ids):
    print(query)
    print(intent)

    if intent:
        query = "(" + intent + ") " + query

    print(document_ids)
    result = hybrid_retrieval.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}})

    response = find_new_document(result, document_ids)
    # this could be an array of conversation history so we dont return a follow response same as before
    return response


def find_new_document(result, existing_document_ids):
    # Iterate through each document in the result
    for document in result['ranker']['documents']:
        # Check if the document is not in the existing document IDs
        if document.id not in existing_document_ids:
            return document
    # If no new document is found, return None or an appropriate message
    return None

###########  in case there is no acceptable answer in document store we use the generative model
########### not in use

template = """
Given the following information, answer the question. 
Very important this is a skin disease Server, only for the following
[psoriasis, dermatitis, lupus, melanoma, urticaria] ONLY.
Sometimes the question starts with the skin disease intent
For example: (Psoriasis) What are the disease symptoms?
In this case the answer should always be bound to the disease

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = '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'
generator = OpenAIGenerator(model="gpt-3.5-turbo")

prompt_builder = PromptBuilder(template=template)
#not using gpu because mac has issues with it
text_embedder_copy = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cpu")
)
embedding_retriever_copy = InMemoryEmbeddingRetriever(document_store)
bm25_retriever_copy = InMemoryBM25Retriever(document_store)

ranker_copy = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

document_joiner_copy = DocumentJoiner()

hybrid_retrieval_rag = Pipeline()
hybrid_retrieval_rag.add_component("text_embedder", text_embedder_copy)
hybrid_retrieval_rag.add_component("embedding_retriever", embedding_retriever_copy)
hybrid_retrieval_rag.add_component("bm25_retriever", bm25_retriever_copy)
hybrid_retrieval_rag.add_component("document_joiner", document_joiner_copy)
hybrid_retrieval_rag.add_component("ranker", ranker_copy)
hybrid_retrieval_rag.add_component("prompt_builder", prompt_builder)
hybrid_retrieval_rag.add_component("llm", generator)

hybrid_retrieval_rag.connect("text_embedder", "embedding_retriever")
hybrid_retrieval_rag.connect("bm25_retriever", "document_joiner")
hybrid_retrieval_rag.connect("embedding_retriever", "document_joiner")
hybrid_retrieval_rag.connect("document_joiner", "ranker")
hybrid_retrieval_rag.connect("ranker", "prompt_builder.documents")
hybrid_retrieval_rag.connect("prompt_builder", "llm")


def generate_response_haystack_llm(query, intent):
    if intent:
        query = "(" + intent + ") " + query

    response = hybrid_retrieval_rag.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query},"prompt_builder": {"question": query}})

    return response["llm"]["replies"][0]


def main():
    resp = generate_response_haystack('I feel a rash in my skin')
    print(resp)


if __name__ == "__main__":
    main()