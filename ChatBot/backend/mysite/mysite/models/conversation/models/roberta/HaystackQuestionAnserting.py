import os

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

document_joiner = DocumentJoiner()


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
document_store = InMemoryDocumentStore()

document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
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

#not using gpu because mac has issues with it
text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cpu")
)
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)


ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base")

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


def generate_response_haystack(query, intent):
    if intent:
        query = "(" + intent + ") " + query
    result = hybrid_retrieval.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}})

    return result['ranker']['documents'][0]


def main():
    resp = generate_response_haystack('I feel a rash in my skin')
    print(resp)

if __name__ == "__main__":
    main()