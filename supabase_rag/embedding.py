from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# initialize the Embedding Model
# all-mpnet-base-v2: dimensions = 768
model = HuggingFaceEmbeddings(model="all-mpnet-base-v2")

# inMemoryStore initialization for local testing
vector_store = InMemoryVectorStore(embedding=model)


# function to get dimension of the model
def get_dimensions(text: str) -> int:
    embedding = model.embed_query(text)
    return len(embedding)


# create embeddings from string
def create_embedding(text: str) -> list[float]:
    embedding = model.embed_query(text)
    return embedding


# decode embeddings
def decode_embedding(embedding: list[float]) -> list[str]:
    return model.decode(embedding)


if __name__ == "__main__":
    texts = texts = [
        "The PCF Parser extracts process control features from configuration documents.",
        "Slack bot automatically summarizes project updates from channel messages.",
        "Meeting transcripts are analyzed to identify new project requirements.",
        "The Neo4j graph stores relationships between PCFs, components, and features.",
        "LangGraph manages workflow nodes that connect tools and memory states.",
        "RAG pipeline retrieves relevant documents before answering user questions.",
        "A knowledge node links each PCF document to its corresponding project module.",
        "Embedding generation is handled locally using the MPNet model.",
        "The Slack integration triggers automation when keywords are detected in messages.",
        "A scheduled job rebuilds the graph connections every midnight to sync data.",
        "Developers can query the MCP Tools Brain using natural language.",
        "Each component in the PCF has associated performance metrics and logs.",
        "LangGraph supports config-driven node definitions for reusable workflows.",
        "When a doctor sends a prescription form, the parser automatically extracts patient data.",
        "The customer intake agent validates information before saving it to the database.",
    ]

    dimensions = get_dimensions(texts[0])
    print(f"Dimensions of embedding: {dimensions}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = splitter.create_documents(texts)
    vector_store.add_documents(docs)
    print(f"Number of documents in vector store: {len(vector_store.store)}")

    # for index, (idx, doc) in enumerate(vector_store.store.items()):
    #     print(doc.keys())
    #     print(
    #         f"Document {idx} | length: {len(doc['vector'])} | Text: {doc['text']} | Metadata: {doc['metadata']}"
    #     )

    query = "How does the system automatically extract structured data from incoming documents?"

    results = vector_store.similarity_search_with_score(query, k=2)
    print(f"Top {len(results)} results for query '{query}':")
    for doc, score in results:
        print(
            f" - Score: {score} | Page Content: {doc.page_content} | Metadata: {doc.metadata}"
        )
