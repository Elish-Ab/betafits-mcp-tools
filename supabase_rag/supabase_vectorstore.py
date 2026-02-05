from langchain_community.vectorstores import SupabaseVectorStore
from dotenv import load_dotenv
from .supabase_client import SUPABASE_CLIENT_
from .embedding import model

load_dotenv()


# Create custom class for supabase Rag
class SupabaseRAG:
    def __init__(
        self,
        table_name: str = "mcp_workflows_rag",  # rag storage table in supabase
        query_name="similarity_search_with_score",  # matching function
        embedding_model=model,
        supabase_client=SUPABASE_CLIENT_,
    ):
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.supabase_client = supabase_client
        self.vectorstore = None
        self.query_name = query_name

    def from_texts(
        self, texts: list[str], metadatas: list[dict] = None
    ) -> SupabaseVectorStore:
        """
        Store data into supabase.

        Args:
            texts: list of input strings
            metadatas: list of dictionary for metadata
        Returns
            SupabaseVectorStore instance
        """
        self.vectorstore = SupabaseVectorStore.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            client=self.supabase_client,
            table_name=self.table_name,
            query_name=self.query_name,
            metadatas=metadatas,
        )
        return self.vectorstore

    def query_rag(self, query_text: str, k: int = 5, filter: dict = None) -> list[dict]:
        """
        Query the RAG system using direct RPC call to bypass LangChain bug.

        Args:
            query_text: The query string
            k: Number of results to return
            filter: Optional metadata filter (e.g., {"source": "docs"})
        Returns:
            list of fetched records
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_query(query_text)

            # Prepare filter parameter
            filter_param = filter if filter else {}

            response = (
                self.supabase_client.rpc(
                    self.query_name,
                    {
                        "query_embeddings": query_embedding,
                        "filter": filter_param,
                    },
                )
                .limit(k)
                .execute()
            )
            return response.data if response.data else []
        except Exception as e:
            print(f"Error during RPC call: {e}")
            return []


if __name__ == "__main__":
    rag = SupabaseRAG()
    query = "process control features from configuration documents"
    texts = [
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
    ]

    # vector_store = rag.from_texts(texts)
    # print(f"Number of documents in vector store: {len(vector_store.store)}")
    results = rag.query_rag(query_text=query)

    print(f"Top {len(results)} results for query '{query}':")
    for idx, doc in enumerate(results):
        print(
            f"Document ID : {doc['id']} | Content: {doc['content']} | Metadata: {doc['metadata']} | Similarity Score: {doc['similarity_score']}"
        )
