import os
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_directory="data/processed/chroma_db"):
        self.persist_directory = persist_directory
        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Use OpenAI Embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="policy_docs_openai",
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks):
        """
        Adds chunks to the ChromaDB collection.
        chunks: list of dicts with 'content' and 'metadata'
        """
        if not chunks:
            return
            
        documents = [c['content'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        
        # Generate unique IDs for each chunk
        source_name = chunks[0]['metadata']['source']
        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks to vector store.")

    def search(self, query, n_results=3):
        """
        Searches the vector store for the query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            # distances are sometimes omitted if include doesn't work, but it should
            distances = results.get('distances', [[0]*n_results])[0]
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                # Chroma uses distance where 0 is perfect match.
                # Let's convert this to a mock confidence score (1 - normalized_distance approx)
                # This depends on the distance metric, but for visualization we'll just store the raw distance.
                dist = distances[i] if distances else 0
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "score": dist,
                    "source_type": "vector"
                })
        return formatted_results
