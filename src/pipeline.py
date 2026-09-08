import os
import glob
from src.ingestion.parser import PolicyParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.llm import GeminiGenerator

class PolicyLensPipeline:
    def __init__(self, raw_pdf_dir="data/raw_pdfs", model_name="gemini-1.5-flash"):
        self.raw_pdf_dir = raw_pdf_dir
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        self.vector_store = VectorStore()
        self.hybrid_searcher = HybridSearcher(self.vector_store)
        self.llm = GeminiGenerator(model_name=model_name)
        
    def ingest_documents(self):
        """Parses all PDFs, chunks them, and adds to the vector store & BM25."""
        pdf_files = glob.glob(os.path.join(self.raw_pdf_dir, "*.pdf"))
        print(f"Found {len(pdf_files)} PDFs for ingestion.")
        
        all_chunks = []
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}...")
            parser = PolicyParser(pdf_path)
            parsed_data = parser.parse_all()
            chunks = self.chunker.chunk_document(parsed_data)
            all_chunks.extend(chunks)
            
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            self.hybrid_searcher.fit(all_chunks)
            print("Ingestion complete.")
        else:
            print("No data extracted.")

    def query(self, user_query, top_k=3):
        """Handles a user query through the full RAG pipeline."""
        print(f"Searching for: {user_query} with top_k={top_k}")
        
        # 1. Retrieve
        top_chunks = self.hybrid_searcher.search(user_query, top_k=top_k)
        if not top_chunks:
            return "No relevant information found in the policy documents."
            
        # 2. Generate
        answer = self.llm.generate_rag_answer(user_query, top_chunks)
        
        # 3. Format response with sources
        sources = [c['metadata']['source'] for c in top_chunks]
        unique_sources = list(set(sources))
        
        return {
            "answer": answer,
            "sources": unique_sources,
            "chunks_used": top_chunks
        }

if __name__ == "__main__":
    pipeline = PolicyLensPipeline()
    # pipeline.ingest_documents()
    # result = pipeline.query("What is the penalty for plagiarism?")
    # print("Answer:", result['answer'])
