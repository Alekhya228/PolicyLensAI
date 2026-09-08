from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearcher:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.chunks = []
        self.bm25 = None
        
    def fit(self, chunks):
        """Fits the BM25 model on the given chunks."""
        self.chunks = chunks
        tokenized_corpus = [c['content'].lower().split(" ") for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def bm25_search(self, query, top_k=3):
        if not self.bm25:
            return []
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                chunk_copy = dict(self.chunks[i])
                chunk_copy['score'] = scores[i]
                chunk_copy['source_type'] = "bm25"
                results.append(chunk_copy)
        return results
        
    def search(self, query, top_k=3):
        """
        Performs hybrid search by getting results from both vector store and BM25,
        then combining them.
        """
        vector_results = self.vector_store.search(query, n_results=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)
        
        # Simple deduplication based on content (or could use IDs)
        combined = []
        seen = set()
        
        for res in vector_results + bm25_results:
            content_hash = hash(res['content'])
            if content_hash not in seen:
                seen.add(content_hash)
                combined.append(res)
                
        return combined[:top_k]
