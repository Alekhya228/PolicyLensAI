class TextChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, parsed_data):
        """
        Chunks the parsed text and tables.
        Returns a list of dicts with 'content' and 'metadata'.
        """
        chunks = []
        source = parsed_data['source']
        
        # Chunk text
        for page_data in parsed_data['text']:
            page_num = page_data['page']
            text = page_data['text']
            
            # Simple character-based sliding window
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": source,
                        "last_updated": parsed_data.get('last_updated', 'Unknown'),
                        "page": page_num,
                        "type": "text"
                    }
                })
                start += self.chunk_size - self.overlap
                
        # Include tables as individual chunks
        for table_data in parsed_data['tables']:
            page_num = table_data['page']
            table_text = table_data['table']
            chunks.append({
                "content": f"TABLE_CONTENT:\n{table_text}",
                "metadata": {
                    "source": source,
                    "last_updated": parsed_data.get('last_updated', 'Unknown'),
                    "page": page_num,
                    "type": "table"
                }
            })
            
        return chunks
