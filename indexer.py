import os
import re
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAPERS_METADATA = {
    "attention_is_all_you_need.pdf": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7",
    },
    "moshi.pdf": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1",
    },
    "dynamic_chunking.pdf": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2",
    }
}

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

def load_paper_pages(base_dir="."):
    documents = []
    for filename, meta in PAPERS_METADATA.items():
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing required paper: {filepath}")
        doc = fitz.open(filepath)
        for page_idx in range(len(doc)):
            page_text = doc[page_idx].get_text()
            # Clean up whitespace
            cleaned_text = re.sub(r"\s+", " ", page_text).strip()
            if cleaned_text:
                documents.append({
                    "paper": meta["title"],
                    "arxiv_id": meta["arxiv_id"],
                    "page": page_idx + 1,
                    "text": cleaned_text
                })
        doc.close()
    return documents

class PaperRetriever:
    def __init__(self, documents, chunk_size=1800, overlap=300, top_k=3, rerank=False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.rerank = rerank
        self.chunks = []
        self._build_chunks(documents)
        self._build_index()

    def _build_chunks(self, documents):
        self.chunks = []
        chunk_id = 0
        for doc in documents:
            text = doc["text"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    self.chunks.append({
                        "chunk_id": chunk_id,
                        "paper": doc["paper"],
                        "arxiv_id": doc["arxiv_id"],
                        "page": doc["page"],
                        "text": chunk_text
                    })
                    chunk_id += 1
                start += max(100, self.chunk_size - self.overlap)

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.85
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [c["text"] for c in self.chunks]
        )

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k
        
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        candidate_count = top_k * 3 if self.rerank else top_k
        best_indices = scores.argsort()[::-1][:candidate_count]

        candidates = []
        for idx in best_indices:
            if scores[idx] <= 0:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(scores[idx])
            candidates.append(item)

        if self.rerank and candidates:
            # Lexical term coverage + TF-IDF score
            q_words = {
                w for w in re.findall(r"\b[a-zA-Z]+\b", query.lower())
                if w not in ENGLISH_STOP_WORDS
            }
            for item in candidates:
                p_words = set(re.findall(r"\b[a-zA-Z]+\b", item["text"].lower()))
                overlap = len(q_words & p_words)
                coverage = overlap / max(1, len(q_words))
                item["final_score"] = 0.5 * item["score"] + 0.5 * coverage
            candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return candidates[:top_k]

if __name__ == "__main__":
    docs = load_paper_pages()
    print(f"Loaded {len(docs)} pages from 3 papers.")
    retriever = PaperRetriever(docs, chunk_size=1800, overlap=300, top_k=3, rerank=False)
    print(f"Created {len(retriever.chunks)} chunks.")
    res = retriever.retrieve("What is the Transformer architecture?")
    for r in res:
        print(f"[{r['paper']} p.{r['page']}] score: {r['score']:.4f}")
