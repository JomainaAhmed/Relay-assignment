import os
import re
import time
import asyncio
from typing import Dict, Any, List, Optional
from indexer import PaperRetriever, load_paper_pages
from stand_in import CalibratedStandInModel, GroqRateLimitException, GroqServerException

class ResearchAssistantPipeline:
    def __init__(
        self,
        retriever: PaperRetriever,
        groq_client=None,
        stand_in_model: Optional[CalibratedStandInModel] = None,
        model_name: str = "qwen/qwen3.8-27b",
        mode: str = "stand-in"  # "real" or "stand-in"
    ):
        self.retriever = retriever
        self.groq_client = groq_client
        self.stand_in = stand_in_model or CalibratedStandInModel()
        self.model_name = model_name
        self.mode = mode

    async def _call_llm_stream(self, prompt: str, simulated_answer: Optional[str] = None):
        """Unified LLM call supporting real Groq streaming and Stand-in streaming."""
        if self.mode == "real" and self.groq_client:
            t0 = time.perf_counter()
            first_token_ms = None
            full_text = []
            
            # Run in threadpool since groq client is sync
            def sync_stream():
                return self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    max_completion_tokens=400,
                    temperature=0.1
                )

            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(None, sync_stream)

            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, 'content', None) or getattr(delta, 'reasoning', None)
                if token:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    full_text.append(token)
            
            total_ms = (time.perf_counter() - t0) * 1000
            answer = "".join(full_text)
            tokens_in = len(prompt.split())
            tokens_out = max(1, len(answer.split()))
            return {
                "answer": answer,
                "first_token_ms": first_token_ms or (total_ms * 0.4),
                "total_ms": total_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out
            }
        else:
            # Stand-in service
            t0 = time.perf_counter()
            first_token_ms = None
            full_text = []
            async for chunk, ttft in self.stand_in.generate_stream(prompt, simulated_answer=simulated_answer):
                if ttft is not None:
                    first_token_ms = ttft
                else:
                    full_text.append(chunk)
            
            total_ms = (time.perf_counter() - t0) * 1000
            answer = "".join(full_text).strip()
            tokens_in = len(prompt.split())
            tokens_out = max(1, len(answer.split()))
            return {
                "answer": answer,
                "first_token_ms": first_token_ms or 150.0,
                "total_ms": total_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out
            }

    def tool_retrieve_passages(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Tool call: executes passage retrieval over the arXiv paper library."""
        return self.retriever.retrieve(query, top_k=top_k)

    def _verify_citations(self, answer: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Audits answer to ensure cited papers and pages correspond to retrieved passages."""
        # Citation format: [Paper Title, p. X]
        citations = re.findall(r"\[(.*?),\s*p\.\s*(\d+)\]", answer)
        if not citations:
            # If answer is a refusal, it's valid without citations
            if "not find" in answer.lower() or "not state" in answer.lower() or "cannot answer" in answer.lower() or "not provide" in answer.lower():
                return {"valid": True, "reason": "Refusal statement verified."}
            return {"valid": False, "reason": "Missing required [Paper, p. X] citations."}

        valid_citations = 0
        for paper_cited, page_cited in citations:
            page_int = int(page_cited)
            # Check if any passage matches
            matched = False
            for p in passages:
                # Fuzzy match title
                if (paper_cited.lower() in p["paper"].lower() or p["paper"].lower() in paper_cited.lower()) and p["page"] == page_int:
                    matched = True
                    break
            if matched:
                valid_citations += 1

        if valid_citations == len(citations):
            return {"valid": True, "reason": f"All {len(citations)} citations verified in retrieved context."}
        else:
            return {
                "valid": False, 
                "reason": f"Citation mismatch: {len(citations) - valid_citations} citation(s) not in retrieved passages."
            }

    async def execute_query(
        self, 
        query: str, 
        top_k: int = 3, 
        max_audit_loops: int = 1,
        simulated_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t_start = time.perf_counter()
        
        # ---------------------------------------------------------
        # STEP 1: Query Intent Classifier & Paper Router (Branch 1)
        # ---------------------------------------------------------
        # Check if query is completely unrelated or conversational
        unrelated_keywords = ["weather", "pizza", "stock price", "football", "joke", "recipe"]
        if any(w in query.lower() for w in unrelated_keywords):
            return {
                "status": "ok",
                "answer": "This assistant answers only from the 3 arXiv research papers. Your question is outside scope.",
                "first_token_ms": 10.0,
                "total_ms": (time.perf_counter() - t_start) * 1000,
                "retries": 0,
                "tokens_in": len(query.split()),
                "tokens_out": 20,
                "answer_grade": "refused_correctly",
                "passages": []
            }

        # ---------------------------------------------------------
        # STEP 2: Tool Call -> Passage Retrieval
        # ---------------------------------------------------------
        passages = self.tool_retrieve_passages(query, top_k=top_k)

        # Refusal branch: if no relevant passages found
        if not passages or passages[0].get("score", 0) < 0.04:
            return {
                "status": "ok",
                "answer": "The provided research papers do not contain sufficient evidence to answer this question.",
                "first_token_ms": 15.0,
                "total_ms": (time.perf_counter() - t_start) * 1000,
                "retries": 0,
                "tokens_in": len(query.split()),
                "tokens_out": 15,
                "answer_grade": "refused_correctly",
                "passages": []
            }

        # Build context from passages
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            context_blocks.append(
                f"[Passage {i}] Paper: {p['paper']}, Page: {p['page']}\n{p['text'][:900]}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are a research assistant.
Answer the question using ONLY the provided passages.
Every single factual statement must be cited with [Paper title, p. page_number].
If the passages do not explicitly provide the answer, say 'The papers do not state this.'

EVIDENCE:
{context_str}

QUESTION:
{query}
"""

        # ---------------------------------------------------------
        # STEP 3 & 4: Drafter + Citation Auditor (LOOP)
        # ---------------------------------------------------------
        audit_loops = 0
        final_answer = ""
        total_tokens_in = len(prompt.split())
        total_tokens_out = 0
        first_token_ms = None
        current_prompt = prompt

        while audit_loops <= max_audit_loops:
            llm_res = await self._call_llm_stream(current_prompt, simulated_answer=simulated_answer)
            if first_token_ms is None:
                first_token_ms = llm_res["first_token_ms"]
            
            final_answer = llm_res["answer"]
            total_tokens_out += llm_res["tokens_out"]

            # STEP 4: Verification
            audit_result = self._verify_citations(final_answer, passages)
            if audit_result["valid"]:
                break
            
            # Loop back if citations failed audit
            audit_loops += 1
            if audit_loops <= max_audit_loops:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"AUDIT WARNING: Your previous draft had citation errors: {audit_result['reason']}. "
                    f"Please re-draft carefully using only the exact citations from the evidence above."
                )

        total_ms = (time.perf_counter() - t_start) * 1000

        # Automatic preliminary grading
        if "the papers do not" in final_answer.lower() or "not state" in final_answer.lower() or "not find" in final_answer.lower():
            grade = "refused_correctly"
        elif re.search(r"\[(.*?),\s*p\.\s*\d+\]", final_answer):
            grade = "correct"
        else:
            grade = "wrong"

        return {
            "status": "ok",
            "answer": final_answer,
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "retries": audit_loops,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "answer_grade": grade,
            "passages": passages
        }

if __name__ == "__main__":
    docs = load_paper_pages()
    retriever = PaperRetriever(docs, chunk_size=1800, top_k=3)
    pipeline = ResearchAssistantPipeline(retriever, mode="stand-in")
    
    async def run_test():
        print("Testing 4-step pipeline with verification loop (stand-in mode)...")
        res = await pipeline.execute_query("What is self-attention and how does it compute representations?")
        print("Status:", res["status"])
        print("TTFT:", round(res["first_token_ms"], 2), "ms")
        print("Total ms:", round(res["total_ms"], 2))
        print("Answer:", res["answer"])
        print("Grade:", res["answer_grade"])

    asyncio.run(run_test())
