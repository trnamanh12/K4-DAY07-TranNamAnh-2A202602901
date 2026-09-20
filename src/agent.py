from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức hiện đang rỗng, không tìm thấy thông tin để trả lời."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks = []
        for i, r in enumerate(results, start=1):
            source = (
                r.get("metadata", {}).get("source")
                or r.get("metadata", {}).get("doc_id")
                or r.get("id", f"chunk_{i}")
            )
            context_blocks.append(f"[{i}] (Nguồn: {source}):\n{r.get('content', '')}")
        context_text = "\n\n".join(context_blocks)

        prompt = (
            f"Dưới đây là các đoạn thông tin ngữ cảnh được trích xuất từ cơ sở tri thức:\n\n"
            f"{context_text}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Hướng dẫn:\n"
            f"- Chỉ trả lời dựa trên các thông tin ngữ cảnh được cung cấp ở trên. Tuyệt đối không tự ý bịa đặt thông tin ngoài ngữ cảnh.\n"
            f"- Nếu thông tin trong ngữ cảnh không đủ để trả lời câu hỏi, hãy nêu rõ là không tìm thấy thông tin.\n"
            f"- Khi đưa ra câu trả lời, hãy trích dẫn số thứ tự nguồn tương ứng (ví dụ: [1], [2]) để đảm bảo tính truy vết (Source Traceability).\n\n"
            f"Câu trả lời:"
        )
        return self.llm_fn(prompt)
