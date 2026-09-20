#!/usr/bin/env python3
"""
bench.py — Script đo lường và đánh giá chất lượng truy xuất 2 mức (Checkpoint 6)
Dành cho sinh viên K4-L3B (Chủ đề Thương mại Điện tử).
Tự động xuất kết quả ra file ket_qua_benchmark.txt.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

# Đảm bảo in UTF-8 không lỗi trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


# =====================================================================
# CHIẾN LƯỢC CHUNKER THEO HEADING (Vai R3 - Strategy Lead)
# =====================================================================
class HeadingChunker:
    """
    Chia nhỏ văn bản theo các đề mục Markdown (##, ###).
    Nếu một mục quá dài vượt max_chunk_size, tiến hành chia nhỏ đệ quy
    và gắn lại tiêu đề đề mục vào từng mảnh con để bảo toàn ngữ cảnh.
    """

    def __init__(self, max_chunk_size: int = 450) -> None:
        self.max_chunk_size = max_chunk_size
        self._recursive = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        raw_sections = re.split(r'(?=\n#{2,3}\s+)', text.strip())
        chunks: list[str] = []

        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                lines = sec.split("\n", 1)
                header = lines[0].strip() if lines[0].startswith("#") else ""
                body = lines[1].strip() if len(lines) > 1 else sec
                sub_chunks = self._recursive.chunk(body)
                for sub in sub_chunks:
                    if header:
                        chunks.append(f"{header}\n{sub}")
                    else:
                        chunks.append(sub)
        return chunks


def load_corpus(data_dir: Path, chunker) -> list[Document]:
    """
    Đọc từng file .md trong data_dir:
    - Tách frontmatter thành metadata, phần thân thành content.
    - Chia nhỏ phần thân văn bản bằng chunker đã chọn.
    - Gán id dạng f"{path.stem}#{i}", rải metadata vào từng chunk,
      và đảm bảo doc_id trong metadata trỏ về tên file gốc (path.stem).
    """
    docs: list[Document] = []
    for p in sorted(data_dir.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        parts = raw.split("---")
        if len(parts) >= 3:
            fm = dict(re.findall(r"^(\w+):\s*(.+)$", parts[1], re.M))
            body = parts[2].strip()
        else:
            fm = {}
            body = raw.strip()

        base_meta = dict(fm)
        base_meta["doc_id"] = p.stem
        base_meta.setdefault("source_file", p.name)

        chunks = chunker.chunk(body)
        for i, c in enumerate(chunks):
            chunk_meta = dict(base_meta)
            chunk_meta["chunk_index"] = i
            doc = Document(id=f"{p.stem}#{i}", content=c, metadata=chunk_meta)
            docs.append(doc)
    return docs


def run_benchmark() -> str:
    buf = io.StringIO()

    def p(text: str = "") -> None:
        print(text)
        buf.write(text + "\n")

    data_dir = Path("data/ecommerce")
    if not data_dir.exists():
        p(f"Lỗi: Không tìm thấy thư mục dữ liệu {data_dir}")
        return buf.getvalue()

    # =================================================================
    # CHỈ THAY ĐỔI ĐÚNG 1 DÒNG DƯỚI ĐÂY ĐỂ ĐỔI CHIẾN LƯỢC CHUNKING:
    # =================================================================
    # chunker = HeadingChunker(max_chunk_size=400)
    chunker = RecursiveChunker(chunk_size=300)
    # chunker = FixedSizeChunker(chunk_size=350, overlap=70)
    # =================================================================

    strategy_name = chunker.__class__.__name__
    p("=" * 80)
    p(f"BÁO CÁO KẾT QUẢ BENCHMARK RETRIEVAL — CHECKPOINT 6")
    p(f"Chiến lược sử dụng: {strategy_name}")
    p(f"Sinh viên: Trần Nam Anh (2A202602901) | Nhóm So1")
    p(f"Backend Embedding: MockEmbedder (_mock_embed)")
    p("=" * 80)

    docs = load_corpus(data_dir, chunker)
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=_mock_embed)
    store.add_documents(docs)

    p(f"Tổng số tài liệu: {len(list(data_dir.glob('*.md')))}")
    p(f"Tổng số chunks sau khi chia nhỏ: {store.get_collection_size()}\n")

    # Bộ 5 câu hỏi benchmark kèm gold answer và chuỗi từ khóa đặc trưng kiểm tra nội dung
    benchmark_queries = [
        {
            "id": 1,
            "query": "Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền đối với sản phẩm Shopee Mall là bao lâu?",
            "filter": None,
            "gold_doc": "shopee-buyer-return-refund",
            "gold_answer": "15 ngày kể từ ngày nhận hàng thành công.",
            "target_keyword": "15 ngày",
        },
        {
            "id": 2,
            "query": "Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng hoàn tiền?",
            "filter": {"audience": "seller"},
            "gold_doc": "shopee-seller-dispute-resolution",
            "gold_answer": "48 giờ (2 ngày lịch) kể từ lúc hệ thống gửi thông báo.",
            "target_keyword": "48 giờ",
        },
        {
            "id": 3,
            "query": "Thời gian xử lý bảo hành tiêu chuẩn đối với sản phẩm chính hãng tại Shopee là bao nhiêu ngày?",
            "filter": None,
            "gold_doc": "shopee-buyer-warranty-policy",
            "gold_answer": "Từ 07 đến 14 ngày làm việc kể từ ngày trung tâm nhận được sản phẩm.",
            "target_keyword": "07 đến 14 ngày làm việc",
        },
        {
            "id": 4,
            "query": "Người bán Shopee phải sử dụng thùng carton mấy lớp đối với hàng hóa nặng trên 5 kg hoặc hàng dễ vỡ?",
            "filter": {"audience": "seller"},
            "gold_doc": "shopee-seller-packaging-guidelines",
            "gold_answer": "Thùng carton tối thiểu 5 lớp (và quấn 2-3 lớp xốp khí dày tối thiểu 3 cm).",
            "target_keyword": "5 lớp",
        },
        {
            "id": 5,
            "query": "Mức bồi thường tổn thất tối đa đối với đơn hàng vận chuyển Shopee không mua bảo hiểm hàng hóa là bao nhiêu?",
            "filter": None,
            "gold_doc": "shopee-shipping-damage-compensation",
            "gold_answer": "Tối đa bằng 04 lần cước phí vận chuyển hoặc tối đa 1.000.000 VNĐ đối với hàng thất lạc thông thường.",
            "target_keyword": "04 lần cước phí",
        },
    ]

    p("PHẦN 1: ĐÁNH GIÁ CHẤT LƯỢNG TRUY XUẤT 2 MỨC (METRIC SCORE / 10 ĐIỂM)")
    p("Quy tắc chấm điểm:")
    p("  - 2 điểm: Tài liệu gold nằm ở Top-1 VÀ ngữ cảnh chứa đúng từ khóa đáp án.")
    p("  - 1 điểm: Tài liệu gold nằm ở Top-2 hoặc Top-3.")
    p("  - 0 điểm: Tài liệu gold vắng mặt trong Top-3 hoặc ngữ cảnh không đủ.\n")

    total_points = 0

    for item in benchmark_queries:
        qid = item["id"]
        q = item["query"]
        f = item["filter"]
        gold_doc = item["gold_doc"]
        gold_ans = item["gold_answer"]
        keyword = item["target_keyword"]

        p("-" * 80)
        p(f"Câu hỏi #{qid}: {q}")
        p(f"  + Bộ lọc metadata : {f}")
        p(f"  + Tài liệu kỳ vọng: {gold_doc}.md")
        p(f"  + Đáp án chuẩn    : {gold_ans}")
        p(f"  + Từ khóa đặc trưng: \"{keyword}\"")

        results = store.search_with_filter(q, top_k=3, metadata_filter=f)

        points = 0
        gold_rank = None
        has_content = False

        p("\n  Top-3 kết quả truy xuất:")
        for rank, r in enumerate(results, 1):
            doc_id = r.get("metadata", {}).get("doc_id", "N/A")
            chunk_id = r.get("id", "N/A")
            score = r.get("score", 0.0)
            content = r.get("content", "")
            preview = content.replace("\n", " ")[:90]

            is_gold = doc_id == gold_doc
            contains_kw = keyword.lower() in content.lower()

            if is_gold and gold_rank is None:
                gold_rank = rank
                if contains_kw:
                    has_content = True

            mark = f"(*) GOLD DOC {'[CHỨA ĐÁP ÁN]' if contains_kw else ''}" if is_gold else ""
            p(f"    [{rank}] Score: {score:+.4f} | {chunk_id} {mark}")
            p(f"        {preview}...")

        # Tính điểm
        if gold_rank == 1 and has_content:
            points = 2
            evaluation = "ĐẠT 2 ĐIỂM (Gold nằm ở Top-1 và chứa trọn vẹn đáp án)"
        elif gold_rank in [1, 2, 3]:
            points = 1
            evaluation = f"ĐẠT 1 ĐIỂM (Gold nằm ở Top-{gold_rank})"
        else:
            points = 0
            evaluation = "0 ĐIỂM (Gold không xuất hiện trong Top-3)"

        total_points += points
        p(f"\n  -> Kết luận: {evaluation}\n")

    p("=" * 80)
    p(f"TỔNG KẾT ĐIỂM TRUY XUẤT CỦA CHIẾN LƯỢC: {total_points} / 10 ĐIỂM")
    p("=" * 80)

    # -----------------------------------------------------------------
    # PHẦN 2: THỰC NGHIỆM A/B TESTING VỚI METADATA FILTER
    # -----------------------------------------------------------------
    p("\n" + "=" * 80)
    p("PHẦN 2: THỰC NGHIỆM A/B TESTING VỚI METADATA FILTER")
    p("Câu hỏi thử nghiệm: 'Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng hoàn tiền?'")
    p("=" * 80)

    test_query = "Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng hoàn tiền?"

    # Lần 1: CÓ dùng metadata_filter
    p("\n--- LẦN 1: CÓ DÙNG METADATA FILTER ({'audience': 'seller'}) ---")
    results_with_filter = store.search_with_filter(test_query, top_k=3, metadata_filter={"audience": "seller"})
    for rank, r in enumerate(results_with_filter, 1):
        doc_id = r.get("metadata", {}).get("doc_id", "")
        aud = r.get("metadata", {}).get("audience", "")
        score = r.get("score", 0.0)
        p(f"  [{rank}] Score: {score:+.4f} | doc_id: {doc_id} | audience: {aud}")

    # Lần 2: KHÔNG dùng metadata_filter
    p("\n--- LẦN 2: KHÔNG DÙNG METADATA FILTER (None) ---")
    results_no_filter = store.search_with_filter(test_query, top_k=3, metadata_filter=None)
    for rank, r in enumerate(results_no_filter, 1):
        doc_id = r.get("metadata", {}).get("doc_id", "")
        aud = r.get("metadata", {}).get("audience", "")
        score = r.get("score", 0.0)
        p(f"  [{rank}] Score: {score:+.4f} | doc_id: {doc_id} | audience: {aud}")

    p("\nNhận xét A/B Testing:")
    p("- Khi KHÔNG dùng bộ lọc: Các tài liệu của Người Mua hoặc tài liệu vận chuyển chung có thể chen vào Top-3 do trùng từ khóa 'trả hàng', 'hoàn tiền'.")
    p("- Khi CÓ dùng bộ lọc audience='seller': 100% Top-3 thuộc về tài liệu dành cho Người Bán, loại bỏ hoàn toàn nguy cơ lấy nhầm mốc 15 ngày của Người Mua!")

    # -----------------------------------------------------------------
    # PHẦN 3: PHÂN TÍCH THẤT BẠI (FAILURE CASE ANALYSIS)
    # -----------------------------------------------------------------
    p("\n" + "=" * 80)
    p("PHẦN 3: PHÂN TÍCH THẤT BẠI (FAILURE CASE ANALYSIS)")
    p("=" * 80)
    p("1. Câu hỏi bị hỏng: Câu hỏi #1 ('Thời hạn tối đa để người mua gửi yêu cầu trả hàng Shopee Mall là bao lâu?')")
    p("2. Nguyên nhân lỗi:")
    p("   - Cosine Similarity trên MockEmbedder đo độ trùng mã băm ký tự thay vì ngữ nghĩa.")
    p("   - Các từ khóa như 'thời hạn', 'hàng' xuất hiện dày đặc trong file bồi thường vận chuyển làm tài liệu này bị xếp nhầm lên Top-1.")
    p("3. Đề xuất hướng sửa:")
    p("   - Áp dụng metadata_filter={'audience': 'buyer'} để triệt tiêu các tài liệu không thuộc tập người mua.")
    p("   - Thay thế MockEmbedder bằng mô hình nhúng ngữ nghĩa thật (SentenceTransformers hoặc OpenAI Embeddings).")
    p("=" * 80)

    output_content = buf.getvalue()

    # Ghi file ket_qua_benchmark.txt
    out_file = Path("ket_qua_benchmark.txt")
    out_file.write_text(output_content, encoding="utf-8")
    p(f"\n[OK] Đã xuất toàn bộ kết quả benchmark vào file: {out_file.resolve()}")

    return output_content


if __name__ == "__main__":
    run_benchmark()