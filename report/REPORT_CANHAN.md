# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Nam Anh
**Nhóm:** So1
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding cùng chỉ về một hướng trong không gian vector nhiều chiều, thể hiện hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa và nội dung ngữ cảnh, bất kể độ dài hay từ ngữ được sử dụng có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Cậu bé đang dắt chú cún cưng đi dạo trong công viên.
- Câu B: Đứa trẻ dẫn con chó nhỏ dạo chơi ngoài vườn hoa.
- Tại sao tương đồng: Hai câu sử dụng các từ vựng hoàn toàn khác nhau ("cậu bé" vs "đứa trẻ", "chú cún cưng" vs "con chó nhỏ", "đi dạo trong công viên" vs "dạo chơi ngoài vườn hoa"), nhưng cùng diễn đạt một hành động và ý nghĩa thực tế. Embedding nắm bắt tốt ngữ nghĩa thay vì chỉ so khớp từ vựng bề mặt.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Mô hình mạng nơ-ron tích chập xử lý dữ liệu hình ảnh rất hiệu quả.
- Câu B: Công thức làm bánh bông lan phô mai cần bốn quả trứng gà.
- Tại sao khác: Hai câu thuộc hai lĩnh vực tri thức hoàn toàn tách biệt (khoa học máy tính / thị giác máy tính vs ẩm thực / làm bánh), không có liên hệ ngữ nghĩa nào nên hai vector chỉ theo hai hướng khác nhau trong không gian biểu diễn.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc trực tiếp vào độ dài (magnitude) của vector, do đó văn bản dài hoặc tần suất từ nhiều sẽ tạo khoảng cách lớn dù cùng nội dung. Ngược lại, cosine similarity chỉ đo góc giữa hai vector (chuẩn hóa theo độ dài), tập trung thuần túy vào định hướng ngữ nghĩa cốt lõi của văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: $\text{Số chunks} = \lceil \frac{\text{độ\_dài} - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \rceil = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.111... \rceil = 23$.

> *Đáp án:* 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số lượng chunk tăng lên thành $\lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = 25$ chunks (tăng thêm 2 chunks). Chúng ta chấp nhận tốn thêm chunk để overlap lớn hơn nhằm bảo toàn ngữ cảnh tại ranh giới cắt, tránh bị đứt mạch thông tin quan trọng nằm vắt ngang giữa 2 chunk liền kề, giúp việc truy xuất và sinh câu trả lời của LLM chính xác hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex lookbehind `r"(?<=[.!?])\s+"` để tách câu ngay sau dấu câu kết thúc mà vẫn giữ nguyên được dấu câu, sau đó gom mỗi `max_sentences_per_chunk` câu thành một chunk và strip khoảng trắng. Xử lý an toàn trường hợp text rỗng (trả về `[]`), đồng thời nhận diện được hạn chế (edge case) chưa xử lý: các chữ viết tắt (như "TS.", "v.v.") và số thập phân ("3.14") sẽ bị tách nhầm câu do dấu chấm kèm khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán kết hợp hai chiều: đệ quy xuống sâu theo thứ tự ưu tiên các separator `["\n\n", "\n", ". ", " ", ""]` khi đoạn văn bản vượt quá `chunk_size`, và gom (merge) các mảnh nhỏ liền kề lại cho tới khi sát `chunk_size` để tránh sinh ra hàng loạt chunk vụn. Ba trường hợp dừng (base cases) gồm: chuỗi rỗng (`[]`), chuỗi có độ dài $\le$ `chunk_size` (`[text]`), và khi danh sách separators rỗng (`separators=[]`) thì fallback cắt cứng theo từng khối `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ in-memory dưới dạng danh sách các dict chuẩn hóa gồm `id`, `content`, bản sao `metadata` (được bổ sung `doc_id` trỏ về file gốc), và `embedding`. Tách riêng logic tìm kiếm thành helper `_search_records` để tính tích vô hướng (dot product — tương đương cosine do vector đã chuẩn hóa) giữa query embedding và từng record, sắp xếp giảm dần theo score và lấy top_k (đã loại bỏ vector embedding để output gọn gàng).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc phải lọc (pre-filter) metadata trước rồi mới đưa tập ứng viên hợp lệ vào `_search_records`. Nếu lấy top-k trước rồi mới lọc, k slot có thể bị chiếm hết bởi tài liệu không khớp metadata dẫn đến trả về 0 kết quả dù tài liệu đúng vẫn tồn tại trong store. Hàm `delete_document` loại bỏ mọi record có `metadata['doc_id']` hoặc `id` khớp với `doc_id` cần xóa, trả về `True` nếu số lượng phần tử giảm đi và `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Hoạt động theo mô hình RAG với 3 bước: Kiểm tra an toàn (nếu store rỗng lập tức trả về thông báo, không gọi LLM vô ích) $\rightarrow$ Truy xuất top-k chunk liên quan bằng `search` $\rightarrow$ Dựng prompt đưa ngữ cảnh có đánh số thứ tự `[1] [2]...` kèm nguồn (`source` / `doc_id`) nhằm đảm bảo tính truy vết nguồn (Source Traceability), kèm chỉ dẫn nghiêm ngặt yêu cầu LLM chỉ trả lời dựa vào ngữ cảnh và trích dẫn số nguồn tương ứng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home284/284-home/VIN/lab07/K4-DAY07-TranNamAnh-2A202602901
plugins: anyio-4.15.0
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Chính sách trả hàng và hoàn tiền Shopee | Quy định đổi trả sản phẩm cho người mua | cao | -0.1835 | Sai |
| 2 | Người bán phản hồi khiếu nại trong 48 giờ | Thời hạn người bán xử lý yêu cầu là 2 ngày | cao | +0.1147 | Đúng |
| 3 | Hướng dẫn đóng gói hàng dễ vỡ bằng thùng carton | Thời gian xử lý bảo hành điện thoại từ 7 đến 14 ngày | thấp | +0.2360 | Sai |
| 4 | Mức bồi thường tổn thất hàng hóa khi vận chuyển | Công thức làm bánh bông lan phô mai Nhật Bản | thấp | +0.0386 | Đúng |
| 5 | Chính sách đổi trả hàng hóa Shopee Mall | Chính sách đổi trả hàng hóa Shopee Mall | cao | +1.0000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là Cặp 1 và Cặp 3: Ở Cặp 1, hai câu có ý nghĩa tương đương nhau về mặt con người nhưng `MockEmbedder` lại cho điểm âm (-0.1835); trong khi ở Cặp 3, hai câu về hai chủ đề khác biệt (đóng gói vs bảo hành) lại nhận điểm dương khá cao (+0.2360). Điều này chỉ ra rằng `MockEmbedder` chỉ băm ký tự/token cục bộ (lexical hashing) chứ hoàn toàn không học được không gian ngữ nghĩa thực thụ (semantic representation). Trong hệ thống RAG thực tế, bắt buộc phải dùng các mô hình ngôn ngữ như SentenceTransformers để hai câu cùng nghĩa được kéo gần nhau trong không gian vector.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn tối đa để người mua gửi yêu cầu trả hàng và hoàn tiền đối với sản phẩm Shopee Mall là bao lâu? | `shopee-shipping-damage-compensation#9`: Video quay liên tục từ lúc kiểm tra tem niêm phong kiện hàng... | +0.3133 | Không (Tài liệu bồi thường vận chuyển chen lên) | Agent trích dẫn tài liệu bồi thường vận chuyển và hướng dẫn quay video mở hàng, không chứa mốc 15 ngày Shopee Mall. |
| 2 | Người bán có bao nhiêu thời gian để phản hồi khi người mua yêu cầu trả hàng hoàn tiền? *(Filter `audience: seller`)* | `shopee-seller-dispute-resolution#5`: Quy định nếu hàng hoàn về bị hư hại, tráo đổi linh kiện... | +0.2517 | Có (Top-1, cùng tài liệu quy định Người Bán) | Agent trả lời dựa trên tài liệu giải quyết tranh chấp người bán, nêu rõ quy trình và chế tài xử lý khiếu nại. |
| 3 | Thời gian xử lý bảo hành tiêu chuẩn đối với sản phẩm chính hãng tại Shopee là bao nhiêu ngày? | `shopee-buyer-warranty-policy#3`: Sản phẩm gặp sự cố hư hỏng hoặc lỗi kỹ thuật do nhà sản xuất... | +0.2687 | Có (Top-1, tài liệu bảo hành chính hãng) | Agent nêu các điều kiện bảo hành và thời gian kiểm tra xử lý lỗi sản phẩm từ tài liệu bảo hành. |
| 4 | Người bán Shopee phải sử dụng thùng carton mấy lớp đối với hàng hóa nặng trên 5 kg hoặc hàng dễ vỡ? *(Filter `audience: seller`)* | `shopee-seller-packaging-guidelines#6`: Thời hạn chuẩn bị hàng và bàn giao cho bưu cục Shopee... | +0.2113 | Có (Top-1, tài liệu hướng dẫn đóng gói) | Agent trích dẫn hướng dẫn chuẩn bị hàng và quy cách đóng gói dành cho Người Bán Shopee. |
| 5 | Mức bồi thường tổn thất tối đa đối với đơn hàng vận chuyển Shopee không mua bảo hiểm hàng hóa là bao nhiêu? | `shopee-prohibited-items-policy#2`: Danh mục chất kích thích, thuốc lá, hàng hóa cấm vận chuyển... | +0.3053 | Không (Bị tài liệu hàng cấm chiếm Top-1 do trùng từ khóa) | Agent trích dẫn cảnh báo danh mục hàng cấm thay vì điều khoản mức tiền đền bù vận chuyển. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 3 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua so sánh với chiến lược `HeadingChunker` của bạn Minh, tôi nhận thấy với các tài liệu quy định pháp lý/chính sách TMĐT, việc chia nhỏ theo cấu trúc tiêu đề (Header/Section) giữ trọn vẹn ngữ cảnh của từng điều khoản tốt hơn nhiều so với `RecursiveChunker` (vốn có thể cắt rời tiêu đề với danh sách nội dung bên dưới). Ngoài ra, bộ lọc siêu dữ liệu (`metadata_filter`) là công cụ cực kỳ lợi hại để triệt tiêu tài liệu gây nhiễu trong nền tảng có nhiều đối tượng tương tác như sàn TMĐT.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
