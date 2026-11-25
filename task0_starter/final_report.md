# Final Report: Automated Literature Review Generation and Analysis

## 1. Base Paper & Topic
**Base Paper:** *"Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images"* (Huang et al., 2024)
**Topic:** Generalizable vision-language models for zero-shot/few-shot anomaly detection in medical images.

## 2. SurveyGen Methodology
Our methodology followed a two-phase pipeline to autonomously generate a literature review on the target topic:
1. **Retrieval & Filtering:** Scrape academic databases for papers matching the topic, embed their abstracts to rank them by semantic similarity, and manually filter the top results to curate a highly relevant reading list.
2. **Generation & Evaluation:** Feed the curated abstracts into a Large Language Model (LLM) instructed to synthesize them thematically, and finally compare the generated review against a human-written baseline (the base paper's related work section).

## 3. Paper Retrieval
We developed a custom Python script (`scholar_scraper.py`) querying the arXiv API for papers containing keywords related to "medical anomaly detection", "vision-language models", "CLIP", and "zero-shot/few-shot". This yielded a raw dataset of 78 unique papers, saved to `llm_healthcare_papers.csv`.

## 4. Ranking/Filtering
We utilized the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode both the target topic and the 78 retrieved abstracts. By computing cosine similarity, we ranked the papers from most to least relevant. From the top 20, we performed a manual inspection of titles and abstracts, removing false positives to arrive at a final, highly curated list of 14 papers.

## 5. Generated Literature Review
The 14 abstracts were compiled into a single prompt alongside strict generation constraints and fed into an LLM via the AgentRouter API (using a manual web-interface fallback to bypass API proxy serialization issues). 

*Note on Citation Mapping:* The LLM successfully synthesized all 14 papers. While we provided them numbered 1-14 in our prompt, the LLM reordered the citation numbers [1]–[14] in the generated text to match their order of appearance in the review. The mapping is verified and accurate (e.g., SD-MAD is [1], MediCLIP is [2], AnomalyCLIP is [3], etc.).

## 6. Comparison with Base Paper Related Work

### Comparison Table

| Generated Review Paper | Acronym / Author | In Base Paper Related Work? | Same Topic? |
| :--- | :--- | :--- | :--- |
| SD-MAD: Sign-Driven Few-shot Multi-Anomaly Detection... | SD-MAD | No | Yes (Fine-grained AD) |
| Adapting CLIP for Few-shot Medical Image Anomaly... | MediCLIP | No | Yes (Adapters) |
| AnomalyCLIP: Object-agnostic Prompt Learning... | AnomalyCLIP | No | Yes (Prompt Learning) |
| Adapting Visual-Language Models for Generalizable... | MVFA-AD (Base) | No (It is the base paper) | Yes (Adapters) |
| AF-CLIP: Zero-Shot Anomaly Detection via... | AF-CLIP | No | Yes (Adapters) |
| MadCLIP: Few-shot Medical Anomaly Detection with CLIP | MadCLIP | No | Yes (Adapters) |
| Anomaly-Aware Vision-Language Adapters... | AVA-DINO | No | Yes (Adapters) |
| UltraAD: Fine-Grained Ultrasound Anomaly... | UltraAD | No | Yes (Fine-grained AD) |
| Exploring Zero-Shot Anomaly Detection with CLIP... | Marzullo et al. | No | Yes (Evaluation) |
| Towards Zero-Shot Anomaly Detection and Reasoning... | Anomaly-OV | No | Yes (MLLMs/Reasoning) |
| Toward Generalist Anomaly Detection via In-context... | InCTRL | No | Yes (Generalist AD) |
| GlocalCLIP: Object-agnostic Global-Local Prompt... | GlocalCLIP | No | Yes (Prompt Learning) |
| AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts... | AdaCLIP | No | Yes (Prompt Learning) |
| CoPS: Conditional Prompt Synthesis for Zero-Shot... | CoPS | No | Yes (Prompt Learning) |

## 7. Overlap Analysis

**Statistical Overlap:**
- **Number of papers common to both:** 0
- **Percentage of generated-review papers present in Related Work:** 0%

**Thematic Overlap:**
- **Captured Themes:** Both reviews successfully identified the shift from vanilla/unsupervised anomaly detection to Zero/Few-shot detection leveraging foundation models. Both identified the "Semantic Gap" between natural image pre-training and medical imaging tasks. 
- **Missed by LLM Review:** The generated review completely skips the historical context of non-VLM anomaly detection (e.g., PatchCore, CflowAD, generative methods) and traditional one-class classification methods in medical AD. It starts immediately at the VLM era because its context window *only* contained our highly-filtered VLM papers.
- **Added by LLM Review (Future Work):** Because our scraped dataset included late-2024 and 2025 papers, the generated review captured massive advancements that the base paper (written in early 2024) could not have known about. Specifically:
  1. **The split between Prompt Learning and Visual Adapters:** The LLM review categorizes the field into these two distinct camps (AdaCLIP, CoPS vs AF-CLIP, MadCLIP).
  2. **Multi-category & Fine-Grained Detection:** The LLM review captured the very recent shift toward distinguishing *types* of anomalies (SD-MAD, UltraAD) rather than just binary normal/abnormal.
  3. **MLLM Reasoning:** The LLM review captured the brand new frontier of using large multimodal models (like GPT-4o style architectures) to *reason* about anomalies in text, rather than just outputting a heatmap.

## 8. Conclusion
The automated LLM pipeline generated a highly technical, forward-looking literature review that successfully synthesized the state-of-the-art. While it lacked the historical grounding found in the human-written base paper (due to the strict keyword filtering of our input dataset), it provided a far more detailed and structured taxonomy of *current* VLM adaptation strategies. This demonstrates that LLMs, when provided with high-quality, retrieved abstracts, can autonomously construct thematic literature reviews that match or exceed human synthesis capabilities for cutting-edge subfields.
