# Customer Support RAG Engine & Automated LLM Evaluation Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Inference](https://img.shields.io/badge/Inference-Groq_Cloud-purple.svg)](https://groq.com/)

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for customer service standard operating procedure (SOP) retrieval, coupled with an automated **LLM-as-a-Judge Evaluation Pipeline** to systematically assess hallucination rates, semantic faithfulness, and instruction compliance.

---

## 📌 Architectural Overview

1. **Knowledge Base Ingestion:** Standard Operating Procedures covering digital payments, refund SLAs, KYC Tier-2 verification, and transaction auto-reversals.
2. **Text Chunking & Dense Embeddings:** 
   - Chunk Size: `350 characters` | Overlap: `50 characters` via LangChain `RecursiveCharacterTextSplitter`.
   - Local Embedding Engine: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
3. **Vector Database:** Local **ChromaDB** indexed with similarity search retriever ($k=2$).
4. **Generator Model:** `qwen/qwen3.6-27b` configured at temperature `0.0` with strict zero-shot anti-hallucination prompt constraints.
5. **Independent Auditor (LLM-as-a-Judge):** `openai/gpt-oss-120b` performing automated scoring (0/1) across three core QA metrics.

---

## 📊 Benchmark Evaluation Matrix

The pipeline was benchmarked against curated ground-truth test cases representing both positive factual retrieval and negative/out-of-domain edge cases:

| Test ID | Category | Query / Intent | Ground Truth Reference | Faithfulness | Relevance | Compliance | Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **TC-01** | Direct Fact Retrieval | Refund timeframe for QR Code / VA | Within 1 business day (24 hours) to wallet | **1.0** | **1.0** | **1.0** | **PASS** |
| **TC-02** | Direct Fact Retrieval | Max failed PIN attempts before suspension | 5 consecutive failed attempts | **1.0** | **1.0** | **1.0** | **PASS** |
| **TC-03** | Out-of-Domain / Negative | Refund request for international flights | Explicit out-of-domain rejection notice | **1.0** | **1.0** | **1.0** | **PASS** |
| **TC-04** | Policy Verification | Non-refundable buyer-initiated fees | Buyer processing fees are non-refundable | **1.0** | **1.0** | **1.0** | **PASS** |

### Evaluation Metrics Defined:
* **Faithfulness (0/1):** Verifies that claims are derived strictly from retrieved chunks without ungrounded external assumptions.
* **Answer Relevance (0/1):** Evaluates if the response addresses the prompt concisely without generating unrelated artifacts.
* **Instruction Compliance (0/1):** Confirms adherence to boundary rules (e.g., executing standard fallback phrases when context is missing).

---

## 📂 Repository Structure

```text
├── knowledge_base.txt          # Raw customer support SOP documentation
├── rag_evaluation_results.csv  # Structured benchmark execution logs & judge scoring
├── rag_llm_evaluation.ipynb    # End-to-end modular Jupyter Notebook pipeline
└── README.md                   # System documentation & benchmark analysis
