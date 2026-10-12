# Intelligent Procurement Document Analysis for Spare Parts Management

**MAAI 902: Natural Language Processing and Large Language Models**
**Assessment 2: NLP Solution Design** | University of Wollongong in Dubai
**Student:** Hala Sarhan (ID 5080307) | **Lecturer:** Dr. Pallavi Ranjan

An end-to-end NLP pipeline that turns raw procurement documents (purchase orders, supplier emails, maintenance reports, general correspondence) into structured, searchable and answerable information. The prototype runs in Google Colab on a **Tesla T4 GPU**.


---

## What the pipeline does

| Stage | Method | Output |
|---|---|---|
| 1. Data | Seeded template generator, 550 documents, 137 with OCR-style noise, gold labels | Reproducible corpus |
| 2. Preprocessing | Cleaning, abbreviation expansion, stopword removal | Normalised text |
| 3. Classification | TF-IDF + 5 classifiers (Logistic Regression, Linear SVM, Naive Bayes, Random Forest, Gradient Boosting) | Document type |
| 4. Entity extraction | Regular expressions + gazetteers | Part, supplier, quantity, date, site, equipment |
| 5. Sentiment and urgency | Keyword rules | Supplier tone, purchase order urgency |
| 6. Retrieval | TF-IDF vs dense (MiniLM + FAISS, GPU) vs hybrid (Reciprocal Rank Fusion) | Most relevant documents |
| 7. Grounded generation | Qwen2.5-3B-Instruct (GPU, fp16) with citations, refusal gate and grounding checks | Cited answers, or a refusal |
| 8. Demo | Gradio web app with three tabs | Interactive prototype |

## Results (from the notebook run)

| Module | Result |
|---|---|
| Classification (best model: Linear SVM) | 100% 5-fold CV, **91.7% on unseen templates**, 100% on noisy documents |
| Entity extraction | F1 1.000 on clean text, **0.916 on noisy text** (precision 0.989, recall 0.853) |
| Urgency (purchase orders) | 100% clean and noisy (n = 39) |
| Sentiment (keyword rules) | 86.0% over 150 supplier emails |
| Retrieval (30 queries) | TF-IDF MRR 0.662, Dense 0.597, **Hybrid 0.681** |
| Grounded generation | 0% in-scope questions wrongly refused, 100% entity grounding, 86.7% citation precision, **15 of 15 off-topic or unanswerable questions refused** |

Why two classification numbers? A random split scores 100% because the same templates appear in training and test data. Testing on templates the model has never seen is the honest figure.

## Repository contents

```
.
├── MAAI_902_Prototype.ipynb     # the full prototype (run top to bottom)
├── requirements.txt             # Python dependencies
├── README.md                    # this file
├── report/
│   └── MAAI902_Assessment2_NLP_Solution_Design_Hala_Sarhan.pdf
└── screenshots/                 # GPU check, results summary, Gradio demo
```

There are no data files. The corpus is generated inside the notebook (Section 2).

## How to run (Google Colab, recommended)

1. Open `MAAI_902_Prototype.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Restart session and run all.** Section 1 installs the extra packages and prints `GPU available: True Tesla T4`.
4. The first run downloads two models from Hugging Face: `sentence-transformers/all-MiniLM-L6-v2` (small) and `Qwen/Qwen2.5-3B-Instruct` (about 6 GB). This takes a few minutes. The language model uses about 6.3 GB of GPU memory and answers in 4 to 6 seconds.
5. The last code cell of Section 7 (end-to-end pipeline and demo) launches the Gradio demo and prints a public link (valid for about a week).

### Running locally

```bash
pip install -r requirements.txt
jupyter notebook MAAI_902_Prototype.ipynb
```

A CUDA GPU with at least 8 GB of memory is needed for the language model. The embedding and classification steps also run on CPU.

## Notebook guide

| Section | Content |
|---|---|
| 1 | Environment, seeds, GPU check |
| 2 | Synthetic dataset generation (550 documents, noise, gold labels, template IDs) |
| 3 | Preprocessing |
| 4 | Module 1, classification: 3 test protocols, 5 models |
| 5 | Module 2, named entity recognition, scored against gold labels |
| 6 | Module 3, sentiment and urgency |
| 7 | Module 4, RAG: retrieval (Part A), retrieval evaluation (Part B), grounded LLM (Part C), safety tests (Part D), then the end-to-end `process_document()` function and the Gradio demo |
| 8 | Performance summary |
| 9 | Conclusion, limitations and production roadmap |

Everything is seeded (`SEED = 42`), so results reproduce. Small differences in the LLM text may occur between GPU types.

## Using the demo

| Tab | What you do | What you get |
|---|---|---|
| **Analyse an email / document** | Enter subject, sender and body (or paste a document), or click an example | Document type, entities, urgency, sentiment (for supplier emails), similar documents |
| **Batch upload** | Upload several `.txt`, `.eml` or `.csv` files (CSV: one document per row) | One results table for all documents |
| **Ask the documents** | Type a question | A cited answer, or "Information not found in the available documents." |

Try the off-topic example question ("Who won the football world cup?") to see the system refuse instead of inventing an answer.

## Ethical design

- **Hallucination control:** a refusal gate (minimum dense similarity 0.53, chosen from the data), mandatory citations, an entity grounding check and a fixed fallback message.
- **Privacy:** synthetic data only. No personal or confidential data is processed.
- **Human in the loop:** the system supports decisions. Raising orders, approving payments and contacting suppliers stay with a person.
- **Fairness and honesty:** unseen-template and noisy-text tests are reported next to the flattering cross-validation score.

## Limitations

- Synthetic, template-based data: real-world accuracy will be lower.
- Small test sets (30 retrieval queries, 150 emails, 15 in-scope questions).
- The refusal threshold was calibrated on the same questions it was evaluated on.
- Rule-based entity extraction only knows the listed suppliers, parts and sites, and recall drops on noisy text (0.853).
- The classifier always chooses one of four document types (there is no "unknown" class).
- English only. Noise is simulated, so real scans would also need OCR.
- The grounding check verifies entities but not the relationships between them.

## Report

The full technical report is in `report/`. It covers the problem definition, literature review, data, preprocessing and embeddings, pipeline design, evaluation, tool stack, ethics, expected value and limitations.

## Author

Hala Sarhan, MAAI 902, Autumn 2026.
