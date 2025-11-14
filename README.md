# HealthQueue Medical QA Chatbot

## Why BERT? Why This Project?

Modern medical chatbots often give generic, vague, or hallucinated answers. We wanted a system that:
- **Extracts real, evidence-based answers** from trusted medical corpora (not just makes up text)
- **Cites its sources** (so users know where info comes from)
- **Handles typos and medical terms** robustly
- **Is fast, open, and privacy-respecting** (no cloud LLM needed)

**BERT** (fine-tuned for extractive QA) is ideal: it can pull precise answer spans from real medical documents, not just generate plausible-sounding text.

## Project Objective

Build a robust, trustworthy medical QA chatbot that:
- Answers user questions by extracting spans from real medical corpora (MedQuAD, MTSamples, guidelines)
- Handles typos, synonyms, and medical abbreviations
- Cites the source for every answer
- Is easy to extend with new data or safety features

## Roadmap: Building the Full Chatbot

### Phase 1: Train Extractive QA Model
- Fine-tune BERT on MedQuAD (Q&A pairs in SQuAD format)
- Validate with EM/F1 on validation split (target: >75%)
- Result: BERT QA model that can extract answer spans from medical text

### Phase 2: Index Medical Corpora for Retrieval
- Index MTSamples (clinical notes), MedQuAD, and any guidelines/curated pages
- Use BM25 (TF-IDF) and semantic vector search (Sentence Transformers)
- Result: Retriever that can find relevant passages for any user query

### Phase 3: Wire the QA Pipeline
- User question → spell-correct/normalize → hybrid retrieve top-k passages
- Run BERT QA on each passage → pick best answer span by score
- Return answer + source citation
- Result: End-to-end QA system (user → answer with source)

### Phase 4: Optional Niceties & Robustness
- Conversational wrapping (make answers sound natural)
- Add safety filters (block unsafe queries)
- Show passage URL/title as citation
- Optionally pretrain BERT on MTSamples (MLM) for more domain knowledge
- Add spell-correction (SymSpell), fuzzy retrieval (RapidFuzz), and synonym expansion
- Hybrid retrieval: BM25 + vectors (so even misspelled queries work)

## Pipeline Overview

1. **User:** "What causes diabetes?"
2. **Retrieval:** Searches BOTH MedQuAD Q&A and MTSamples clinical notes
3. **Top-k passages:** (from either source)
4. **BERT QA:** Extracts answer spans from all passages
5. **Best answer:** Returned with its source (regardless of which corpus)

## Why Not Use MedAlpaca, LLMs, or Generative Models?
- Generative models (like MedAlpaca, GPT, Gemini) can hallucinate or give generic, non-specific answers
- They don't cite sources, so you can't verify the info
- Our BERT QA system is **faster, more accurate, and always grounded in real medical text**
- No need for extra modules or cloud APIs

## Progress So Far
- ✅ Phase 1: BERT QA trained on MedQuAD (EM: 76%, F1: 77%)
- ✅ Phase 2: 46k docs indexed (MedQuAD + MTSamples)
- ✅ Phase 3: Pipeline wired (retrieval → QA → answer)
- ✅ Conversational wrapping added (answers sound more natural)
- ✅ Memory cleanup (removed old checkpoints, only final model kept)
- ⚠️ Phase 4: Safety filters, spell-correction, and fuzzy retrieval are next

## Why Are Replies Sometimes Unnatural?
- Extractive QA gives you the **real answer from the text**, but sometimes the best span is still a bit robotic or generic
- If the source text is vague, the answer will be too
- We wrap answers in conversational templates, but for truly natural dialog, a generative LLM is needed (with risk of hallucination)
- **To keep answers specific and useful:**
  - Improve retrieval (hybrid search, better ranking)
  - Clean up source passages (remove generic intros)
  - Add fallback to show top passages if confidence is low

## Next Steps
- Add spell-correction and fuzzy retrieval for typo-robustness
- Add safety filters and citation display
- Optionally pretrain BERT on MTSamples for more medical knowledge
- Expand corpus with more guidelines and curated pages

## Example Flow

```
User: "What causes diabetes?"
   ↓
1. Retrieval searches BOTH:
   - MedQuAD Q&A (structured answers)
   - MTSamples (clinical notes mentioning diabetes)
   ↓
2. Top-5 passages retrieved (could be from either source)
   ↓
3. BERT QA (trained on MedQuAD) extracts answer spans from ALL passages
   ↓
4. Best answer returned (regardless of source)
```

## Why This Approach?
- **Grounded answers:** Always from real medical text
- **No hallucinations:** No made-up info
- **Citations:** User can verify every answer
- **Extendable:** Add more corpora, filters, or features as needed

---

**This repo is a work in progress. See `backend/PHASE_STATUS.md` for detailed status and next steps.**