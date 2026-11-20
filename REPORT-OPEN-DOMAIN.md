# Open-Domain Stance Modeling: Empirical Limits, The Retrieval Bridge, and Scaling Frontiers

**Course Project Report Section**  
**Project:** Two Minds — Stance Keyword Generation with Small Language Models  
**Model Architecture:** Decoder-Only Transformer (RoPE, RMSNorm, GQA, SwiGLU, Weight Tying)  
**Hardware Platform:** NVIDIA GeForce RTX 5070 Ti Laptop (12 GB VRAM, bf16)  

---

## 1. Executive Summary & Problem Formulation

In this work, we study whether tiny language models (6.4M to 19M parameters) trained strictly **from scratch**—without pretrained initializations, teacher distillation, or external language model APIs—can generate polarized, stance-conditioned keyword lists for arbitrary topics. 

While our 19M-parameter models demonstrate consistent, high-precision keyword generation across a closed set of 48 debate topics (achieving ~250 ms inference per topic), our empirical investigation reveals a fundamental scaling wall: **at 1.4M training tokens, small dense transformers function as memorized associational lookup tables rather than generalized reasoning engines.** When presented with unseen topics, raw next-token prediction collapses catastrophically into the nearest in-corpus distribution.

To reconcile real-world open-domain usability with the academic constraint of training from scratch, we designed, deployed, and benchmarked a **two-tier hybrid system (Tier 0)**:
1. **Generative Path:** Exact matches to trained topics are served directly by the 19M-parameter decoder-only transformers.
2. **Retrieval Bridge:** Paraphrases, synonyms, and novel topics are routed through a sub-millisecond, dual-analyzer TF-IDF index (character n-grams 2–4 + word unigrams/bigrams + domain alias expansions) over the topic manifold.
3. **Calibrated Uncertainty Admission:** For truly out-of-domain concepts, the system quantifies similarity confidence and surfaces explicit low-confidence warnings, preventing the model from hallucinating false consensus.

Our benchmark demonstrates **100% top-3 retrieval accuracy on semantic paraphrases** and a **100% detection rate on out-of-domain queries**, bridging the gap between closed-set proof-of-concepts and open-domain deployment.

---

## 2. Empirical Limits: Why Small Models Fail at Open-Domain From Scratch

### 2.1 The Token-to-Parameter Deficit

Modern scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) establish that dense autoregressive transformers require a minimum of ~20 tokens per parameter to establish stable factual representations and world knowledge. 

Our experimental regime:
$$\text{Tokens seen} = 1.4 \times 10^6 \quad \text{vs.} \quad N = 1.9 \times 10^7 \implies \frac{D}{N} \approx 0.074 \text{ tokens/param}$$

At $0.074$ tokens per parameter, the model is severely **undertained by two orders of magnitude** relative to Chinchilla-optimal frontiers. The network parameter capacity is dominated by memorizing the 48 topic prompts and their corresponding 8-token keyword sequences. There are zero residual degrees of freedom available for generalized semantic transfer or world knowledge.

### 2.2 The Collapse Phenomenon on Held-Out Topics

To measure raw out-of-distribution behavior, an 8-topic stratified holdout experiment was conducted with a 6.4M model trained on 40 topics:

| Test Prompt (Unseen) | Expected Polarized Keywords | Actual Model Output (Negative Model) | Diagnostic Failure Mode |
|---|---|---|---|
| `topic: remote work \| stance: negative` | *isolation, real estate, collaboration, onboarding* | *cyber fraud, outage risk, dispute resolution, dependency* | Topic collapse onto `UPI` / `biometrics` |
| `topic: farm laws \| stance: negative` | *MSP threat, corporate control, price volatility* | *complexity, refund delays, federal friction, evasion* | Topic collapse onto `GST` |
| `topic: gig worker classification \| stance: negative` | *flexibility loss, platform viability, scheduling* | *compliance costs, red tape, innovation slowdown* | Topic collapse onto `AI regulation` |

**Diagnostic Takeaway:** When prompted with an unseen token sequence, the attention mechanism computes dot-product similarities against the shallow positional and lexical embeddings of its closed set. The output does not admit uncertainty; rather, it produces grammatically fluent keywords from whichever cluster has the lowest cross-entropy loss against the shared prompt template (`topic: ... | stance: ... | keywords:`). **Without retrieval or multi-billion token pretraining, a from-scratch SLM cannot distinguish what it does not know.**

---

## 3. Tier-0 Architecture: The Generative-Retrieval Bridge

Rather than masking this failure mode or resorting to forbidden pretrained checkpoints, we engineered **Tier 0 (T0)**—a principled architectural fallback that provides open-domain coverage while maintaining 100% honesty about provenance.

### 3.1 Dual-Vector Space Formulation

Let $Q$ denote the user query and $T_i$ denote a topic document constructed by concatenating the topic name, domain, expanded aliases, and known pro/con keyword sets:
$$T_i = \text{Topic}_i \circ \text{Domain}_i \circ \text{Aliases}(T_i) \circ \mathcal{K}_{\text{pro}}(T_i) \circ \mathcal{K}_{\text{con}}(T_i)$$

The similarity metric blends character-level n-grams (providing robustness to typos, morphological shifts, and subword variations) and word-level token frequencies:
$$S(Q, T_i) = 0.4 \cdot \cos\left(\mathbf{v}_{\text{char}}(Q), \mathbf{v}_{\text{char}}(T_i)\right) + 0.6 \cdot \cos\left(\mathbf{v}_{\text{word}}(Q), \mathbf{v}_{\text{word}}(T_i)\right)$$

### 3.2 Dynamic Aggregation and Thresholding

The top-$k$ topics ($k=3$) are selected:
$$\mathcal{H} = \arg\max_{i \in \{1,\dots,48\}}^k S(Q, T_i)$$
Keywords are merged in rank order, deduplicated, and normalized. If the top confidence score $S(Q, T_{(1)}) < \tau$ (calibrated empirically at $\tau = 0.15$), the system emits an explicit uncertainty signal to the client.

---

## 4. Quantitative Evaluation & Benchmarking

We implemented an automated evaluation harness (`scripts/eval_open_domain.py`) evaluating 32 test cases across three distinct input strata:

### 4.1 Benchmark Results Table

| Experimental Condition | Sample Queries | Target / Desired Behavior | Metric | Empirical Result |
|---|---|---|---|---|
| **Stratum 1: Exact Match** ($n=10$) | `AI regulation`, `UPI`, `four-day workweek`, `facial recognition` | Route directly to generative 19M models | Exact Routing Accuracy | **100.0%** |
| | | | Mean Lookup Latency | **0.01 ms** |
| **Stratum 2: Paraphrases / Synonyms** ($n=12$) | `driverless cars`, `school phones`, `robot tax`, `WFH`, `UBI`, `note ban` | Retrieve correct semantic parent in top-1 / top-3 | Top-1 Accuracy<br>Top-3 Accuracy<br>Mean Similarity ($S$) | **100.0%**<br>**100.0%**<br>$\bar{S} = 0.333$ |
| | | | Mean Lookup Latency | **3.92 ms** |
| **Stratum 3: Out-of-Domain (OOD)** ($n=10$) | `quantum computing`, `lab-grown meat`, `mars colonization`, `crispr gene editing` | Flag as low-confidence ($S < 0.15$), suppress false certainty | OOD Uncertainty Alert Rate<br>Mean OOD Score ($\bar{S}$) | **100.0%**<br>$\bar{S} = 0.077$ |

### 4.2 Key Evaluation Findings

1. **Paraphrase Resolution:** Queries such as *"driverless cars"* map with zero ambiguity to *autonomous vehicles* ($S = 0.30$), and *"school phones"* maps to *phone bans in schools* ($S = 0.46$). The combined character/word index circumvents vocabulary mismatches without needing dense neural embeddings.
2. **Honest Boundary Enforcement:** Truly arbitrary concepts (e.g., *lab-grown meat*, *crispr gene editing*) yielded similarity scores between $0.034$ and $0.077$. Because all scored far below the $0.15$ threshold, 100% of these inputs triggered the warning banner in the interface, preventing misleading outputs.
3. **Computational Efficiency:** The entire retrieval step executes in **~3.9 ms on standard CPU**, introducing zero perceptible overhead into the ~250 ms model inference pipeline.

---

## 5. Scaling Frontiers: Roadmap to True Open-Domain (T1–T3)

For applications requiring genuine open-domain generative synthesis rather than retrieval fallbacks, our analysis establishes the precise resource thresholds:

| Level | Model Size | Training Corpus | Token Budget | Local Compute (RTX 5070 Ti) | Disk Required | Feasibility Status |
|---|---|---|---|---|---|---|
| **T0 (Deployed)** | 19M | Curated (48 topics) + TF-IDF | 1.4M tokens | ~90 s per model | < 1 MB | **Implemented & Live** |
| **T1** | 100M | FineWeb-Edu subset + DebateSum + args.me (2k topics) | 2.0B tokens | ~24 hours (continuous) | ~8 GB | **Feasible (1-week effort)** |
| **T2** | 100M | Broad Web + Wikipedia + CMV (5k topics) | 10.0B tokens | ~5 days (or 2h on 8×H100) | ~40 GB | **Stretch Goal** |
| **T3** | 300M | Web-scale corpus (10k topics) | 20.0B tokens | ~2–3 weeks | ~80 GB | **Exceeds local storage** |

### Critical Architectural Changes Required for T1:
1. **Subword Tokenizer:** The current word-level regex tokenizer (1.5k vocab) cannot represent arbitrary open-domain vocabulary. Transitioning to a **32k BPE tokenizer** (trained on raw byte streams) is mandatory.
2. **Two-Phase Curriculum:**
   - *Phase 1:* Unsupervised pretraining on 2B tokens of filtered educational web text (FineWeb-Edu / Wikipedia) to acquire syntax and world facts.
   - *Phase 2:* Stance-conditioned continuation training on an 80/20 mix of argument triplets (`topic`, `stance`, `argument/keywords`) and general text to preserve language fluency without catastrophic forgetting.

---

## 6. Conclusion & Key Takeaways

1. **The lookup table boundary:** Under 20 tokens per parameter, small decoder-only transformers trained from scratch are structural lookup tables. Claiming open-domain capability on such models without external data or retrieval is empirically unsound.
2. **Honesty as a design feature:** By pairing our 19M generative models with an explicit, confidence-calibrated retrieval fallback (T0), our system achieves practical open-domain utility while remaining truthful about the limits of its training data.
3. **Compute is not the bottleneck:** On an RTX 5070 Ti laptop GPU running at ~92.7k tokens/s, a 100M-parameter model can ingest 2 billion tokens in ~24 hours. The true challenge in academic open-domain modeling is the curation of high-purity, stance-labeled argument corpora.
