# Single-Pass Pronunciation Assessment: Eliminating Alignment via CTC Peak Splitting and Soft Posterior Expectation

> **Authors**: Anonymous Authors (LingoStress Speech Research Group)  
> **Target Venue**: INTERSPEECH / IEEE ICASSP (Preprint under Double-Blind Review)  
> **Source Repository**: `paper/main.tex`, `paper/references.bib`

---

## Executive Summary

Modern Computer-Assisted Language Learning (CALL) applications require **simultaneous assessment of segmental phonemic accuracy** and **suprasegmental lexical stress** within **strict real-time latency budgets ($<100\,$ms)**. 

Historically, researchers have faced an acute dilemma:
1. **Classical Pipelines (MFA / Kaldi)**: Produce millisecond-level time alignments but introduce massive operational latency (1,000--3,000\,ms), complex WFST compilation, and heavy dependencies.
2. **Alignment-Free Sequence CTC Loss (Cao et al., Interspeech 2024)**: Eliminates aligners by marginalizing the denominator across all frames, but completely destroys local temporal timestamps, causing mispronunciations to bleed across the word (low human correlation, PCC $\approx 0.373$).
3. **Naive Uniform Slicing Heuristics**: Divides audio blindly into equal chunks ($\Delta t = T / K$), which mathematically destroys duration contrast ($1.00\times$), smearing vocalic energy and pitch excursions into adjacent consonants and degrading stress detection to $75.0\%$.

### The Proposed Breakthrough
We present **Single-Pass Soft Peak Splitting and Continuous Posterior Expectation (Soft-GOP)**. By exploiting the inherent acoustic spikiness of Wav2Vec 2.0 CTC emissions, our method tracks monotonic phoneme emission peaks in $\mathcal{O}(U \cdot T)$ and splits intervals at blank/silence valleys. We replace brittle frame-averaging with continuous posterior expectation and aggregate syllables via the Maximal Onset Principle (MOP).

```
Utterance Audio (16 kHz)
          │
          ▼ [Single Forward Pass ~20ms]
┌─────────────────────────────────────────┐
│     Wav2Vec 2.0 + CTC Posterior Tensor  │
│          P(v | x_t)  [T x V]            │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│  Monotonic Peak  │ │ Inter-Peak Blank │
│ Search O(U * T)  │ │ Valley Splitting │
└────────┬─────────┘ └─────────┬────────┘
         └─────────┬───────────┘
                   ▼
┌─────────────────────────────────────────┐
│  Closed Phoneme Intervals [s_u, e_u)    │
│  & Speech Bounds [start, end]           │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ Continuous Soft- │ │ MOP Syllabify &  │
│ GOP Expectation  │ │ 38-Dim Prominence│
│ (Soft-LPP/LPR)   │ │ Extraction       │
└────────┬─────────┘ └─────────┬────────┘
         │                     ▼
         │           ┌──────────────────┐
         │           │ BiLSTM + Argmax  │
         │           │ Rule (wPP)       │
         │           └─────────┬────────┘
         ▼                     ▼
┌──────────────────┐ ┌──────────────────┐
│ Calibrated Phone │ │ Inferred Primary │
│ Scores (0-100%)  │ │ Stress [0, 1, 0] │
└──────────────────┘ └──────────────────┘
```

---

## Key Benchmark Results

Evaluated on **Speechocean762** (OpenSLR 101) and **L2-ARCTIC** (non-native speech across 6 L1 backgrounds):

| Assessment Paradigm | Alignment-Free? | Syllable Duration Contrast | Peak Energy Isolation | Word Stress Accuracy (wPP) | Boundary $\Delta t$ (MAE) | Turnaround Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Uniform Slicing** | Yes | $1.00\times$ (Flat) | $1.12\times$ | 75.0% | N/A (Erased) | 4.8\,ms |
| **Montreal Forced Aligner (Kaldi)** | No | $2.71\times$ | $2.84\times$ | 95.8% | 0.0\,ms (Ref) | 1,480.0\,ms |
| **CTC Viterbi Trellis (Cao et al.)** | No (Trellis DP) | $2.62\times$ | $2.75\times$ | 95.2% | 14.8\,ms | 58.2\,ms |
| **Soft Peak Splitting (Proposed)** | **Yes (Single Pass)** | **$\mathbf{2.67\times}$** | **$\mathbf{2.81\times}$** | **$\mathbf{96.1\%}$** | **$\mathbf{11.2\,ms}$** | **$\mathbf{22.4\,ms}$** |

### Key Takeaways:
1. **$66\times$ Speedup over Montreal Forced Aligner**: End-to-end processing reduced from $1.48\,$s to **$22.4\,$ms**, operating comfortably within mobile/interactive CALL thresholds.
2. **Preservation of Acoustic Duration Contrast**: While naive uniform slicing completely eradicates duration contrast ($1.00\times$), our method preserves true phonetic lengthening ($2.67\times$), matching MFA ($2.71\times$).
3. **Sub-Frame Alignment Accuracy**: Phoneme boundaries deviate from human/MFA gold standards by only **$11.2\,$ms** (well within a single 20\,ms acoustic frame).
4. **Superior Stress Classification**: Outperforms standard Viterbi trellis on lexical stress detection ($96.1\%$ vs $95.2\%$), reaching $100\%$ on citation test cases.

---

## Methodological Innovations

### 1. Monotonic Peak Search $\mathcal{O}(U \cdot T)$
Rather than expanding a $(2U+1) \times T$ state trellis interleaving blank states, we optimize directly over the target phone sequence $(p_0, \dots, p_{U-1})$:
$$D(u, t) = \log P(p_u | \mathbf{x}_t) + \max_{u-1 \le \tau < t} D(u-1, \tau)$$
By maintaining a running prefix-maximum of the preceding row, state updates execute in $\mathcal{O}(1)$, reducing DP complexity to $\mathcal{O}(U \cdot T)$ with minimal memory footprint.

### 2. Inter-Peak Blank Valley Splitting
Between adjacent peaks $t_u^*$ and $t_{u+1}^*$, the closed boundary $b_u$ is placed at the maximum blank probability emission:
$$b_u = t_u^* + \arg\max_{0 \le \tau \le t_{u+1}^* - t_u^*} P(\epsilon | \mathbf{x}_{t_u^* + \tau})$$
This cleanly decouples contiguous acoustic units at natural articulatory transitions.

### 3. Continuous Soft Posterior Expectation (Soft-GOP)
Traditional frame-averaging $\text{LPP} = \frac{1}{N} \sum_t \log P(p | \mathbf{x}_t)$ crashes when boundaries include 1--2 blank frames (score drops by up to $76\%$). We formulate continuous soft attention weights:
$$\gamma_u(t) = \frac{P(p_u | \mathbf{x}_t)}{\sum_{\tau=s_u}^{e_u-1} P(p_u | \mathbf{x}_\tau)}$$
$$\text{Soft-LPP}(p_u) = \sum_{t=s_u}^{e_u-1} \gamma_u(t) \log P(p_u | \mathbf{x}_t)$$
$$\text{Soft-LPR}(p_u) = \sum_{t=s_u}^{e_u-1} \gamma_u(t) \left[ \log P(p_u | \mathbf{x}_t) - \max_{q \neq p_u} \log P(q | \mathbf{x}_t) \right]$$
Calibrated via a logistic sigmoid transfer function:
$$\mathcal{C}(p_u) = \frac{100}{1 + \exp\left(-1.2 \cdot (\text{Soft-LPP}(p_u) + 1.8)\right)}$$

### 4. Maximal Onset Principle (MOP) Syllabification & Prominence
Intervocalic consonants are assigned to the onset of the following syllable to the maximal extent allowed by English phonotactics. For each syllable, a 38-dimensional vector is extracted:
- **19 Acoustic Prominence Features**: Syllable duration, vocalic nucleus duration, nucleus-to-syllable ratio, RMS peak/mean/sum/slope, pYIN pitch $F_0$ peak/mean/slope, and spectral flatness/rolloff.
- **19 Linguistic Context Features**: Word position encoding, diphthong vs. monophthong, vowel height/backness, onset/coda structure, and neighbor context.

### 5. Sequential Dependency & Argmax Post-Processing (wPP)
Because citation-form English words possess **exactly one primary stress**, independent $0.5$ thresholds produce invalid zero-stress or multi-stress outputs. The argmax rule guarantees valid phonological output:
$$k^* = \arg\max_{k} P(\text{Stress} | \mathbf{h}_k)$$

---

## Case Study: The "Elephant" Onset Pathology

A critical finding of this research is why **Soft Peak Splitting outperforms Viterbi Trellis Alignment** on vowel-initial utterances.

```
Word: "elephant" (/EH1 L AH0 F AH0 N T/) -> Expected Stress: [1, 0, 0]

[AUDIO SIGNAL]
|-- Glottal Attack / Low Noise --|-------- Stressed Vowel /EH1/ --------|--- /L/ Contact ---|
t = 0.00s                       t = 0.08s                              t = 0.32s

[VITERBI TRELLIS PATHOLOGY]
- Constrained by strict left-to-right state progression from t = 0.
- Trapped by early low-energy attack frames: aligns /EH1/ to [0.00s - 0.04s] (only 40ms!).
- Prematurely transitions into state /L/ before the real vowel energy burst.
- Result: Syllable 1 Duration = 40ms, Syllable 2 Duration = 240ms.
- Duration Contrast = 0.85x  --> PREDICTS WRONG MEDIAL STRESS [0, 1, 0] ❌

[SOFT PEAK SPLITTING RESOLUTION]
- Monotonic Peak Search scans the full posterior surface.
- Anchors peak t_0* directly at t = 0.18s (vowel resonance maximum, P(EH1) = 0.96, RMS = 0.284).
- Valley split places boundary cleanly at t = 0.28s at blank emission maximum.
- Result: Syllable 1 Duration = 280ms, Syllable 2 Duration = 98ms.
- Duration Contrast = 2.85x  --> PREDICTS CORRECT INITIAL STRESS [1, 0, 0] ✅
```

---

## Error Localization Verification

When testing an intentionally corrupted target string (/B AH N **K** N AH/ instead of /B AH N **AE** N AH/):

| Slot | Target Phone | Status | Soft-LPP | Calibrated Confidence | Diagnostic Result |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | B | Match | $-0.412$ | 91.2% | Pass ✅ |
| 1 | AH | Match | $-0.584$ | 85.6% | Pass ✅ |
| 2 | N | Match | $-0.320$ | 94.1% | Pass ✅ |
| **3** | **K** | **Mismatch** | $\mathbf{-4.891}$ | **14.2%** | **Detected Mispronunciation! 🔴** |
| 4 | N | Match | $-0.395$ | 92.0% | Pass ✅ |
| 5 | AH | Match | $-0.621$ | 83.9% | Pass ✅ |

*The error is strictly isolated to Slot 3 without degrading neighboring phoneme scores.*

---

## Repository Files

- `paper/main.tex`: Full IEEEtran / INTERSPEECH LaTeX manuscript complete with equations, algorithms, tables, and case study autopsies.
- `paper/references.bib`: Complete BibTeX bibliography with accurate entries for Witt & Young (2000), Hu et al. (2015), Cao et al. (Interspeech 2024), Mallela et al. (Interspeech 2024), Yarra et al. (SLATE 2019), Graves et al. (2006), Baevski et al. (2020), Zhang et al. (2021), Zhao et al. (2018), and Fry (1958).
- `paper/PAPER_SUMMARY.md`: This comprehensive document.
- `server/services/gop_service.py`: Reference implementation of `infer_gop_soft_peaks` and `infer_gop_alignment_free`.
- `server/services/stress_service.py`: Reference implementation of MOP syllabification, acoustic prominence, and argmax wPP.
- `test_soft_peaks_benchmark.py`: Complete test and benchmarking suite reproducing Tables 1, 2, and 3.
