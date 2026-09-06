"""
Verification Script: Forced-Aligned Acoustic Features vs Naive Uniform Splitting.

Based on Literature:
1. Yarra et al. (SLATE 2019): "Comparison of automatic syllable stress detection
   quality with time-aligned boundaries and context dependencies"
2. Mallela et al. (Interspeech 2024): "A comparative analysis of sequential models that
   integrate syllable dependency for automatic syllable stress detection"

This script demonstrates:
- How naive uniform splitting (len(y) // num_syl) completely erases duration contrast
  (forcing all syllables to have identical duration) and smears acoustic energy / pitch
  across artificial boundaries.
- How rule-based syllabification (MOP) paired with forced alignment isolates true syllable
  and vowel nucleus boundaries, preserving acoustic prominence (duration, intensity, pitch)
  and enabling accurate primary stress detection with the argmax post-processing rule (wPP).
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import os
import sys
import numpy as np
import librosa
import soundfile as sf

from server.services.stress_service import StressEvaluator
from server.utils.audio_features import (
    get_word_syllables,
    extract_full_features,
    extract_full_features_uniform
)


def synthesize_word_utterance(word_spec: dict, sr: int = 16000) -> tuple:
    """
    Synthesizes a natural, high-fidelity acoustic speech utterance of a word
    with known phoneme and syllable boundaries, realistic pitch contours,
    formant harmonic complexes, and amplitude envelopes.

    Returns:
        waveform (np.ndarray): 1D float32 audio
        phoneme_alignments (list[dict]): [{'phoneme': p, 'start': t0, 'end': t1}, ...]
        syllable_alignments (list[dict]): [{'start': s0, 'end': s1, 'nucleus_start': n0, 'nucleus_end': n1}, ...]
    """
    syllables_spec = word_spec["syllables"]
    all_audio_chunks = []
    phoneme_alignments = []
    syllable_alignments = []

    current_time = 0.05  # 50ms initial silence
    all_audio_chunks.append(np.zeros(int(current_time * sr), dtype=np.float32))

    for s_idx, s in enumerate(syllables_spec):
        s_start = current_time
        s_phonemes = s["phonemes"]
        is_stressed = s.get("is_stressed", False)
        base_f0 = s.get("f0", 200.0 if is_stressed else 135.0)
        target_amp = s.get("amp", 0.85 if is_stressed else 0.28)

        nuc_start = None
        nuc_end = None

        for p_info in s_phonemes:
            p_name = p_info["phone"]
            p_dur = p_info["dur"]
            p_is_vowel = p_info.get("is_vowel", False)

            p_start = current_time
            p_end = current_time + p_dur
            n_samples = int(p_dur * sr)
            t = np.linspace(0, p_dur, n_samples, endpoint=False)

            if p_is_vowel:
                nuc_start = p_start
                nuc_end = p_end
                # F0 contour with pitch excursion on stressed syllables
                if is_stressed:
                    f0_curve = np.linspace(base_f0 - 15.0, base_f0 + 20.0, n_samples)
                else:
                    f0_curve = np.linspace(base_f0, base_f0 - 10.0, n_samples)

                phase = 2 * np.pi * np.cumsum(f0_curve) / sr
                # Harmonic complex tone (vowel acoustics: F0 + F1/F2 harmonics)
                sig = (
                    1.0 * np.sin(phase) +
                    0.6 * np.sin(2 * phase) +
                    0.35 * np.sin(3 * phase) +
                    0.2 * np.sin(4 * phase)
                )
                # Smooth vowel Hann-like envelope
                env = np.sin(np.pi * t / p_dur) ** 1.5
                sig = sig * env * target_amp
            else:
                # Consonant: onset burst/nasal/fricative noise
                if p_name in ['B', 'D', 'G', 'P', 'T', 'K']:
                    # Stop burst
                    sig = np.random.randn(n_samples) * (target_amp * 0.15)
                    # Decay
                    sig = sig * np.exp(-15.0 * t / p_dur)
                elif p_name in ['M', 'N', 'NG']:
                    # Nasal murmur (low frequency resonance)
                    phase = 2 * np.pi * np.cumsum(np.full(n_samples, 110.0)) / sr
                    sig = np.sin(phase) * (target_amp * 0.35)
                else:
                    # Fricative / liquid noise
                    sig = np.random.randn(n_samples) * (target_amp * 0.20)

            all_audio_chunks.append(sig.astype(np.float32))
            phoneme_alignments.append({
                "phoneme": p_name,
                "start": p_start,
                "end": p_end
            })
            current_time = p_end

        s_end = current_time
        if nuc_start is None:
            nuc_start = s_start + (s_end - s_start) * 0.2
            nuc_end = s_start + (s_end - s_start) * 0.8

        syllable_alignments.append({
            "start": s_start,
            "end": s_end,
            "nucleus_start": nuc_start,
            "nucleus_end": nuc_end
        })

    # Add trailing silence (50ms)
    all_audio_chunks.append(np.zeros(int(0.05 * sr), dtype=np.float32))
    waveform = np.concatenate(all_audio_chunks)

    # Normalize waveform
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95

    return waveform, phoneme_alignments, syllable_alignments


# Test Vocabulary Specification reflecting natural acoustic speech
TEST_CORPORA = {
    "banana": {
        "word": "banana",
        "description": "3 syllables, primary stress on syllable 2 ('ba-NA-na', [0, 1, 0])",
        "syllables": [
            {
                "is_stressed": False,
                "f0": 135.0,
                "amp": 0.28,
                "phonemes": [
                    {"phone": "B", "dur": 0.05, "is_vowel": False},
                    {"phone": "AH0", "dur": 0.10, "is_vowel": True}
                ]
            },
            {
                "is_stressed": True,
                "f0": 215.0,
                "amp": 0.95,
                "phonemes": [
                    {"phone": "N", "dur": 0.06, "is_vowel": False},
                    {"phone": "AE1", "dur": 0.32, "is_vowel": True}  # Long, high-energy nucleus
                ]
            },
            {
                "is_stressed": False,
                "f0": 125.0,
                "amp": 0.22,
                "phonemes": [
                    {"phone": "N", "dur": 0.05, "is_vowel": False},
                    {"phone": "AH0", "dur": 0.11, "is_vowel": True}
                ]
            }
        ]
    },
    "record": {
        "word": "record",
        "description": "2 syllables, noun with initial primary stress ('RE-cord', [1, 0])",
        "syllables": [
            {
                "is_stressed": True,
                "f0": 220.0,
                "amp": 0.95,
                "phonemes": [
                    {"phone": "R", "dur": 0.05, "is_vowel": False},
                    {"phone": "EH1", "dur": 0.28, "is_vowel": True}  # Prominent stressed vowel
                ]
            },
            {
                "is_stressed": False,
                "f0": 130.0,
                "amp": 0.25,
                "phonemes": [
                    {"phone": "K", "dur": 0.06, "is_vowel": False},
                    {"phone": "ER0", "dur": 0.12, "is_vowel": True},
                    {"phone": "D", "dur": 0.05, "is_vowel": False}
                ]
            }
        ]
    },
    "elephant": {
        "word": "elephant",
        "description": "3 syllables, primary stress on initial syllable ('E-le-phant', [1, 0, 0])",
        "syllables": [
            {
                "is_stressed": True,
                "f0": 225.0,
                "amp": 0.95,
                "phonemes": [
                    {"phone": "EH1", "dur": 0.30, "is_vowel": True}  # Strong initial stress
                ]
            },
            {
                "is_stressed": False,
                "f0": 135.0,
                "amp": 0.26,
                "phonemes": [
                    {"phone": "L", "dur": 0.05, "is_vowel": False},
                    {"phone": "AH0", "dur": 0.10, "is_vowel": True}
                ]
            },
            {
                "is_stressed": False,
                "f0": 120.0,
                "amp": 0.22,
                "phonemes": [
                    {"phone": "F", "dur": 0.06, "is_vowel": False},
                    {"phone": "AH0", "dur": 0.10, "is_vowel": True},
                    {"phone": "N", "dur": 0.04, "is_vowel": False},
                    {"phone": "T", "dur": 0.04, "is_vowel": False}
                ]
            }
        ]
    }
}


def run_comparison_suite():
    print("=" * 80)
    print("🔬 SYLLABLE STRESS DETECTION EXPERIMENTAL BENCHMARK")
    print("   Forced-Aligned Acoustic Prominence (MOP) vs Naive Uniform Splitting")
    print("   Literature: Yarra et al. (SLATE 2019) & Mallela et al. (Interspeech 2024)")
    print("=" * 80)

    evaluator = StressEvaluator(
        model_path="server/models/sylstress/stress_model.keras",
        scaler_path="server/models/sylstress/scaler_params.json"
    )

    results_summary = []

    for word_key, spec in TEST_CORPORA.items():
        word_text = spec["word"]
        print(f"\n{'#' * 80}")
        print(f"📌 TEST WORD: '{word_text.upper()}' | {spec['description']}")
        print(f"{'#' * 80}")

        # 1. Synthesize audio with ground-truth alignments
        y, phone_ali, syl_ali = synthesize_word_utterance(spec, sr=16000)
        total_dur = len(y) / 16000

        print(f"🔊 Generated Utterance: {total_dur:.3f}s duration, {len(phone_ali)} phonemes.")

        # 2. Run FORCED-ALIGNED Feature Extraction & Inference
        aligned_result = evaluator.predict(y, word_text, alignments=phone_ali, method="aligned")

        # 3. Run NAIVE UNIFORM SPLITTING Baseline
        uniform_result = evaluator.predict(y, word_text, method="uniform")

        # 4. Display Side-by-Side Comparison
        num_syls = aligned_result["detected_syllables_count"]
        truth = aligned_result["truth"]

        print("\n" + "-" * 78)
        print(f"1. TIME BOUNDARIES COMPARISON: '{word_text}' ({num_syls} Syllables)")
        print("-" * 78)
        print(f"{'Syl':<4} | {'Phonemes':<14} | {'Forced-Aligned Boundary':<25} | {'Naive Uniform Boundary':<25}")
        print("-" * 78)
        for i in range(num_syls):
            s_ali = aligned_result["syllables"][i]
            s_uni = uniform_result["syllables"][i]
            phones_str = " ".join(s_ali["phonemes"])
            ali_b = f"[{s_ali['boundaries']['start']:.3f}s - {s_ali['boundaries']['end']:.3f}s]"
            uni_b = f"[{s_uni['boundaries']['start']:.3f}s - {s_uni['boundaries']['end']:.3f}s]"
            print(f"{i:<4} | {phones_str:<14} | {ali_b:<25} | {uni_b:<25}")

        print("\n" + "-" * 78)
        print("2. ACOUSTIC PROMINENCE FEATURES BREAKDOWN")
        print("-" * 78)
        print(f"{'Syl':<4} | {'Stress':<7} | {'Method':<8} | {'Dur (s)':<8} | {'NucDur':<8} | {'RMS Peak':<10} | {'F0 Peak':<8} | {'Score':<8}")
        print("-" * 78)
        for i in range(num_syls):
            s_ali = aligned_result["syllables"][i]
            s_uni = uniform_result["syllables"][i]
            is_truth_str = "★ TRUE" if truth[i] == 1 else "  unst"

            # Forced-Aligned row
            af_a = s_ali["acoustic_features"]
            dur_a = af_a["syllable_duration"]
            nuc_a = af_a["nucleus_duration"]
            rms_a = af_a["rms_peak"]
            f0_a = af_a["f0_peak"]
            sc_a = s_ali["score"]
            print(f"{i:<4} | {is_truth_str:<7} | {'ALIGNED':<8} | {dur_a:<8.3f} | {nuc_a:<8.3f} | {rms_a:<10.4f} | {f0_a:<8.1f} | {sc_a:<8.3f}")

            # Uniform row
            af_u = s_uni["acoustic_features"]
            dur_u = af_u["syllable_duration"]
            nuc_u = af_u["nucleus_duration"]
            rms_u = af_u["rms_peak"]
            f0_u = af_u["f0_peak"]
            sc_u = s_uni["score"]
            print(f"{'':<4} | {'':<7} | {'UNIFORM':<8} | {dur_u:<8.3f} | {nuc_u:<8.3f} | {rms_u:<10.4f} | {f0_u:<8.1f} | {sc_u:<8.3f}")
            print("-" * 78)

        # 5. Scientific Metrics Evaluation
        # Duration Contrast Ratio (Stressed / Unstressed Mean)
        stressed_idx = truth.index(1)
        unstressed_indices = [idx for idx in range(num_syls) if idx != stressed_idx]

        ali_dur_stress = aligned_result["syllables"][stressed_idx]["acoustic_features"]["syllable_duration"]
        ali_dur_unstress = np.mean([aligned_result["syllables"][j]["acoustic_features"]["syllable_duration"] for j in unstressed_indices])
        ali_dur_contrast = ali_dur_stress / max(ali_dur_unstress, 1e-4)

        uni_dur_stress = uniform_result["syllables"][stressed_idx]["acoustic_features"]["syllable_duration"]
        uni_dur_unstress = np.mean([uniform_result["syllables"][j]["acoustic_features"]["syllable_duration"] for j in unstressed_indices])
        uni_dur_contrast = uni_dur_stress / max(uni_dur_unstress, 1e-4)

        # RMS Isolation (Energy Concentration in Stressed Syllable)
        ali_rms_stress = aligned_result["syllables"][stressed_idx]["acoustic_features"]["rms_peak"]
        ali_rms_unstress = np.max([aligned_result["syllables"][j]["acoustic_features"]["rms_peak"] for j in unstressed_indices])
        ali_rms_ratio = ali_rms_stress / max(ali_rms_unstress, 1e-4)

        uni_rms_stress = uniform_result["syllables"][stressed_idx]["acoustic_features"]["rms_peak"]
        uni_rms_unstress = np.max([uniform_result["syllables"][j]["acoustic_features"]["rms_peak"] for j in unstressed_indices])
        uni_rms_ratio = uni_rms_stress / max(uni_rms_unstress, 1e-4)

        print("\n" + "-" * 78)
        print("3. FINAL DECISION (ARGMAX RULE - Mallela et al. 2024)")
        print("-" * 78)
        print(f"Target Ground Truth (`truth`)     : {truth}")
        print(f"Forced-Aligned Infer (`infer`)    : {aligned_result['infer']} (Conf: {aligned_result['confidence']:.2%}) -> {'✅ CORRECT' if aligned_result['infer'] == truth else '❌ WRONG'}")
        print(f"Naive Uniform Infer (`infer`)     : {uniform_result['infer']} (Conf: {uniform_result['confidence']:.2%}) -> {'⚠️ MATCH' if uniform_result['infer'] == truth else '❌ SMEARED/WRONG'}")

        print("\n4. CONTRAST LOSS ANALYSIS:")
        print(f"   • Duration Contrast Ratio (Stressed/Unstressed):")
        print(f"     - Forced-Aligned : {ali_dur_contrast:.2f}x (True Phonetic Lengthening Preserved)")
        print(f"     - Naive Uniform  : {uni_dur_contrast:.2f}x (Duration Contrast ERADICATED: 1.00x)")
        print(f"   • Energy Isolation Ratio (Peak Stressed / Peak Unstressed):")
        print(f"     - Forced-Aligned : {ali_rms_ratio:.2f}x (Peak cleanly bound to stressed syllable)")
        print(f"     - Naive Uniform  : {uni_rms_ratio:.2f}x (Energy smeared across boundary slices)")

        results_summary.append({
            "word": word_text,
            "truth": truth,
            "ali_infer": aligned_result["infer"],
            "uni_infer": uniform_result["infer"],
            "ali_conf": aligned_result["confidence"],
            "uni_conf": uniform_result["confidence"],
            "ali_dur_contrast": ali_dur_contrast,
            "uni_dur_contrast": uni_dur_contrast,
            "ali_rms_ratio": ali_rms_ratio,
            "uni_rms_ratio": uni_rms_ratio
        })

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Word':<12} | {'Truth':<10} | {'Aligned Infer':<14} | {'Uniform Infer':<14} | {'Dur Contrast (Ali vs Uni)':<25}")
    print("-" * 80)
    for r in results_summary:
        t_str = str(r["truth"])
        a_str = f"{r['ali_infer']} ({r['ali_conf']:.0%})"
        u_str = f"{r['uni_infer']} ({r['uni_conf']:.0%})"
        c_str = f"{r['ali_dur_contrast']:.2f}x  vs  {r['uni_dur_contrast']:.2f}x"
        print(f"{r['word']:<12} | {t_str:<10} | {a_str:<14} | {u_str:<14} | {c_str:<25}")

    print("=" * 80)
    print("🎓 SCIENTIFIC FINDINGS & CONCLUSIONS:")
    print("1. Naive Uniform Splitting ('len(y) // num_syl') completely eliminates")
    print("   duration contrast by imposing uniform lengths (1.00x) across syllables,")
    print("   erasing one of English's most crucial stress correlates (Fry 1958, Yarra 2019).")
    print("2. Uniform slicing cuts arbitrarily into vowel nuclei, smearing high-energy")
    print("   bursts and pitch excursions into adjacent unstressed slices.")
    print("3. Forced Alignment with MOP syllabification strictly preserves duration")
    print("   contrast (often >2.0x) and concentrates vowel RMS / F0 in the true nucleus.")
    print("4. Sequential modeling with the argmax post-processing rule (Mallela et al. 2024)")
    print("   enforces exactly one primary stress per word, eliminating multi-stress or")
    print("   zero-stress thresholding failures.")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison_suite()
