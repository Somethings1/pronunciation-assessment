import os
import io
import re
import logging
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from g2p_en import G2p
import syllapy

from server.utils.audio_processor import preprocess_audio

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL_NAME = "mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39"


class GOPEvaluator:
    """
    Goodness of Pronunciation (GOP) Evaluator using Forced Alignment
    based on Wav2Vec2 CTC emissions and Dynamic Programming Trellis Alignment
    (Witt & Young 2000, Hu et al. 2015, Cao et al. Interspeech 2024 'GOP-CTC-align').

    Provides:
      - Forced alignment with exact temporal start/end timestamps and frame intervals.
      - Frame-averaged Log Posterior Probability (LPP).
      - Log Posterior Ratio (LPR) against competing phoneme hypotheses.
      - Calibrated percentage confidence score (0-100%) mapped via logistic scaling.
      - Speech boundaries for smart audio cropping.
      - Alignment-free CTC loss (SDI) comparative benchmark method.
    """

    def __init__(self, model_path=None):
        self.g2p = G2p()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_duration = 0.02  # 20ms per frame for Wav2Vec2 (stride 320 at 16kHz)

        # Determine model path / fallback
        chosen_path = model_path if model_path else DEFAULT_MODEL_NAME
        if os.path.exists(chosen_path) and os.path.isdir(chosen_path):
            has_weights = any(
                os.path.exists(os.path.join(chosen_path, f))
                for f in ["model.safetensors", "pytorch_model.bin"]
            )
            if not has_weights:
                print(
                    f"⚠️ Local model path '{chosen_path}' contains config but missing weights "
                    f"(model.safetensors / pytorch_model.bin). Falling back to '{DEFAULT_MODEL_NAME}'."
                )
                chosen_path = DEFAULT_MODEL_NAME

        print(f"⏳ Loading GOP Model from: {chosen_path} on {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(chosen_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(chosen_path).to(self.device)
            self.model.eval()

            self.vocab = self.processor.tokenizer.get_vocab()
            self.blank_id = self.processor.tokenizer.pad_token_id
            if self.blank_id is None:
                self.blank_id = 0

            # Build case-insensitive vocabulary lookup
            self.token_to_id = {}
            for token, idx in self.vocab.items():
                self.token_to_id[token] = idx
                self.token_to_id[token.upper()] = idx
                self.token_to_id[token.lower()] = idx

            # Support standard TIMIT 39-phoneme mergers if missing in vocab
            # (e.g. AO -> AA, AX -> AH, AXR -> ER)
            if "AA" in self.token_to_id and "AO" not in self.token_to_id:
                self.token_to_id["AO"] = self.token_to_id["AA"]
                self.token_to_id["ao"] = self.token_to_id["AA"]
            if "AH" in self.token_to_id and "AX" not in self.token_to_id:
                self.token_to_id["AX"] = self.token_to_id["AH"]
                self.token_to_id["ax"] = self.token_to_id["AH"]
            if "ER" in self.token_to_id and "AXR" not in self.token_to_id:
                self.token_to_id["AXR"] = self.token_to_id["ER"]
                self.token_to_id["axr"] = self.token_to_id["ER"]

            # Reverse map for non-special phonemes
            self.id_to_phoneme = {}
            for token, idx in self.vocab.items():
                if not (token.startswith("<") or token.startswith("[") or token.startswith("s")):
                    self.id_to_phoneme[idx] = token.upper()

            # Non-phoneme special token IDs to exclude from competing phonemes
            self.special_ids = set([self.blank_id])
            for sp in ["<pad>", "<unk>", "<s>", "</s>", "[PAD]", "[UNK]"]:
                if sp in self.vocab:
                    self.special_ids.add(self.vocab[sp])

            print(
                f"✅ Loaded GOP model successfully. Vocab size: {len(self.vocab)}, "
                f"Blank ID: {self.blank_id}."
            )

        except Exception as e:
            print(f"❌ Failed to load GOP model: {e}")
            self.model = None

    def _load_audio(self, audio_bytes):
        """Reads audio bytes and ensures 16kHz mono float32 numpy array."""
        try:
            y, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        return y.astype(np.float32), sr

    def text_to_phonemes(self, transcript_text):
        """
        Converts input text to clean CMU Arpabet phonemes and model token IDs.
        Strips stress numbers and filters out punctuation.
        """
        raw_phonemes = self.g2p(transcript_text)
        phoneme_list = []
        for p in raw_phonemes:
            # Strip stress numbers (e.g. AH0 -> AH, AE1 -> AE)
            cleaned = re.sub(r"\d+", "", p).strip()
            if cleaned and cleaned not in [" ", "'", ",", ".", "?", "!", "-", ";", ":", '"']:
                phoneme_list.append(cleaned.upper())

        valid_label_ids = []
        valid_phoneme_str = []
        for p in phoneme_list:
            tid = self.token_to_id.get(p)
            if tid is not None:
                valid_label_ids.append(tid)
                valid_phoneme_str.append(p)
            else:
                print(f"⚠️ Warning: Phoneme '{p}' not found in model vocab -> Skipped.")

        return valid_phoneme_str, valid_label_ids

    def viterbi_forced_align(self, log_probs_np, target_ids):
        """
        Dynamic Programming / Viterbi Forced Alignment for CTC emissions.
        Given log_probs (T, V) and target_ids (U), constructs the CTC trellis
        with 2U + 1 states (alternating blank and label states) and finds the
        optimal state path that maximizes emission probability under CTC topology.

        Returns:
            unit_frames: dict mapping unit index u -> list of aligned frame indices
            path: array of shape (T,) with state index per frame
        """
        T, V = log_probs_np.shape
        U = len(target_ids)
        if U == 0:
            return {}, np.zeros(T, dtype=np.int32)

        S = 2 * U + 1  # 2U + 1 states

        state_tokens = np.zeros(S, dtype=np.int32)
        for s in range(S):
            if s % 2 == 0:
                state_tokens[s] = self.blank_id
            else:
                state_tokens[s] = target_ids[(s - 1) // 2]

        # Viterbi DP trellis in log domain
        V_mat = np.full((T, S), -np.inf, dtype=np.float64)
        backtrack = np.zeros((T, S), dtype=np.int32)

        # Initial frame t = 0
        V_mat[0, 0] = log_probs_np[0, self.blank_id]
        if S > 1:
            V_mat[0, 1] = log_probs_np[0, target_ids[0]]

        for t in range(1, T):
            # CTC boundary constraints
            min_s = max(0, S - 2 * (T - t))
            max_s = min(S, 2 * (t + 1))
            for s in range(min_s, max_s):
                token = state_tokens[s]
                p_emit = log_probs_np[t, token]

                # Option 1: stay in state s
                best_prev = s
                best_val = V_mat[t - 1, s]

                # Option 2: transition from s - 1
                if s > 0 and V_mat[t - 1, s - 1] > best_val:
                    best_val = V_mat[t - 1, s - 1]
                    best_prev = s - 1

                # Option 3: skip blank transition from s - 2 (only if odd and label differs)
                if s % 2 == 1 and s >= 2 and state_tokens[s] != state_tokens[s - 2]:
                    if V_mat[t - 1, s - 2] > best_val:
                        best_val = V_mat[t - 1, s - 2]
                        best_prev = s - 2

                V_mat[t, s] = p_emit + best_val
                backtrack[t, s] = best_prev

        # Best terminal state (CTC allows ending in final label state S-2 or trailing blank S-1)
        final_states = [S - 1, S - 2] if S > 1 else [0]
        best_s = final_states[0]
        if len(final_states) > 1 and V_mat[T - 1, final_states[1]] > V_mat[T - 1, final_states[0]]:
            best_s = final_states[1]

        # Traceback path
        path = np.zeros(T, dtype=np.int32)
        path[T - 1] = best_s
        for t in range(T - 2, -1, -1):
            path[t] = backtrack[t + 1, path[t + 1]]

        # Group frames by target unit index u:
        # State 2u + 1 belongs to target unit u
        unit_frames = {u: [] for u in range(U)}
        for t, s in enumerate(path):
            if s % 2 == 1:
                u = (s - 1) // 2
                unit_frames[u].append(t)

        return unit_frames, path

    def _logistic_calibration(self, lpp, alpha=1.2, x0=-1.8):
        """
        Maps frame-averaged Log Posterior Probability (LPP) to a calibrated
        confidence score percentage in [0, 100%].
        Parameters are calibrated so reasonable pronunciations (LPP in [-1.0, 0.0])
        score in 70-100%, fair pronunciations in 50-70%, and mispronunciations lower.
        """
        z = alpha * (lpp - x0)
        conf = 100.0 / (1.0 + np.exp(-z))
        return float(np.clip(conf, 0.0, 100.0))

    def infer_gop(self, audio_bytes, transcript_text, target_phonemes=None, method="forced_align"):
        """
        Main GOP evaluation entry point.
        By default, runs Forced-Alignment GOP (Hu et al. 2015, Witt & Young 2000, Cao et al. 2024).
        If method == 'alignment_free', delegates to alignment-free CTC loss (SDI) benchmark.
        Optionally accepts target_phonemes directly (e.g. ['B', 'AH', 'N', 'AE', 'N', 'AH']).
        """
        if not self.model:
            return {"error": "Model not loaded"}

        if method == "alignment_free":
            return self.infer_gop_alignment_free(audio_bytes, transcript_text, target_phonemes=target_phonemes)

        # 1. Load and prepare audio
        try:
            y, sr = self._load_audio(audio_bytes)
        except Exception as e:
            return {"error": f"Audio loading error: {str(e)}"}

        total_duration = len(y) / 16000.0

        # 2. Convert text to target phonemes or use provided target_phonemes
        if target_phonemes is not None:
            valid_phonemes = []
            valid_label_ids = []
            for p in target_phonemes:
                p_clean = re.sub(r"\d+", "", p).strip().upper()
                tid = self.token_to_id.get(p_clean)
                if tid is not None:
                    valid_phonemes.append(p_clean)
                    valid_label_ids.append(tid)
                else:
                    print(f"⚠️ Warning: Phoneme '{p_clean}' not in model vocab -> Skipped.")
        else:
            valid_phonemes, valid_label_ids = self.text_to_phonemes(transcript_text)
        if not valid_label_ids:
            return {"error": "No valid phonemes found in model dictionary."}

        phoneme_string = " ".join(valid_phonemes)
        U = len(valid_label_ids)

        # 3. Model Forward Pass
        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            logits = self.model(input_values).logits[0]  # (T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        log_probs_np = log_probs.cpu().numpy()
        T, V = log_probs_np.shape

        # Ensure enough frames for forced alignment trellis
        if T < 2 * U + 1:
            pad_len = (2 * U + 1) - T + 5
            pad_tensor = np.full((pad_len, V), -10.0, dtype=np.float64)
            pad_tensor[:, self.blank_id] = 0.0
            log_probs_np = np.vstack([log_probs_np, pad_tensor])
            T = log_probs_np.shape[0]

        # 4. Run Dynamic Programming Forced Alignment
        unit_frames, state_path = self.viterbi_forced_align(log_probs_np, valid_label_ids)

        # 5. Compute phone temporal intervals (start_time, end_time) and boundaries
        # For each unit u, compute start and end time by splitting blank gaps
        unit_boundaries = []
        for u in range(U):
            frames = unit_frames.get(u, [])
            if frames:
                t_first = frames[0]
                t_last = frames[-1]
            else:
                t_first = int(u * (T / U))
                t_last = t_first

            unit_boundaries.append((t_first, t_last, frames))

        # Build contiguous intervals
        unit_segments = []
        for u in range(U):
            t_first, t_last, frames = unit_boundaries[u]

            # Start frame
            if u == 0:
                s_frame = max(0, t_first - 1)
            else:
                prev_last = unit_boundaries[u - 1][1]
                s_frame = (prev_last + 1 + t_first) // 2

            # End frame
            if u == U - 1:
                e_frame = min(T, t_last + 2)
            else:
                next_first = unit_boundaries[u + 1][0]
                e_frame = (t_last + 1 + next_first) // 2

            s_time = max(0.0, s_frame * self.frame_duration)
            e_time = min(total_duration, e_frame * self.frame_duration)
            if e_time <= s_time:
                e_time = min(total_duration, s_time + self.frame_duration)

            unit_segments.append({
                "unit_idx": u,
                "phoneme": valid_phonemes[u],
                "label_id": valid_label_ids[u],
                "emission_frames": frames if frames else [t_first],
                "start_frame": int(s_frame),
                "end_frame": int(e_frame),
                "start_time": round(float(s_time), 3),
                "end_time": round(float(e_time), 3),
                "duration": round(float(e_time - s_time), 3)
            })

        # Speech bounds for smart crop (from first emission to last emission, padded by 0.1s)
        first_frame = unit_boundaries[0][0]
        last_frame = unit_boundaries[-1][1]
        speech_start = max(0.0, first_frame * self.frame_duration - 0.1)
        speech_end = min(total_duration, (last_frame + 1) * self.frame_duration + 0.1)
        speech_bounds = {
            "start": round(float(speech_start), 3),
            "end": round(float(speech_end), 3)
        }

        # 6. Compute Frame-level Goodness of Pronunciation (LPP, LPR, and Calibrated Confidence)
        details = {}
        alignment_list = []
        gop_scores_list = []
        conf_scores_list = []

        for u in range(U):
            seg = unit_segments[u]
            tid = seg["label_id"]
            p_name = seg["phoneme"]
            frames = seg["emission_frames"]

            lpp_vals = []
            lpr_vals = []

            for t in frames:
                t_clamped = min(t, log_probs.shape[0] - 1)
                target_logp = log_probs[t_clamped, tid].item()

                # Mask out target token and non-phoneme tokens (blank, pad, special)
                mask = torch.ones(V, dtype=torch.bool)
                mask[tid] = False
                for sp_id in self.special_ids:
                    if sp_id < V:
                        mask[sp_id] = False

                comp_logp = torch.max(log_probs[t_clamped, mask]).item()

                lpp_vals.append(target_logp)
                lpr_vals.append(target_logp - comp_logp)

            avg_lpp = float(np.mean(lpp_vals)) if lpp_vals else -10.0
            avg_lpr = float(np.mean(lpr_vals)) if lpr_vals else -10.0

            # Calibrate confidence score
            conf_score = self._logistic_calibration(avg_lpp)
            gop_score = avg_lpp  # Frame-averaged Log Posterior Probability

            key = f"{p_name}_{u}"
            detail_item = {
                "phoneme": p_name,
                "gop_score": round(gop_score, 4),
                "confidence_score": round(conf_score, 2),
                "lpp": round(avg_lpp, 4),
                "lpr": round(avg_lpr, 4),
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["duration"],
                "frame_interval": [seg["start_frame"], seg["end_frame"]]
            }
            details[key] = detail_item

            alignment_list.append({
                "phoneme": p_name,
                "start": seg["start_time"],
                "end": seg["end_time"],
                "gop_score": round(gop_score, 4),
                "confidence_score": round(conf_score, 2)
            })

            gop_scores_list.append(gop_score)
            conf_scores_list.append(conf_score)

        overall_score = float(np.mean(conf_scores_list)) if conf_scores_list else 0.0
        average_gop = float(np.mean(gop_scores_list)) if gop_scores_list else 0.0

        # Audit log for inspection
        self._audit_log(transcript_text, unit_segments, speech_bounds)

        return {
            "transcript_text": transcript_text,
            "transcript_phonemes": phoneme_string,
            "overall_score": round(overall_score, 1),
            "average_gop": round(average_gop, 4),
            "details": details,
            "alignment": alignment_list,
            "speech_bounds": speech_bounds
        }

    def _audit_log(self, transcript_text, unit_segments, speech_bounds):
        """Prints alignment audit summary for developer visibility."""
        num_syllables = syllapy.count(transcript_text)
        if num_syllables == 0:
            num_syllables = 1

        print("\n" + "=" * 70)
        print(f"🎯 FORCED ALIGNMENT AUDIT: '{transcript_text}' ({num_syllables} syls)")
        print(f"   Speech bounds: {speech_bounds['start']:.3f}s -> {speech_bounds['end']:.3f}s")
        print(f"   Phoneme breakdown ({len(unit_segments)} units):")
        print(f"   {'PHONE':<8} | {'TIME INTERVAL':<18} | {'FRAMES':<12}")
        print("   " + "-" * 55)
        for seg in unit_segments:
            time_str = f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s"
            frames_str = f"{seg['emission_frames']}"
            print(f"   {seg['phoneme']:<8} | {time_str:<18} | {frames_str:<12}")
        print("=" * 70 + "\n")

    # =========================================================================
    # ALIGNMENT-FREE CTC GOP (SDI) - COMPARATIVE BENCHMARK METHOD
    # =========================================================================

    def infer_gop_alignment_free(self, audio_bytes, transcript_text, target_phonemes=None):
        """
        Runs the Alignment-Free CTC Loss (SDI) Goodness of Pronunciation method
        by calculating standard CTC loss (numerator) and denominator CTC loss.
        Provided for benchmarking against Forced-Alignment GOP.
        """
        y, sr = self._load_audio(audio_bytes)
        if target_phonemes is not None:
            valid_phonemes = []
            valid_label_ids = []
            for p in target_phonemes:
                p_clean = re.sub(r"\d+", "", p).strip().upper()
                tid = self.token_to_id.get(p_clean)
                if tid is not None:
                    valid_phonemes.append(p_clean)
                    valid_label_ids.append(tid)
        else:
            valid_phonemes, valid_label_ids = self.text_to_phonemes(transcript_text)
        if not valid_label_ids:
            return {"error": "No valid phonemes found."}

        labels = torch.tensor(valid_label_ids, dtype=torch.int32)

        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            logits = self.model(input_values).logits
            post_mat = torch.nn.functional.softmax(logits[0], dim=-1).double().cpu()
            params = post_mat.transpose(0, 1)

        ll_self_cost = self.ctc_loss(params, labels, self.blank_id)

        gop_scores = {}
        alignment_list = []
        conf_scores = []
        gop_vals = []

        total_duration = len(y) / 16000.0
        predicted_ids = torch.argmax(logits[0], dim=-1).cpu()
        non_blank_indices = torch.nonzero(predicted_ids != self.blank_id, as_tuple=True)[0]

        if len(non_blank_indices) > 0:
            first_frame = non_blank_indices[0].item()
            last_frame = non_blank_indices[-1].item()
            speech_start = max(0.0, first_frame * 0.02 - 0.1)
            speech_end = min(total_duration, last_frame * 0.02 + 0.1)
        else:
            speech_start = 0.0
            speech_end = total_duration

        for i, pid in enumerate(valid_label_ids):
            ll_denom_cost = self.ctc_loss_denom(params, labels, i, self.blank_id)
            gop = -ll_self_cost + ll_denom_cost
            score = gop.item()

            token_str = valid_phonemes[i]
            key = f"{token_str}_{i}"
            conf = float(np.clip(np.exp(score) * 100.0, 0.0, 100.0))

            gop_scores[key] = {
                "phoneme": token_str,
                "gop_score": round(score, 4),
                "confidence_score": round(conf, 2),
                "start_time": 0.0,  # Alignment-free has no phone boundaries
                "end_time": 0.0
            }
            alignment_list.append({
                "phoneme": token_str,
                "start": 0.0,
                "end": 0.0,
                "gop_score": round(score, 4),
                "confidence_score": round(conf, 2)
            })
            gop_vals.append(score)
            conf_scores.append(conf)

        avg_gop = float(np.mean(gop_vals)) if gop_vals else 0.0
        overall_score = float(np.mean(conf_scores)) if conf_scores else 0.0

        return {
            "transcript_text": transcript_text,
            "transcript_phonemes": " ".join(valid_phonemes),
            "overall_score": round(overall_score, 1),
            "average_gop": round(avg_gop, 4),
            "details": gop_scores,
            "alignment": alignment_list,
            "speech_bounds": {"start": round(speech_start, 3), "end": round(speech_end, 3)}
        }

    def check_arbitrary(self, in_alphas, s, t, zero_pos=[]):
        if in_alphas[s, t].sum() > 0:
            if len(zero_pos) != 0:
                mask = torch.ones_like(in_alphas[s, t])
                for i in zero_pos:
                    mask[i] = 0
                return sum(in_alphas[s, t][mask.bool()])
            else:
                return sum(in_alphas[s, t][:])
        else:
            return 0.0

    def ctc_loss(self, params, seq, blank=0):
        """Standard CTC Loss (Numerator Forward Trellis)"""
        seqLen = seq.shape[0]
        L = 2 * seqLen + 1
        T = params.shape[1]
        alphas = torch.zeros((L, T)).double()

        alphas[0, 0] = params[blank, 0]
        if L > 1:
            alphas[1, 0] = params[seq[0], 0]

        for t in range(1, T):
            start = max(0, L - 2 * (T - t))
            for s in range(start, L):
                l = int((s - 1) / 2)
                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
                elif s == 1 or seq[l] == seq[l - 1]:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * params[seq[l], t]

        if L > 1:
            final = alphas[L - 1, T - 1] + alphas[L - 2, T - 1]
        else:
            final = alphas[0, T - 1]

        return -torch.log(final + 1e-30)

    def ctc_loss_denom(self, params, seq, pos, blank=0):
        """SDI Denominator CTC Loss"""
        seqLen = seq.shape[0]
        L = 2 * seqLen + 1
        T = params.shape[1]
        P = params.shape[0]
        alphas = torch.zeros((L, T, P)).double()

        if pos == 0:
            alphas[0, 0, 0] = params[blank, 0]
            alphas[2, 0, 0] = 0
            if seqLen > 1:
                alphas[3, 0, 0] = params[seq[1], 0]
            alphas[1, 0] = params[:, 0]
            alphas[1, 0, blank] = 0
        else:
            alphas[0, 0, 0] = params[blank, 0]
            alphas[1, 0, 0] = params[seq[0], 0]

        for t in range(1, T):
            if pos == seqLen - 1:
                lowest = L - 2 * (T - t + 1)
            else:
                lowest = L - 2 * (T - t)
            start = max(0, lowest)

            for s in range(start, L):
                l = int((s - 1) / 2)

                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t, 0] = alphas[s, t - 1, 0] * params[blank, t]
                    else:
                        sum_val = self.check_arbitrary(alphas, s - 1, t - 1, [blank])
                        if sum_val:
                            alphas[s, t, 0] = (alphas[s, t - 1, 0] + sum_val) * params[blank, t]
                        else:
                            alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[blank, t]

                elif pos != l and pos != l - 1:
                    if s == 1 or seq[l] == seq[l - 1]:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]) * params[seq[l], t]
                    else:
                        alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + alphas[s - 2, t - 1, 0]) * params[seq[l], t]

                elif pos == l - 1:
                    sum_val = self.check_arbitrary(alphas, s - 2, t - 1, [blank, seq[l]])
                    if l - 2 < 0 or seq[l - 2] == seq[l]:
                        skip_token = 0
                    else:
                        skip_token = alphas[s - 4, t - 1, 0] * params[seq[l], t]
                    skip_empty = alphas[s - 3, t - 1, 0] * params[seq[l], t]
                    alphas[s, t, 0] = (alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + sum_val) * params[seq[l], t] + skip_empty + skip_token

                else:
                    if s == 1:
                        empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                        empty_prob[blank] = 0
                        term1 = (alphas[s, t - 1, :].view(1, -1) * params[:, t].view(-1, 1)).sum(-1)
                        alphas[s, t, :] = term1 + empty_prob
                    else:
                        skip_prob = alphas[s - 2, t - 1, 0] * params[:, t]
                        skip_prob[seq[l - 1]] = 0
                        skip_prob[blank] = 0
                        empty_prob = alphas[s - 1, t - 1, 0] * params[:, t]
                        empty_prob[blank] = 0

                        term1 = (alphas[s, t - 1, :].view(1, -1) * params[:, t].view(-1, 1)).sum(-1)
                        alphas[s, t, :] = term1 + skip_prob + empty_prob

        sum_val = self.check_arbitrary(alphas, L - 2, T - 1, [blank])
        if sum_val:
            final = alphas[L - 1, T - 1, 0] + sum_val + alphas[L - 3, T - 1, 0] + alphas[L - 4, T - 1, 0]
        else:
            final = alphas[L - 1, T - 1, 0] + alphas[L - 2, T - 1, 0]

        return -torch.log(final + 1e-30)
