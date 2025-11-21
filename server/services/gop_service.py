import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import io
import logging
import re
from g2p_en import G2p

logging.getLogger("transformers").setLevel(logging.ERROR)


class GOPEvaluator:
    def __init__(self, model_path):
        print(f"⏳ Loading GOP Model from: {model_path}...")
        self.g2p = G2p()
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
            self.model.eval()

            self.vocab = self.processor.tokenizer.get_vocab()
            self.blank_id = self.processor.tokenizer.pad_token_id
            if self.blank_id is None:
                self.blank_id = 0

            has_numbers = any(re.search(r'\d', k) for k in self.vocab.keys() if len(k) > 1)
            self.strip_stress = not has_numbers

            print(f"✅ Loaded. Vocab: {len(self.vocab)} tokens. Auto-strip numbers: {self.strip_stress}")

        except Exception as e:
            print(f"❌ Failed to load GOP model: {e}")
            self.model = None

    def infer_gop(self, audio_bytes, transcript_text):
        if not self.model:
            return {"error": "Model chưa load"}

        # 1. Preprocess Audio
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            # Cắt khoảng lặng & Normalize
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) < 1000:
                y_trimmed = y
            y_normalized = librosa.util.normalize(y_trimmed)
        except Exception as e:
            return {"error": f"Audio lỗi: {str(e)}"}

        # 2. Convert Text -> Phonemes
        raw_phonemes = self.g2p(transcript_text)
        phoneme_list = [p for p in raw_phonemes if p not in [" ", "'", ",", ".", "?", "!"]]

        if self.strip_stress:
            phoneme_list = [re.sub(r'\d+', '', p) for p in phoneme_list]

        valid_label_ids = []
        valid_phoneme_str = []

        for p in phoneme_list:
            tid = self.vocab.get(p)
            if tid is not None:
                valid_label_ids.append(tid)
                valid_phoneme_str.append(p)
            else:
                print(f"⚠️ Warning: Phoneme '{p}' không có trong vocab -> Bỏ qua.")

        if not valid_label_ids:
            return {"error": "Không tìm thấy phoneme nào hợp lệ trong từ điển model."}

        phoneme_string = " ".join(valid_phoneme_str)
        print(f"🔍 Phonemes xử lý: {phoneme_string}")

        labels = torch.tensor(valid_label_ids, dtype=torch.int32)

        # 3. Forward Model
        with torch.no_grad():
            inputs = self.processor(y_normalized, sampling_rate=16000, return_tensors="pt")
            logits = self.model(inputs.input_values).logits

        # 4. Tính Posterior (Dùng Softmax thường + Double Precision)
        post_mat = torch.nn.functional.softmax(logits[0], dim=-1).double()
        params = post_mat.transpose(0, 1)

        # 5. Tính GOP (Logic gốc: Cost = -LogLikelihood)
        ll_self_cost = self.ctc_loss(params, labels, self.blank_id)

        gop_scores = {}
        tokens = self.processor.tokenizer.convert_ids_to_tokens(valid_label_ids)

        for i, pid in enumerate(valid_label_ids):
            ll_denom_cost = self.ctc_loss_denom(params, labels, i, self.blank_id)

            # GOP = log(P_cano / P_denom) = log(P_cano) - log(P_denom)
            #     = (-ll_self_cost) - (-ll_denom_cost)
            #     = ll_denom_cost - ll_self_cost
            # Kết quả: Số ÂM (hoặc xấp xỉ 0). Càng gần 0 càng tốt.
            gop = -ll_self_cost + ll_denom_cost
            score = gop.item()

            token_str = tokens[i]
            key = f"{token_str}_{i}"

            # Confidence: exp(score)
            conf = np.exp(score) * 100

            gop_scores[key] = {
                "phoneme": token_str,
                "gop_score": round(score, 4),
                "confidence_score": round(conf, 2)
            }

        return {
            "transcript_text": transcript_text,
            "transcript_phonemes": phoneme_string,
            "average_gop": np.mean([v['gop_score'] for v in gop_scores.values()]) if gop_scores else 0,
            "details": gop_scores
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
        """Standard CTC Loss"""
        seqLen = seq.shape[0]
        L = 2*seqLen + 1
        T = params.shape[1]
        alphas = torch.zeros((L, T)).double()

        alphas[0, 0] = params[blank, 0]
        if L > 1:
            alphas[1, 0] = params[seq[0], 0]

        for t in range(1, T):
            start = max(0, L - 2*(T-t))
            for s in range(start, L):
                l = int((s - 1) / 2)
                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
                elif s == 1 or seq[l] == seq[l-1]:
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
            if pos == seqLen-1:
                lowest = L-2*(T-t+1)
            else:
                lowest = L-2*(T-t)
            start = max(0,  lowest)

            for s in range(start,  L):
                l = int((s-1)/2)

                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t, 0] = alphas[s, t-1, 0] * params[blank, t]
                    else:
                        sum_val = self.check_arbitrary(alphas,  s-1,  t-1,  [blank])
                        if sum_val:
                            alphas[s, t, 0] = (alphas[s, t-1, 0] + sum_val) * params[blank, t]
                        else:
                            alphas[s, t, 0] = (alphas[s, t-1, 0] + alphas[s-1, t-1, 0]) * params[blank, t]

                elif pos != l and pos != l-1:
                    if s == 1 or seq[l] == seq[l-1]:
                        alphas[s, t, 0] = (alphas[s, t-1, 0] + alphas[s-1, t-1, 0]) * params[seq[l], t]
                    else:
                        alphas[s, t, 0] = (alphas[s, t-1, 0] + alphas[s-1, t-1, 0] + alphas[s-2, t-1, 0]) * params[seq[l], t]

                elif pos == l-1:
                    sum_val = self.check_arbitrary(alphas,  s-2,  t-1,  [blank, seq[l]])
                    if l-2 < 0 or seq[l-2] == seq[l]:
                        skip_token = 0
                    else:
                        skip_token = alphas[s-4, t-1, 0] * params[seq[l], t]
                    skip_empty = alphas[s-3, t-1, 0] * params[seq[l], t]
                    alphas[s, t, 0] = (alphas[s, t-1, 0] + alphas[s-1, t-1, 0] + sum_val) * params[seq[l], t] + skip_empty + skip_token

                else:
                    if s == 1:
                        empty_prob = alphas[s-1, t-1, 0] * params[:, t]
                        empty_prob[blank] = 0
                        # Broadcast
                        term1 = (alphas[s, t-1, :].view(1, -1) * params[:, t].view(-1, 1)).sum(-1)
                        alphas[s, t, :] = term1 + empty_prob
                    else:
                        skip_prob = alphas[s-2, t-1, 0] * params[:, t]
                        skip_prob[seq[l-1]] = 0
                        skip_prob[blank] = 0
                        empty_prob = alphas[s-1, t-1, 0] * params[:, t]
                        empty_prob[blank] = 0

                        term1 = (alphas[s, t-1, :].view(1, -1) * params[:, t].view(-1, 1)).sum(-1)
                        alphas[s, t, :] = term1 + skip_prob + empty_prob

        sum_val = self.check_arbitrary(alphas,  L-2,  T-1,  [blank])
        if sum_val:
            final = alphas[L-1,  T-1,  0] + sum_val + alphas[L-3,  T-1,  0] + alphas[L-4,  T-1,  0]
        else:
            final = alphas[L-1,  T-1,  0] + alphas[L-2,  T-1,  0]

        return -torch.log(final + 1e-30)
