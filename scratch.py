from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("/Users/admin/Documents/Personal/projects/pronunciation-assessment/models/wav2vec2-large-lv60_phoneme-timit_english_timit-4k", local_files_only=True)
print(list(model.config.id2label.values())[:100])
print("Total phones:", len(model.config.id2label))
