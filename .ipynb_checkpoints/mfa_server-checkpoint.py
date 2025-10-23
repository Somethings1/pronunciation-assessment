from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.gop import GopAnalyzer
from pathlib import Path

corpus_dir = Path("data/mfa_input/dev")
dictionary_path = "/Users/admin/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"
acoustic_model_path = "/Users/admin/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip"
output_dir = Path("mfa_output_sample_gop")

# Step 1: align as usual
aligner = PretrainedAligner(
    corpus_directory=corpus_dir,
    dictionary_path=dictionary_path,
    acoustic_model_path=acoustic_model_path,
    output_directory=output_dir,
)

aligner.align()

# Step 2: compute GOP using the same model
gop_analyzer = GopAnalyzer(
    corpus_directory=corpus_dir,
    dictionary_path=dictionary_path,
    acoustic_model_path=acoustic_model_path,
    align_directory=output_dir,
    output_directory=output_dir / "gop",
)

gop_analyzer.compute_gop()

print("✅ GOP analysis complete!")
print(f"Results saved to: {output_dir / 'gop'}")
