from faster_whisper import WhisperModel

AUDIO_FILE = "test.wav"

print("Ładowanie modelu Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Model załadowany.")
print("Rozpoznawanie mowy...\n")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="pl",
    beam_size=5,
    vad_filter=True
)

print(f"Język: {info.language}")
print(f"Prawdopodobieństwo: {info.language_probability:.2f}")

print("\n--- TRANSKRYPCJA ---")

for segment in segments:
    print(
        f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
        f"{segment.text.strip()}"
    )