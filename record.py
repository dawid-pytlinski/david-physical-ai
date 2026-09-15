import sounddevice as sd
from scipy.io.wavfile import write

DEVICE_ID = 1
DURATION = 10

device_info = sd.query_devices(DEVICE_ID, "input")
sample_rate = int(device_info["default_samplerate"])

print("Urządzenie:")
print(device_info["name"])
print(f"Sample rate: {sample_rate} Hz")

print("\nNagrywanie przez 5 sekund...")
print("Powiedz coś do ReSpeakera.")

audio = sd.rec(
    int(DURATION * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="int16",
    device=DEVICE_ID
)

sd.wait()

write("test.wav", sample_rate, audio)

print("\nNagrywanie zakończone.")
print("Zapisano: test.wav")