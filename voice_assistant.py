import re
import unicodedata

import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


DEVICE_ID = 1
DURATION = 5
AUDIO_FILE = "command.wav"

# Komendy dozwolone przez system.
# Na tym etapie NIC jeszcze fizycznie nie steruje dronem.
COMMANDS = {
    "SYSTEM_EXIT": [
        "koniec",
        "zakoncz",
        "zakonczam",
        "wylacz system",
        "wylacz program",
    ],

    "DRONE_TAKEOFF": [
        "wystartuj",
        "rozpocznij lot",
        "startuj",
    ],

    "DRONE_LAND": [
        "wyladuj",
        "laduj",
        "rozpocznij ladowanie",
    ],

    "EMERGENCY_STOP": [
        "stop",
        "awaryjny stop",
        "zatrzymaj",
    ],
}


def normalize_text(text: str) -> str:
    """
    Normalizacja tekstu z Whispera.

    Przykład:
        'Zakończ.' -> 'zakoncz'
        'Wystartuj, rozpocznij lot!' ->
        'wystartuj rozpocznij lot'
    """

    text = text.lower().strip()

    # Usuń polskie znaki / diakrytykę:
    # ą -> a, ć -> c, ń -> n itd.
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    # Usuń interpunkcję
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Usuń wielokrotne spacje
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_intent(text: str):
    """
    Zamienia tekst na bezpieczną, zdefiniowaną intencję.
    """

    normalized = normalize_text(text)

    # Wyjście z programu
    if normalized in COMMANDS["SYSTEM_EXIT"]:
        return "SYSTEM_EXIT"

    # Emergency stop traktujemy restrykcyjnie.
    if normalized in COMMANDS["EMERGENCY_STOP"]:
        return "EMERGENCY_STOP"

    # Start drona
    if (
        normalized in COMMANDS["DRONE_TAKEOFF"]
        or normalized.startswith("wystartuj")
        or "rozpocznij lot" in normalized
    ):
        return "DRONE_TAKEOFF"

    # Lądowanie
    if (
        normalized in COMMANDS["DRONE_LAND"]
        or normalized.startswith("wyladuj")
        or "rozpocznij ladowanie" in normalized
    ):
        return "DRONE_LAND"

    return "UNKNOWN"


# -------------------------------------------------------
# AUDIO
# -------------------------------------------------------

device_info = sd.query_devices(DEVICE_ID, "input")
sample_rate = int(device_info["default_samplerate"])

print("Ładowanie Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print()
print("====================================")
print(" Physical AI Voice Interface v0.2")
print("====================================")
print(f"Mikrofon: {device_info['name']}")
print(f"Sample rate: {sample_rate} Hz")
print()
print("System gotowy.")
print()


try:

    while True:

        input("Naciśnij ENTER i powiedz komendę...")

        print("Słucham...")

        audio = sd.rec(
            int(DURATION * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=DEVICE_ID,
        )

        sd.wait()

        write(
            AUDIO_FILE,
            sample_rate,
            audio
        )

        print("Rozpoznawanie...")

        segments, info = model.transcribe(
            AUDIO_FILE,

            language="pl",

            beam_size=5,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 300
            },

            # Podpowiadamy Whisperowi słownictwo
            # występujące w naszym systemie.
            hotwords=(
                "wystartuj, rozpocznij lot, "
                "wyląduj, zakończ, wyłącz system, "
                "stop, awaryjny stop, Physical AI"
            )
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        print()
        print("------------------------------------")
        print(f"HEARD:      {text}")
        print(f"NORMALIZED: {normalize_text(text)}")

        intent = detect_intent(text)

        print(f"INTENT:     {intent}")
        print("------------------------------------")
        print()

        # ---------------------------------------
        # EXECUTION LAYER
        # ---------------------------------------

        if intent == "SYSTEM_EXIT":

            print("SYSTEM: kończę działanie.")
            break

        elif intent == "DRONE_TAKEOFF":

            print("SYSTEM: rozpoznano polecenie START LOTU.")

            # Jeszcze NIE uruchamiamy drona.
            # Tutaj później pojawi się:
            #
            # drone.takeoff()

        elif intent == "DRONE_LAND":

            print("SYSTEM: rozpoznano polecenie LĄDOWANIA.")

            # Później:
            #
            # drone.land()

        elif intent == "EMERGENCY_STOP":

            print("SYSTEM: EMERGENCY STOP.")

        else:

            print("SYSTEM: nie rozpoznano komendy.")


except KeyboardInterrupt:

    print()
    print()
    print("SYSTEM: przerwano z klawiatury.")

finally:

    print("SYSTEM: zamknięty.")