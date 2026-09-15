import re
import socket
import time
import unicodedata
import winsound
from collections import deque
from math import gcd

import numpy as np
import sounddevice as sd
import webrtcvad

from scipy.io.wavfile import write
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION
# ============================================================

WAKE_WORD = "david"

FRAME_MS = 30

VAD_SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 2

PRE_ROLL_MS = 300
END_SILENCE_MS = 700

MIN_SPEECH_MS = 180

COMMAND_TIMEOUT = 8.0

WAKE_AUDIO_FILE = "wake.wav"
COMMAND_AUDIO_FILE = "command.wav"


# ============================================================
# ROS 2 / WSL UDP BRIDGE
# ============================================================

# Aktualny adres WSL z:
# hostname -I
ROS_BRIDGE_HOST = "172.26.99.219"

ROS_BRIDGE_PORT = 5055

ROS_ALLOWED_INTENTS = {
    "HAND_OPEN",
    "HAND_CLOSE",
    "HAND_PEACE",
    "EMERGENCY_STOP",
}

udp_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)


def send_ros_intent(intent: str):
    """
    Wysyła do bridge'a w WSL wyłącznie intencje,
    które mogą sterować warstwą wykonawczą ROS 2.

    UNKNOWN oraz SYSTEM_EXIT nie są publikowane
    jako komendy dłoni.
    """

    if intent not in ROS_ALLOWED_INTENTS:
        return False

    try:

        udp_socket.sendto(
            intent.encode("utf-8"),
            (
                ROS_BRIDGE_HOST,
                ROS_BRIDGE_PORT,
            ),
        )

        print(
            f"ROS BRIDGE: sent {intent}"
        )

        return True

    except OSError as error:

        print(
            f"ROS BRIDGE ERROR: {error}"
        )

        return False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower().strip()

    # Znaki, których unicodedata nie upraszcza poprawnie,
    # szczególnie polskie "ł".
    polish_chars = str.maketrans({
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
    })

    text = text.translate(
        polish_chars
    )

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# RESPEAKER DETECTION
# ============================================================

def find_respeaker_input():

    devices = sd.query_devices()

    default_input = sd.default.device[0]

    if (
        default_input is not None
        and default_input >= 0
    ):

        device = devices[
            default_input
        ]

        if (
            "respeaker"
            in device["name"].lower()
            and device["max_input_channels"] > 0
        ):
            return default_input

    for index, device in enumerate(
        devices
    ):

        if (
            "respeaker"
            in device["name"].lower()
            and device["max_input_channels"] > 0
        ):
            return index

    raise RuntimeError(
        "Nie znaleziono mikrofonu ReSpeaker Lite."
    )


DEVICE_ID = find_respeaker_input()

device_info = sd.query_devices(
    DEVICE_ID,
    "input"
)

SAMPLE_RATE = int(
    device_info["default_samplerate"]
)

FRAME_SAMPLES = int(
    SAMPLE_RATE
    * FRAME_MS
    / 1000
)


# ============================================================
# VAD
# ============================================================

vad = webrtcvad.Vad(
    VAD_AGGRESSIVENESS
)


def convert_for_vad(audio_frame):

    """
    ReSpeaker pracuje natywnie np. 44100 Hz.

    WebRTC VAD oczekuje:
    8000 / 16000 / 32000 / 48000 Hz.

    Dlatego małe ramki przeskalowujemy do 16 kHz.
    """

    if SAMPLE_RATE == VAD_SAMPLE_RATE:

        converted = audio_frame

    else:

        divisor = gcd(
            SAMPLE_RATE,
            VAD_SAMPLE_RATE
        )

        up = (
            VAD_SAMPLE_RATE
            // divisor
        )

        down = (
            SAMPLE_RATE
            // divisor
        )

        converted = resample_poly(
            audio_frame,
            up,
            down
        )

    converted = np.clip(
        converted,
        -32768,
        32767
    ).astype(np.int16)

    return converted.tobytes()


# ============================================================
# AUDIO CAPTURE
# ============================================================

def capture_utterance(
    stream,
    timeout=None,
    max_duration=8.0
):

    pre_roll_frames = max(
        1,
        PRE_ROLL_MS // FRAME_MS
    )

    end_silence_frames = max(
        1,
        END_SILENCE_MS // FRAME_MS
    )

    minimum_voice_frames = max(
        1,
        MIN_SPEECH_MS // FRAME_MS
    )

    pre_roll = deque(
        maxlen=pre_roll_frames
    )

    recording = False

    recorded_frames = []

    voice_frames = 0
    silence_frames = 0

    wait_started = time.monotonic()
    speech_started = None

    while True:

        data, overflowed = stream.read(
            FRAME_SAMPLES
        )

        if overflowed:
            print(
                "AUDIO WARNING: input overflow"
            )

        frame = data[:, 0].copy()

        vad_audio = convert_for_vad(
            frame
        )

        is_speech = vad.is_speech(
            vad_audio,
            VAD_SAMPLE_RATE
        )

        if not recording:

            pre_roll.append(
                frame
            )

            if is_speech:

                recording = True

                speech_started = (
                    time.monotonic()
                )

                recorded_frames.extend(
                    list(pre_roll)
                )

                voice_frames = 1
                silence_frames = 0

            elif (
                timeout is not None
                and
                time.monotonic()
                - wait_started
                >= timeout
            ):

                return None

        else:

            recorded_frames.append(
                frame
            )

            if is_speech:

                voice_frames += 1
                silence_frames = 0

            else:

                silence_frames += 1

            if (
                silence_frames
                >= end_silence_frames
                and
                voice_frames
                >= minimum_voice_frames
            ):

                return np.concatenate(
                    recorded_frames
                )

            if (
                time.monotonic()
                - speech_started
                >= max_duration
            ):

                return np.concatenate(
                    recorded_frames
                )


# ============================================================
# INTENT ENGINE
# ============================================================

def detect_intent(text: str):

    text = normalize_text(
        text
    )

    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in [
            "zakoncz",
            "koniec",
            "wylacz system",
        ]
    ):

        return "SYSTEM_EXIT"

    # --------------------------------------------------------
    # HAND OPEN
    # --------------------------------------------------------

    if (
        "otworz dlon" in text
        or
        "otworz reke" in text
    ):

        return "HAND_OPEN"

    # --------------------------------------------------------
    # HAND CLOSE / FIST
    # --------------------------------------------------------

    if (
        "zamknij dlon" in text
        or
        "zamknij reke" in text
        or
        "zacisnij piesc" in text
    ):

        return "HAND_CLOSE"

    # --------------------------------------------------------
    # PEACE / TWO
    # --------------------------------------------------------

    if (
        "pokaz dwa" in text
        or
        "pokaz dwojke" in text
    ):

        return "HAND_PEACE"

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    if text in [
        "stop",
        "zatrzymaj",
        "awaryjny stop",
    ]:

        return "EMERGENCY_STOP"

    return "UNKNOWN"


# ============================================================
# MODELS
# ============================================================

print()
print(
    "Ładowanie modeli AI..."
)

wake_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

command_model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print(
    "Modele gotowe."
)


# ============================================================
# WHISPER
# ============================================================

def detect_wake_word(audio):

    write(
        WAKE_AUDIO_FILE,
        SAMPLE_RATE,
        audio
    )

    segments, _ = (
        wake_model.transcribe(
            WAKE_AUDIO_FILE,
            language="en",
            beam_size=5,
            vad_filter=True,
            hotwords="David"
        )
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    normalized = normalize_text(
        text
    )

    if text:

        print(
            f"WAKE HEARD: {text}"
        )

    return (
        WAKE_WORD
        in normalized
    )


def transcribe_command(audio):

    write(
        COMMAND_AUDIO_FILE,
        SAMPLE_RATE,
        audio
    )

    segments, _ = (
        command_model.transcribe(
            COMMAND_AUDIO_FILE,
            language="pl",
            beam_size=5,
            vad_filter=True,
            hotwords=(
                "otwórz dłoń, "
                "otwórz rękę, "
                "zamknij dłoń, "
                "zamknij rękę, "
                "zaciśnij pięść, "
                "pokaż dwa, "
                "stop, "
                "zakończ"
            )
        )
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()


# ============================================================
# STATUS
# ============================================================

def print_system_status():

    print()
    print(
        "========================================"
    )
    print(
        " DAVID - PHYSICAL AI VOICE INTERFACE"
    )
    print(
        "========================================"
    )
    print()

    print(
        f"Microphone: {device_info['name']}"
    )

    print(
        f"Native sample rate: "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"ROS Bridge: "
        f"{ROS_BRIDGE_HOST}:"
        f"{ROS_BRIDGE_PORT}"
    )

    print()

    print(
        "STATE: IDLE"
    )

    print()

    print(
        'Powiedz po angielsku: "David"'
    )

    print()

    print(
        "CTRL+C = zakończenie"
    )

    print()


# ============================================================
# MAIN LOOP
# ============================================================

print_system_status()


try:

    with sd.InputStream(
        device=DEVICE_ID,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES
    ) as stream:

        while True:

            # =================================================
            # IDLE / WAKE WORD
            # =================================================

            wake_audio = capture_utterance(
                stream,
                timeout=None,
                max_duration=3.0
            )

            if wake_audio is None:
                continue

            if not detect_wake_word(
                wake_audio
            ):
                continue

            print()

            print(
                ">>> DAVID DETECTED <<<"
            )

            # -------------------------------------------------
            # BEEP
            # -------------------------------------------------

            winsound.Beep(
                1000,
                180
            )

            # Nie pozwalamy, aby beep został potraktowany
            # jako następna komenda.
            time.sleep(
                0.25
            )

            print()
            print(
                "STATE: ACTIVE"
            )

            print(
                "Słucham..."
            )

            print()

            # =================================================
            # COMMAND
            # =================================================

            command_audio = capture_utterance(
                stream,
                timeout=COMMAND_TIMEOUT,
                max_duration=7.0
            )

            if command_audio is None:

                print(
                    "SYSTEM: nie usłyszałem komendy."
                )

                print(
                    "STATE: IDLE"
                )

                print()

                continue

            print(
                "Rozpoznawanie komendy..."
            )

            text = transcribe_command(
                command_audio
            )

            normalized_text = normalize_text(
                text
            )

            intent = detect_intent(
                text
            )

            # =================================================
            # DIAGNOSTICS
            # =================================================

            print()

            print(
                "--------------------------------"
            )

            print(
                f"HEARD:      {text}"
            )

            print(
                f"NORMALIZED: {normalized_text}"
            )

            print(
                f"INTENT:     {intent}"
            )

            print(
                "--------------------------------"
            )

            # =================================================
            # ROS 2 TRANSPORT
            # =================================================

            ros_sent = send_ros_intent(
                intent
            )

            # =================================================
            # EXECUTION / LOCAL STATUS
            # =================================================

            if intent == "SYSTEM_EXIT":

                print(
                    "SYSTEM: kończę działanie."
                )

                break

            elif intent == "HAND_OPEN":

                print(
                    "HAND: OPEN"
                )

            elif intent == "HAND_CLOSE":

                print(
                    "HAND: CLOSE"
                )

            elif intent == "HAND_PEACE":

                print(
                    "HAND: PEACE GESTURE"
                )

            elif intent == "EMERGENCY_STOP":

                print(
                    "SYSTEM: EMERGENCY STOP"
                )

            else:

                print(
                    "SYSTEM: nieznana komenda."
                )

            print()

            if ros_sent:

                print(
                    "ROS STATUS: command forwarded."
                )

            print(
                "STATE: IDLE"
            )

            print()


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print()

    print(
        "SYSTEM: przerwano przez CTRL+C."
    )


# ============================================================
# CLEANUP
# ============================================================

finally:

    udp_socket.close()

    print(
        "ROS BRIDGE: socket closed."
    )

    print(
        "DAVID: OFFLINE"
    )