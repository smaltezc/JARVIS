import argparse
import os
import subprocess
import sys
import uuid

import requests
import yaml

parser = argparse.ArgumentParser(
    description="Talk to Jarvis with your voice and hear it answer back."
)
parser.add_argument("--config", type=str, default="configs/config.default.yaml")
parser.add_argument("--audio", type=str, default=None,
                    help="Answer this audio file once instead of recording from the microphone.")
parser.add_argument("--seconds", type=float, default=5.0,
                    help="How long to record each turn from the microphone.")
parser.add_argument("--backend", type=str, default=None, choices=["local", "huggingface"],
                    help="Where the speech models run. Defaults to the config's inference_mode.")
parser.add_argument("--asr-model", type=str, default="openai/whisper-base")
parser.add_argument("--tts-model", type=str, default="microsoft/speecht5_tts")
parser.add_argument("--no-play", action="store_true", help="Write the reply audio but do not play it.")
args = parser.parse_args()

config = yaml.load(open(args.config), Loader=yaml.FullLoader)

BACKEND = args.backend or ("local" if config["inference_mode"] == "local" else "huggingface")
HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {config['huggingface']['token']}"}
LOCAL_ENDPOINT = "http://{host}:{port}".format(**config["local_inference_endpoint"])
JARVIS_ENDPOINT = "http://{host}:{port}/hugginggpt".format(
    host="localhost" if config["http_listen"]["host"] == "0.0.0.0" else config["http_listen"]["host"],
    port=config["http_listen"]["port"],
)
AUDIO_DIR = "public/audios"
os.makedirs(AUDIO_DIR, exist_ok=True)


def record(seconds):
    """Capture `seconds` of microphone audio and return the path to a 16kHz wav."""
    try:
        import sounddevice
        import soundfile
    except ImportError:
        sys.exit("Microphone capture needs `pip install sounddevice soundfile`, "
                 "or pass --audio <file> to skip recording.")

    rate = 16000
    print(f"[ listening ] {seconds:g}s...", flush=True)
    frames = sounddevice.rec(int(seconds * rate), samplerate=rate, channels=1)
    sounddevice.wait()
    path = os.path.join(AUDIO_DIR, f"in_{str(uuid.uuid4())[:4]}.wav")
    soundfile.write(path, frames, rate)
    return path


def transcribe(audio_path):
    """Speech in, text out."""
    if BACKEND == "local":
        response = requests.post(
            f"{LOCAL_ENDPOINT}/models/{args.asr_model}",
            json={"audio_url": os.path.abspath(audio_path)},
        )
        response.raise_for_status()
        return response.json()["text"].strip()

    with open(audio_path, "rb") as f:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{args.asr_model}",
            headers=HUGGINGFACE_HEADERS,
            data=f.read(),
        )
    response.raise_for_status()
    return response.json()["text"].strip()


def ask_jarvis(messages):
    """Text in, text out -- this is the existing /hugginggpt pipeline, untouched."""
    response = requests.post(JARVIS_ENDPOINT, json={"messages": messages})
    response.raise_for_status()
    answer = response.json()
    if "error" in answer:
        raise RuntimeError(answer["error"])
    return answer["message"]


def synthesize(text):
    """Text in, path to an audio file out."""
    if BACKEND == "local":
        response = requests.post(
            f"{LOCAL_ENDPOINT}/models/{args.tts_model}",
            json={"text": text},
        )
        response.raise_for_status()
        # models_server writes into its own public/ and serves no static files,
        # so this only resolves when it shares a filesystem with us.
        path = "public" + response.json()["path"]
        if not os.path.exists(path):
            raise RuntimeError(
                f"models_server wrote {path} on its own machine. Run this script there, "
                f"share the volume, or use --backend huggingface."
            )
        return path

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{args.tts_model}",
        headers=HUGGINGFACE_HEADERS,
        json={"inputs": text},
    )
    response.raise_for_status()
    path = os.path.join(AUDIO_DIR, f"out_{str(uuid.uuid4())[:4]}.flac")
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def play(path):
    for player in (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                   ["aplay", path],
                   ["afplay", path]):
        try:
            subprocess.run(player, check=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"[ no audio player found -- the reply is saved at {path} ]")


def turn(messages, audio_path):
    """One full loop: audio in, audio out."""
    heard = transcribe(audio_path)
    if not heard:
        print("[ heard nothing ]")
        return
    print(f"[ You ]: {heard}")

    messages.append({"role": "user", "content": heard})
    answer = ask_jarvis(messages)
    messages.append({"role": "assistant", "content": answer})
    print(f"[ Jarvis ]: {answer}")

    spoken = synthesize(answer)
    print(f"[ spoken ]: {spoken}")
    if not args.no_play:
        play(spoken)


if __name__ == "__main__":
    print(f"Jarvis voice loop -- speech models on `{BACKEND}`, chat via {JARVIS_ENDPOINT}.")
    messages = []

    if args.audio:
        try:
            turn(messages, args.audio)
        except Exception as e:
            sys.exit(f"[ error ]: {e}")
        sys.exit(0)

    print("Press Enter to speak, or type `exit` to quit.")
    while True:
        if input().strip() == "exit":
            break
        try:
            turn(messages, record(args.seconds))
        except Exception as e:
            print(f"[ error ]: {e}")
