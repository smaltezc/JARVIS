"""Prove the voice loop is wired correctly without models, keys or a microphone.

Stands up fake ASR / TTS / Jarvis endpoints, runs voice_chat.py against them,
and checks that audio went in, text flowed through, and audio came back out.

    python selftest_voice.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HEARD = "Selam Jarvis, arabanin farlarini ac."
SPOKEN = "Farlari actim efendim."
seen = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(self.path)

        if "whisper" in self.path or "asr" in self.path:
            assert os.path.isabs(body["audio_url"]), \
                f"ASR needs an absolute path, got {body['audio_url']!r}"
            assert os.path.exists(body["audio_url"]), "ASR was handed a path that does not exist"
            out = {"text": f"  {HEARD}  "}          # padded: the loop must strip it
        elif "tts" in self.path:
            os.makedirs("public/audios", exist_ok=True)
            with open("public/audios/selftest_reply.wav", "wb") as f:
                f.write(b"RIFF" + b"\0" * 40)
            out = {"path": "/audios/selftest_reply.wav"}
        elif "hugginggpt" in self.path:
            assert body["messages"][-1]["content"] == HEARD, \
                f"Jarvis got {body['messages'][-1]['content']!r}, expected the transcript"
            out = {"message": SPOKEN}
        else:
            out = {"error": f"unexpected call to {self.path}"}

        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(port):
    server = HTTPServer(("127.0.0.1", port), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    jarvis_port, models_port = 8114, 8115
    serve(jarvis_port)
    serve(models_port)

    config = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    config.write(
        "huggingface:\n  token: selftest\n"
        "inference_mode: local\n"
        f"http_listen:\n  host: 127.0.0.1\n  port: {jarvis_port}\n"
        f"local_inference_endpoint:\n  host: 127.0.0.1\n  port: {models_port}\n"
    )
    config.close()

    os.makedirs("public/audios", exist_ok=True)
    clip = "public/audios/selftest_input.wav"
    with open(clip, "wb") as f:
        f.write(b"RIFF" + b"\0" * 40)

    run = subprocess.run(
        [sys.executable, "voice_chat.py", "--config", config.name,
         "--backend", "local", "--audio", clip, "--no-play"],
        capture_output=True, text=True,
    )
    print(run.stdout, end="")
    if run.stderr:
        print(run.stderr, end="")

    for path in (clip, "public/audios/selftest_reply.wav", config.name):
        if os.path.exists(path):
            os.remove(path)

    problems = []
    if run.returncode != 0:
        problems.append(f"voice_chat.py exited {run.returncode}")
    if f"[ You ]: {HEARD}" not in run.stdout:
        problems.append("the transcript never reached the console")
    if f"[ Jarvis ]: {SPOKEN}" not in run.stdout:
        problems.append("Jarvis's answer never came back")
    if "[ spoken ]" not in run.stdout:
        problems.append("no reply audio was produced")
    for stage, needle in (("ASR", "whisper"), ("Jarvis", "hugginggpt"), ("TTS", "tts")):
        if not any(needle in p for p in seen):
            problems.append(f"the {stage} stage was never called")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nPASS: audio in -> transcript -> Jarvis -> reply audio. All four stages fired.")
