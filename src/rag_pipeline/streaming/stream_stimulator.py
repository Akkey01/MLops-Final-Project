import argparse, json, os, time, random, requests, pathlib
from datetime import datetime, timedelta, timezone
parser = argparse.ArgumentParser()
parser.add_argument("--rate", default="100/day")  # e.g. 5/min
args = parser.parse_args()

RATE, UNIT = args.rate.split("/")
per_second = {"sec":1, "min":60, "hour":3600, "day":86400}[UNIT[:3]]
interval = per_second / float(RATE)

files = list(pathlib.Path("/mnt/block/raw/icsi/audio").rglob("*.txt"))
random.shuffle(files)

while True:
    f = files.pop()
    text = open(f, encoding="utf-8").read()[:4000]
    payload = {"prompt": text}
    # hit friend’s model
    r1 = requests.post(os.getenv("MODEL_ENDPOINT"), json=payload).json()
    # hit Together Ai
    r2 = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"},
        json={**payload, "model":"mistralai/Mixtral-8x7B-Instruct-v0.1"}
    ).json()
    # append simple JSON log (for dashboard)
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "file": f.name,
        "friend_model": r1.get("text",""),
        "together": r2.get("choices",[{}])[0].get("text","")
    }
    with open("/mnt/block/stream_logs.jsonl", "a") as fp:
        fp.write(json.dumps(out)+"\n")
    time.sleep(interval)
