#!/usr/bin/env python3
import time, json, argparse, pathlib, requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prod_json",
                   default="/data/processed/prod_seed_chunks.jsonl",
                   help="JSONL of chunks to stream")
    p.add_argument("--endpoint",
                   default=os.getenv("RAG_ENDPOINT_URL"),
                   help="Your RAG predict URL")
    p.add_argument("--rate", type=float, default=0.2,
                   help="Requests per second")
    args = p.parse_args()

    # load chunks
    recs = [json.loads(l) for l in pathlib.Path(args.prod_json).read_text().splitlines()]
    for rec in recs:
        query = rec["chunk"]
        t0 = time.time()
        try:
            r = requests.post(args.endpoint, json={"query": query})
            status = r.status_code
        except:
            status = None
        latency = time.time() - t0

        # log it
        log = {"timestamp": time.time(), "latency": latency, "status": status}
        with open("/data/metrics/stream_metrics.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

        time.sleep(1.0 / args.rate)

if __name__ == "__main__":
    import os
    main()
