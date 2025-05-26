#!/usr/bin/env python3
import os, json, pathlib, random, argparse, asyncio, time
import aiohttp

# ─── Helpers ───────────────────────────────────────────────────────────────────
async def call_friend(session, url, payload, timeout=15):
    """Call the friend model endpoint with better error handling and increased timeout"""
    try:
        # Increased timeout from 10s to 15s
        async with session.post(url, json=payload, timeout=timeout) as r:
            if r.status != 200:
                return f"⚠️ model error: HTTP {r.status} - {await r.text()}"

            data = await r.json()
            return data.get("text", "")
    except asyncio.TimeoutError:
        return f"⚠️ model error: Connection timed out after {timeout}s"
    except aiohttp.ClientConnectorError as e:
        return f"⚠️ model error: Connection failed - {e}"
    except Exception as e:
        return f"⚠️ model error: {e}"

async def call_together(session, api_key, prompt, timeout=15):
    """Call the Together API with better error handling and increased timeout"""
    if not api_key:
        return "⚠️ together error: No API key provided"

    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 50  # Adding explicit max_tokens parameter
    }

    try:
        # Increased timeout from 10s to 15s
        async with session.post(
            "https://api.together.xyz/v1/completions",
            headers=headers, json=body, timeout=timeout
        ) as r:
            if r.status != 200:
                return f"⚠️ together error: HTTP {r.status} - {await r.text()}"

            data = await r.json()

            # More robust handling of the response
            if "choices" not in data:
                return f"⚠️ together error: Missing 'choices' in response: {json.dumps(data)[:100]}..."

            if not data["choices"] or "text" not in data["choices"][0]:
                return f"⚠️ together error: Invalid response format"

            return data["choices"][0].get("text", "")
    except asyncio.TimeoutError:
        return f"⚠️ together error: Connection timed out after {timeout}s"
    except aiohttp.ClientConnectorError as e:
        return f"⚠️ together error: Connection failed - {e}"
    except Exception as e:
        return f"⚠️ together error: {e}"

async def process_batch(files, friend_url, together_key, max_tokens):
    """Process a batch of files with better error handling"""
    if not friend_url:
        print("⚠️ Warning: MODEL_ENDPOINT not set")

    if not together_key:
        print("⚠️ Warning: TOGETHER_API_KEY not set")

    # TCP connection pooling and more resilient session
    conn = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        for fpath in files:
            try:
                text = fpath.read_text(errors="ignore")[:4000]
                payload = {"text": text, "max_new_tokens": max_tokens}
                tasks.append(call_friend(session, friend_url, payload))
                tasks.append(call_together(session, together_key, text))
            except Exception as e:
                print(f"Error preparing file {fpath}: {e}")
                continue

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        results = [
            str(r) if isinstance(r, Exception) else r
            for r in results
        ]

    # interleave results: [f0,t0, f1,t1, ...]
    records = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i, fpath in enumerate(files):
        if 2*i+1 >= len(results):
            continue  # Skip if we don't have enough results

        a1 = results[2*i]
        a2 = results[2*i+1]
        rec = {
            "ts": ts,
            "file": fpath.name,
            "friend_model": a1,
            "together": a2
        }
        records.append(rec)

    # Make sure the output directory exists
    os.makedirs("/mnt/block", exist_ok=True)

    # Write results to log file
    with open("/mnt/block/stream_logs.jsonl", "a") as out:
        for rec in records:
            out.write(json.dumps(rec)+"\n")

    return len(records)

# ─── Network Check ────────────────────────────────────────────────────────
async def check_connectivity():
    """Check network connectivity to key endpoints"""
    conn = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=conn) as session:
        results = []

        # Test internet connectivity
        try:
            async with session.get("https://api.together.xyz", timeout=5) as r:
                results.append(f"✅ Internet connectivity: HTTP {r.status}")
        except Exception as e:
            results.append(f"❌ Internet connectivity error: {e}")

        # Test friend model endpoint
        friend_url = os.getenv("MODEL_ENDPOINT")
        if friend_url:
            try:
                # Send a simple health check
                async with session.get(friend_url.split('/generate')[0], timeout=5) as r:
                    results.append(f"✅ Friend model reachable: HTTP {r.status}")
            except Exception as e:
                results.append(f"❌ Friend model error: {e}")
        else:
            results.append("⚠️ MODEL_ENDPOINT not set")

    return results

# ─── Main Loop ─────────────────────────────────────────────────────────────────
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--interval", type=float, default=30)  # seconds
    p.add_argument("--check-connectivity", action="store_true", help="Check connectivity to services")
    args = p.parse_args()

    # Configuration
    base = pathlib.Path("/mnt/block/raw/icsi/audio")
    friend_url = os.getenv("MODEL_ENDPOINT")
    together_key = os.getenv("TOGETHER_API_KEY")
    max_tokens = int(os.getenv("MAX_NEW_TOKENS", "50"))

    # Validate configuration
    if not friend_url:
        print("⚠️ Warning: MODEL_ENDPOINT environment variable not set")

    if not together_key:
        print("⚠️ Warning: TOGETHER_API_KEY environment variable not set")

    print(f"▶️  Simulator: batch={args.batch_size}, interval={args.interval}s")

    # Check connectivity if requested
    if args.check_connectivity:
        print("Checking connectivity...")
        results = await check_connectivity()
        for result in results:
            print(result)
        print("Connectivity check complete")

    # Create log directory if it doesn't exist
    os.makedirs("/mnt/block", exist_ok=True)

    # Main simulation loop
    failure_count = 0
    while True:
        try:
            # Check if transcript directory exists
            if not base.exists():
                print(f"⚠️ Transcript directory not found: {base}")
                print("Creating directory structure...")
                os.makedirs(base, exist_ok=True)
                await asyncio.sleep(60)
                continue

            # Find transcript files
            files = list(base.rglob("*.txt"))
            if not files:
                print("⚠️ No transcripts found — sleeping 60s")
                await asyncio.sleep(60)
                continue

            # Process batch
            batch = random.sample(files, min(args.batch_size, len(files)))
            processed = await process_batch(batch, friend_url, together_key, max_tokens)
            print(f"✅ Processed {processed} files")

            # Reset failure counter on success
            failure_count = 0

            await asyncio.sleep(args.interval)

        except Exception as e:
            failure_count += 1
            print(f"❌ Error in main loop: {e}")

            # Backoff on repeated failures
            wait_time = min(60 * failure_count, 300)  # Max 5 minutes
            print(f"Waiting {wait_time}s before retrying...")
            await asyncio.sleep(wait_time)

if __name__ == "__main__":
    asyncio.run(main())