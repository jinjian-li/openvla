#!/usr/bin/env python3
"""
Download physical-intelligence/libero dataset from HF mirror.
Uses concurrent HTTP downloads — fast, simple, auto-resume.
"""
import os, sys, time, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

MIRROR = "https://hf-mirror.com"
REPO = "physical-intelligence/libero"
DEST = Path(os.environ.get("DATASET_DIR", "/root/autodl-tmp/datasets/physical-intelligence/libero"))
WORKERS = int(os.environ.get("WORKERS", "8"))
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))

def get_file_list():
    """Fetch complete file list from HF mirror API."""
    url = f"{MIRROR}/api/datasets/{REPO}/tree/main?recursive=True"
    print(f"Fetching file list from {url}...", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    files = [f["path"] for f in resp.json()]
    print(f"Total files to download: {len(files)}", flush=True)
    return files

def download_one(path):
    """Download a single file. Returns (path, success, size_bytes)."""
    local = DEST / path
    if local.exists():
        return (path, True, local.stat().st_size, "skipped")

    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{path}"

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(local, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size = local.stat().st_size
            return (path, True, size, f"ok ({size} bytes)")
        except Exception as e:
            if attempt == 2:
                return (path, False, 0, str(e)[:80])
            time.sleep(2)
    return (path, False, 0, "unknown")

def main():
    files = get_file_list()

    # Check what's already downloaded
    existing = sum(1 for f in files if (DEST / f).exists())
    if existing > 0:
        total_size = sum((DEST / f).stat().st_size for f in files if (DEST / f).exists())
        print(f"Already have {existing}/{len(files)} files ({total_size/1e9:.1f} GB)", flush=True)

    to_download = [f for f in files if not (DEST / f).exists()]
    if not to_download:
        print("All files already downloaded!", flush=True)
        return

    print(f"Downloading {len(to_download)} files with {WORKERS} workers...", flush=True)
    start = time.time()
    done = 0
    failed = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, f): f for f in to_download}
        for future in as_completed(futures):
            path, ok, size, msg = future.result()
            done += 1
            if ok:
                total_bytes += size
            else:
                failed += 1
                print(f"  FAIL [{done}/{len(to_download)}] {path}: {msg}", flush=True)

            if done % 50 == 0 or done == len(to_download):
                elapsed = time.time() - start
                speed = total_bytes / elapsed / 1e6 if elapsed > 0 else 0
                print(f"  [{done}/{len(to_download)}] {total_bytes/1e9:.2f} GB "
                      f"({speed:.1f} MB/s) {failed} failed", flush=True)

    elapsed = time.time() - start
    print(f"\nDone: {done} files, {total_bytes/1e9:.2f} GB in {elapsed/60:.0f} min "
          f"({failed} failed)", flush=True)

    # Quick sanity check
    meta = DEST / "meta" / "info.json"
    data_files = list(DEST.glob("data/**/*.parquet"))
    print(f"Sanity: meta/info.json={'OK' if meta.exists() else 'MISSING'}, "
          f"parquet files={len(data_files)}", flush=True)

if __name__ == "__main__":
    main()
