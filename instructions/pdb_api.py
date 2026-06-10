#!/usr/bin/env python3
"""
Fast parallel PDB downloader with live progress bar.

- Reads 4-letter PDB IDs (one per line) from PDB_ID_FILE
- Downloads .pdb.gz from RCSB, decompresses locally, writes as .pdb
- Skips IDs whose .pdb already exists
- Concurrency, HTTP/2 (if available), keep-alive, retries with backoff
- Live single-line progress bar using tqdm
"""

import asyncio
import gzip
import io
import sys
from pathlib import Path
from typing import Dict

import httpx  # pip install "httpx[http2]"
from tqdm import tqdm  # pip install tqdm

# --- Configuration ---
PDB_ID_FILE = Path("Boltz_Data_Files/pdb_ids_with_glycans.txt")
OUTPUT_DIR = Path("PDB_Downloads")
DOWNLOAD_URL_BASE = "https://files.rcsb.org/download"  # we'll fetch {ID}.pdb.gz
CONCURRENCY = 12          # try 8–16; lower if you see throttling
TIMEOUT_S = 30
MAX_RETRIES = 4
# --- End of Configuration ---

# Optional HTTP/2 auto-detect (falls back to HTTP/1.1 if h2 not installed)
try:
    import h2  # type: ignore  # noqa: F401
    HTTP2 = True
except Exception:
    HTTP2 = False

def _dedupe_preserve_order(ids):
    seen = set()
    out = []
    for x in ids:
        x = x.strip().upper()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _decompress_gzip(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        return f.read()

async def fetch_one(pdb_id: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> str:
    """
    Returns one of: 'downloaded', 'skipped', 'not_found', 'error'
    """
    out_path = OUTPUT_DIR / f"{pdb_id}.pdb"
    if out_path.exists():
        return "skipped"

    url_gz = f"{DOWNLOAD_URL_BASE}/{pdb_id}.pdb.gz"
    url_pdb = f"{DOWNLOAD_URL_BASE}/{pdb_id}.pdb"  # fallback if .gz missing

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                r = await client.get(url_gz, timeout=TIMEOUT_S)
            if r.status_code == 200:
                try:
                    content = _decompress_gzip(r.content)
                except OSError:
                    # rare: server sent plain .pdb with .gz name
                    content = r.content
                out_path.write_bytes(content)
                return "downloaded"

            if r.status_code == 404:
                # try uncompressed .pdb as a fallback
                async with sem:
                    r2 = await client.get(url_pdb, timeout=TIMEOUT_S)
                if r2.status_code == 200:
                    out_path.write_bytes(r2.content)
                    return "downloaded"
                elif r2.status_code == 404:
                    return "not_found"
                else:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue

            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(min(2 ** attempt, 8))
                continue

            return "error"

        except httpx.TimeoutException:
            await asyncio.sleep(min(2 ** attempt, 8))
        except httpx.RequestError:
            await asyncio.sleep(min(2 ** attempt, 8))
        except Exception:
            return "error"

    return "error"

async def main_async() -> None:
    if not PDB_ID_FILE.is_file():
        print(f"Error: Input file not found: {PDB_ID_FILE.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloads will be saved to: {OUTPUT_DIR.resolve()}")

    with open(PDB_ID_FILE, "r") as f:
        pdb_ids = _dedupe_preserve_order(f.readlines())

    if not pdb_ids:
        print("The input file is empty. Nothing to do.")
        return

    total = len(pdb_ids)
    print(f"Found {total} unique PDB IDs.")

    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)

    counts: Dict[str, int] = {"downloaded": 0, "skipped": 0, "not_found": 0, "error": 0}

    async with httpx.AsyncClient(http2=HTTP2, limits=limits, follow_redirects=True, headers={
        "User-Agent": "fast-pdb-downloader/1.1 (+https://rcsb.org)"
    }) as client:

        # Create tasks up front; iterate completions to drive the progress bar
        tasks = [asyncio.create_task(fetch_one(pid, client, sem)) for pid in pdb_ids]

        with tqdm(total=total, unit="file", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            pbar.set_description("Downloading")
            for fut in asyncio.as_completed(tasks):
                status = await fut
                counts[status] = counts.get(status, 0) + 1
                pbar.update(1)
                # show rolling status in the same line
                pbar.set_postfix(dl=counts["downloaded"], skip=counts["skipped"],
                                 not_found=counts["not_found"], err=counts["error"])

    print("\n========== Summary ==========")
    print(f"Total processed: {sum(counts.values())}")
    for k in ("downloaded", "skipped", "not_found", "error"):
        print(f"  {k:>12}: {counts[k]}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted.")

if __name__ == "__main__":
    main()
