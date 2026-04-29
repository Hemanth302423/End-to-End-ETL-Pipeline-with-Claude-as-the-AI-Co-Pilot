"""
extractors/nyc_tlc_extractor.py
--------------------------------
Extracts NYC TLC Yellow Taxi trip data from the public dataset.
Supports downloading Parquet files directly from the TLC website.

Prompt used with Claude Desktop:
  "Write a robust Python extractor for NYC TLC taxi data that supports
   downloading multiple months, validates the file after download,
   and logs progress. Include retry logic and checksum validation."
"""

import os
import logging
import hashlib
import time
from pathlib import Path
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NYC TLC open data base URL
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DEFAULT_OUTPUT_DIR = Path("data/raw")


def build_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def build_url(year: int, month: int, taxi_type: str = "yellow") -> str:
    """Build the download URL for a given year/month."""
    return f"{BASE_URL}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def download_file(
    url: str,
    output_path: Path,
    session: Optional[requests.Session] = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> Path:
    """
    Download a file with streaming and progress logging.

    Args:
        url: Source URL
        output_path: Local path to save the file
        session: Optional requests session (creates one if not provided)
        chunk_size: Bytes per chunk during streaming

    Returns:
        Path to the downloaded file
    """
    session = session or build_session()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting download: {url}")
    start = time.time()

    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        logger.debug(f"  Progress: {pct:.1f}% ({downloaded:,} / {total_size:,} bytes)")

    elapsed = time.time() - start
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Download complete: {output_path.name} ({file_size_mb:.1f} MB in {elapsed:.1f}s)")
    return output_path


def validate_parquet(filepath: Path) -> bool:
    """
    Validate downloaded Parquet file is readable and non-empty.

    Returns:
        True if valid, False otherwise
    """
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(filepath)
        row_count = meta.num_rows
        logger.info(f"Validation passed: {filepath.name} — {row_count:,} rows, {meta.num_row_groups} row groups")
        return row_count > 0
    except Exception as e:
        logger.error(f"Validation failed for {filepath.name}: {e}")
        return False


def extract(
    year: int,
    months: list[int],
    taxi_type: str = "yellow",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    skip_existing: bool = True,
) -> list[Path]:
    """
    Extract NYC TLC trip data for given year and months.

    Args:
        year: The year to extract (e.g. 2023)
        months: List of months to extract (e.g. [1, 2, 3])
        taxi_type: 'yellow', 'green', or 'fhv'
        output_dir: Directory to save raw files
        skip_existing: Skip download if file already exists

    Returns:
        List of successfully downloaded file paths
    """
    session = build_session()
    downloaded_files = []

    for month in months:
        url = build_url(year, month, taxi_type)
        filename = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
        output_path = output_dir / filename

        if skip_existing and output_path.exists():
            logger.info(f"Skipping (already exists): {filename}")
            downloaded_files.append(output_path)
            continue

        try:
            path = download_file(url, output_path, session)
            if validate_parquet(path):
                downloaded_files.append(path)
            else:
                logger.warning(f"Removing invalid file: {path}")
                path.unlink(missing_ok=True)
        except requests.HTTPError as e:
            logger.error(f"HTTP error for {filename}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {filename}: {e}", exc_info=True)

    logger.info(f"Extraction complete. {len(downloaded_files)}/{len(months)} files downloaded successfully.")
    return downloaded_files


if __name__ == "__main__":
    # Quick test: download Jan 2023
    files = extract(year=2023, months=[1], taxi_type="yellow")
    for f in files:
        checksum = compute_md5(f)
        print(f"{f.name}: MD5={checksum}")