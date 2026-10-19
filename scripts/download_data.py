"""Data download script for Spider and BIRD benchmarks."""

import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data").resolve()
SPIDER_DIR = DATA_DIR / "spider"
BIRD_DIR = DATA_DIR / "bird"

# Official download URLs
SPIDER_URL = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_MINI_DEV_JSON_URL = "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite-00000-of-00001.json"


def download_file_with_progress(url: str, dest_path: Path, description: str) -> None:
    """Download a remote file with a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".download")

    logger.info("Downloading %s from %s...", description, url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with requests.get(url, stream=True, headers=headers, timeout=60) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f, tqdm(
            desc=description,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    shutil.move(str(temp_path), str(dest_path))
    logger.info("Successfully downloaded %s (%.2f MB)", description, dest_path.stat().st_size / (1024 * 1024))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive with progress reporting, ignoring __MACOSX metadata."""
    logger.info("Extracting %s to %s...", zip_path.name, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if "__MACOSX" not in m.filename]
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="files"):
            zf.extract(member, extract_to)
    logger.info("Extraction complete for %s", zip_path.name)


def setup_spider() -> None:
    """Download and unpack Spider dataset."""
    dev_json = SPIDER_DIR / "dev.json"
    tables_json = SPIDER_DIR / "tables.json"
    db_dir = SPIDER_DIR / "database"

    if dev_json.exists() and tables_json.exists() and db_dir.exists():
        logger.info("Spider dataset already present at %s. Skipping download.", SPIDER_DIR)
        return

    SPIDER_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = DATA_DIR / "spider.zip"

    try:
        if not zip_dest.exists() or zip_dest.stat().st_size < 100_000_000:
            download_file_with_progress(SPIDER_URL, zip_dest, "Spider Dataset (~205 MB)")

        extract_zip(zip_dest, DATA_DIR)

        # Handle various zip root folder names (spider_data, spider, etc.)
        for candidate_name in ["spider_data", "spider"]:
            candidate_dir = DATA_DIR / candidate_name
            if candidate_dir.exists() and candidate_dir != SPIDER_DIR:
                for item in candidate_dir.iterdir():
                    target = SPIDER_DIR / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))
                shutil.rmtree(str(candidate_dir), ignore_errors=True)

        logger.info("Spider verified: dev.json=%s, tables.json=%s, databases=%d",
                    dev_json.exists(), tables_json.exists(), len(list(db_dir.glob("*"))))
    except Exception as exc:
        logger.error("Spider download/extraction failed: %s", exc)
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED FOR SPIDER")
        print("1. Download 'spider.zip' from: https://yale-lily.github.io/spider")
        print(f"2. Extract into: {SPIDER_DIR}")
        print("Expected structure:")
        print("  data/spider/dev.json")
        print("  data/spider/train_spider.json")
        print("  data/spider/tables.json")
        print("  data/spider/database/<db_id>/<db_id>.sqlite")
        print("=" * 60 + "\n")
        raise


def setup_bird() -> None:
    """Download and unpack BIRD mini-dev dataset."""
    mini_dev_json = BIRD_DIR / "mini_dev_sqlite.json"
    db_dir = BIRD_DIR / "dev_databases"

    if mini_dev_json.exists() and db_dir.exists():
        logger.info("BIRD dataset already present at %s. Skipping download.", BIRD_DIR)
        return

    BIRD_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = DATA_DIR / "bird_dev.zip"

    try:
        # 1. Download dev.zip (databases and descriptions)
        if not zip_dest.exists() or zip_dest.stat().st_size < 100_000_000:
            download_file_with_progress(BIRD_DEV_URL, zip_dest, "BIRD Dev Databases (~346 MB)")

        extract_zip(zip_dest, BIRD_DIR)

        # 2. Download mini_dev_sqlite.json
        if not mini_dev_json.exists():
            download_file_with_progress(
                BIRD_MINI_DEV_JSON_URL, mini_dev_json, "BIRD Mini-Dev JSON (~1.5 MB)"
            )

        # Check for nested dev_databases.zip inside dev_20240627 or BIRD_DIR
        nested_zips = list(BIRD_DIR.glob("**/dev_databases.zip"))
        if nested_zips:
            extract_zip(nested_zips[0], BIRD_DIR)

        # Ensure dev_databases directory is at BIRD_DIR / dev_databases
        candidates = list(BIRD_DIR.glob("**/dev_databases"))
        for cand in candidates:
            if cand != db_dir and cand.is_dir():
                if db_dir.exists():
                    shutil.rmtree(str(db_dir))
                shutil.move(str(cand), str(db_dir))
                break

        num_dbs = len(list(db_dir.glob("*"))) if db_dir.exists() else 0
        logger.info("BIRD verified: mini_dev_sqlite.json=%s, databases=%d",
                    mini_dev_json.exists(), num_dbs)
    except Exception as exc:
        logger.error("BIRD download/extraction failed: %s", exc)
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED FOR BIRD")
        print("1. Download 'dev.zip' from: https://bird-bench.github.io")
        print("2. Extract databases into: data/bird/dev_databases/")
        print("3. Download mini_dev_sqlite.json into: data/bird/mini_dev_sqlite.json")
        print("Expected structure:")
        print("  data/bird/mini_dev_sqlite.json")
        print("  data/bird/dev_databases/<db_id>/<db_id>.sqlite")
        print("  data/bird/dev_databases/<db_id>/database_description/*.csv")
        print("=" * 60 + "\n")
        raise


def main() -> None:
    logger.info("Initializing dataset downloads into %s...", DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_spider()
    setup_bird()
    logger.info("All datasets downloaded and extracted successfully.")


if __name__ == "__main__":
    main()

