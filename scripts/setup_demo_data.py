"""Setup demo databases and lightweight demo Chroma index for Streamlit deployment."""

import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.fewshot_index import FewShotIndex
from app.schema_index import SchemaIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEMO_DBS = ["concert_singer", "pets_1", "car_1", "museum_visit"]
DEMO_DIR = Path("./demo_data")
DEMO_CHROMA_DIR = DEMO_DIR / "chroma"


def setup_demo_data() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    spider_db_root = Path("./data/spider/database")

    # 1. Copy sample SQLite databases
    logger.info("Copying demo databases into %s...", DEMO_DIR)
    for db_id in DEMO_DBS:
        src = spider_db_root / db_id / f"{db_id}.sqlite"
        if src.exists():
            dest = DEMO_DIR / f"{db_id}.sqlite"
            shutil.copy2(src, dest)
            logger.info("Copied %s (%.1f KB)", db_id, dest.stat().st_size / 1024.0)
        else:
            logger.warning("Source database not found: %s", src)

    # 2. Build lightweight demo Chroma index
    logger.info("Building standalone demo Chroma index in %s...", DEMO_CHROMA_DIR)
    demo_schema_idx = SchemaIndex(chroma_dir=DEMO_CHROMA_DIR)
    for db_id in DEMO_DBS:
        db_file = DEMO_DIR / f"{db_id}.sqlite"
        if db_file.exists():
            demo_schema_idx.index_database(db_id, db_path=db_file)

    # 3. Build demo few-shot examples (100 examples)
    demo_fs_idx = FewShotIndex(chroma_dir=DEMO_CHROMA_DIR)
    demo_fs_idx.build_from_train(max_samples=100)

    # 4. Save sample questions metadata for Streamlit UI
    demo_questions = {
        "concert_singer": [
            "How many singers do we have?",
            "What is the average, minimum, and maximum age of all singers?",
            "List the names of singers who have performed in more than one concert.",
            "Find the name and song release year for the oldest singer.",
            "Show the stadium names and their capacities where concerts took place in 2014.",
        ],
        "pets_1": [
            "How many students have pets?",
            "What is the average weight of pets by pet type?",
            "List the first and last names of students who own a dog.",
            "Which pet is the oldest?",
            "Show the student id and pet count for each student who has more than one pet.",
        ],
        "car_1": [
            "How many car models are there?",
            "List all car makers and their countries of origin.",
            "What is the average horsepower of cars produced after 1980?",
            "Find the model and year of the car with the highest MPG.",
            "Count how many cars were produced by American manufacturers.",
        ],
        "museum_visit": [
            "How many museums are open?",
            "List the names of all visitors who visited in 2023.",
            "What is the name of the visitor who spent the most money?",
            "Show the museum names and ticket prices ordered by price descending.",
            "Which museum was visited by both Alice and Bob?",
        ],
    }

    questions_path = DEMO_DIR / "sample_questions.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(demo_questions, f, indent=2)

    logger.info("Demo setup complete. Total demo folder size: %.2f MB",
                sum(f.stat().st_size for f in DEMO_DIR.rglob("*") if f.is_file()) / (1024 * 1024))


if __name__ == "__main__":
    setup_demo_data()

