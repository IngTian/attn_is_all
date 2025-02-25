import sys
import logging
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create log file with timestamp
LOG_FILE = LOG_DIR / f"paper_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure default logger
logger = logging.getLogger("paper_transformer")
logger.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
