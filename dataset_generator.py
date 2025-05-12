
import os
import shutil
import logging
import zipfile
import uuid
import json
from PIL import Image, UnidentifiedImageError
import cv2
import imagehash
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from icrawler import ImageDownloader
import random

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MAX_FETCH_ATTEMPTS = 5         # retries per category
MIN_SIZE          = (200, 200)   # minimum width, height
PHASH_MAX_DIST    = 10           # hamming distance threshold for duplicate images
JPEG_QUALITY      = 90           # JPEG output quality

HISTORY_DIR = "data_sessions"

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal state for duplicate-URL prevention
# -----------------------------------------------------------------------------
session_urls = {}
current_session = None

class RecordingDownloader(ImageDownloader):
    """Skip any URL we've already downloaded in this session."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls = session_urls.setdefault(current_session, set())

    def download(self, task, default_ext, timeout=5, **kwargs):
        url = task.get('file_url')
        if url in self.urls:
            return False
        ok = super().download(task, default_ext, timeout=timeout, **kwargs)
        if ok and url:
            self.urls.add(url)
        return ok

# -----------------------------------------------------------------------------
# Check image size & sharpness
# -----------------------------------------------------------------------------
def is_valid_image(path):
    img = cv2.imread(path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= 35

# -----------------------------------------------------------------------------
# Crawl keywords to fill exactly `qty` images per category
# -----------------------------------------------------------------------------
def generate_text_dataset(categories, reset, base_dir, session_id, progress_callback=None):
    """
    categories: [ { "name": str, "qty": int }, ... ]
    reset:      wipe base_dir if True
    base_dir:   e.g. data_sessions/<user>/<pid>
    session_id: for dedupe across crawls
    progress_callback(cat, fetched, total)
    """
    global current_session
    current_session = session_id
    session_urls[session_id] = set()

    # prepare directories
    if reset and os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    results = {}
    for cat in categories:
        name, total = cat["name"], cat["qty"]
        safe = name.replace(" ", "_")
        img_dir = os.path.join(base_dir, "images", safe)
        raw_dir = os.path.join(base_dir, "raw",    safe)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        fetched = len(os.listdir(img_dir))
        logger.info(f"[{name}] starting with {fetched}/{total}")
        attempt = 0

        while fetched < total+1 and attempt < MAX_FETCH_ATTEMPTS:
            attempt += 1
            need = total - fetched
            shutil.rmtree(raw_dir)
            os.makedirs(raw_dir, exist_ok=True)
            # crawl up to `need` new images
            for C in (GoogleImageCrawler, BingImageCrawler):
                crawler = C(
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=4,
                    storage={"root_dir": raw_dir},
                    downloader_cls=RecordingDownloader
                )
                crawler.crawl(keyword=name, max_num=need, min_size=MIN_SIZE)

            # filter & move
            for fn in os.listdir(raw_dir):
                if fetched == total:
                    break

                src = os.path.join(raw_dir, fn)
                if not is_valid_image(src):
                    continue

                dst = os.path.join(img_dir, fn)
                if os.path.exists(dst):
                    continue

                # save as JPEG to normalize
                try:
                    with Image.open(src) as im:
                        rgb = im.convert("RGB")
                        rgb.save(dst, format="JPEG", quality=JPEG_QUALITY)
                except (UnidentifiedImageError, OSError) as e:
                    logger.warning(f"Skipping {src}: {e}")
                    continue

                fetched += 1
                if progress_callback:
                    progress_callback(name, fetched, total)

            logger.info(f"[{name}] attempt {attempt}: now {fetched}/{total}")

        results[name] = fetched
    shutil.rmtree(raw_dir)
    return results

# -----------------------------------------------------------------------------
# Handle user-uploaded images (dedupe by perceptual hash)
# -----------------------------------------------------------------------------
def generate_image_dataset(file_paths, num_images, base_dir, progress_callback=None):
    """
    file_paths: local paths
    num_images: desired count
    base_dir:   session workspace
    """
    img_dir = os.path.join(base_dir, "images", "uploaded")
    os.makedirs(img_dir, exist_ok=True)

    phashes = set()
    kept = 0

    for path in file_paths:
        if kept >= num_images:
            break
        if not is_valid_image(path):
            continue

        try:
            with Image.open(path) as im:
                ph = imagehash.phash(im.convert("RGB"))
        except Exception:
            continue

        if any(abs(ph - o) < PHASH_MAX_DIST for o in phashes):
            continue

        phashes.add(ph)
        dst = os.path.join(img_dir, f"upload_{kept}.jpg")
        shutil.copy(path, dst)
        kept += 1

        if progress_callback:
            progress_callback("uploaded", kept, num_images)

    return kept

# -----------------------------------------------------------------------------
# Persist or retrieve user history
# -----------------------------------------------------------------------------
def get_history(username, entry=None):
    user_dir = os.path.join(HISTORY_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    hfile = os.path.join(user_dir, "history.json")

    hist = []
    if os.path.exists(hfile):
        with open(hfile, "r") as f:
            hist = json.load(f)

    if entry:
        hist.append(entry)
        with open(hfile, "w") as f:
            json.dump(hist, f, indent=2)

    return hist

# -----------------------------------------------------------------------------
# Zip up all images for download
# -----------------------------------------------------------------------------
def create_zip(username, pid):
    session_dir = os.path.join(HISTORY_DIR, username, pid)
    zip_path = os.path.join(session_dir, f"{pid}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        root = os.path.join(session_dir, "images")
        for dirpath, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(dirpath, fn)
                rel  = os.path.relpath(full, session_dir)
                zf.write(full, rel)
    return zip_path

# -----------------------------------------------------------------------------
# Generate a unique process ID
# -----------------------------------------------------------------------------
def make_pid():
    # Generate a human-readable PID: "RKGENDATA" followed by 5 random digits (e.g., "RKGENDATA-48392")
    suffix = "".join(random.choices("0123456789", k=5))
    return f"RKGENDATA-{suffix}"
