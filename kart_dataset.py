"""
PyTorch Dataset that loads Mario Kart frame data from Supabase and produces
(image, question, answer) triplets matching the Moondream inference flow.

Supports three tasks:
  - "scene":    full frame  → "yes" / "no"
  - "position": bottom-right crop → "1"–"24" / "n/a"
  - "coins":    bottom-left crop  → "0"–"20" / "n/a"

Usage:
    ds = KartSceneDataset.from_supabase(email, password)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=ds.collate)
    for batch in loader:
        images, prompts, answers = batch["images"], batch["prompts"], batch["answers"]
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Literal

import requests
from PIL import Image
from supabase import Client, create_client
from torch.utils.data import Dataset

STORAGE_BUCKET = "frame-images"
SIGNED_URL_TTL = 60 * 60 * 24  # 24 h

SCENE_QUESTION = "Is this an active mario kart race? Response yes no or unsure"
POSITION_QUESTION = (
    "What position number (1-24) is shown? "
    "Respond with just the number or n/a if nothing is shown."
)
COINS_QUESTION = (
    "How many coins are shown? "
    "Respond with just the number or n/a if nothing is shown."
)

Task = Literal["scene", "position", "coins"]


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _connect(
    url: str | None = None,
    key: str | None = None,
    email: str | None = None,
    password: str | None = None,
) -> tuple[Client, str]:
    """Create an authenticated Supabase client; return (client, user_id)."""
    url = url or os.environ["NEXT_PUBLIC_SUPABASE_URL"]
    key = key or os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"]
    client = create_client(url, key)
    email = email or os.environ.get("SUPABASE_USER_EMAIL", "")
    password = password or os.environ.get("SUPABASE_USER_PASSWORD", "")
    if not email or not password:
        raise ValueError(
            "Provide email/password or set SUPABASE_USER_EMAIL / SUPABASE_USER_PASSWORD"
        )
    auth = client.auth.sign_in_with_password({"email": email, "password": password})
    return client, auth.user.id


def _fetch_all_sessions(client: Client, user_id: str) -> list[dict]:
    """Return all analysis_sessions rows for user, newest first."""
    return (
        client.table("analysis_sessions")
        .select("id, video_name, sample_interval, frame_annotations, race_data, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    ).data


def _sign_urls(
    client: Client,
    user_id: str,
    session_id: str,
    n_frames: int,
    kind: Literal["thumb", "hires"] = "hires",
) -> list[str]:
    """Return signed download URLs for all frames in a session."""
    paths = [f"{user_id}/{session_id}/{i}_{kind}.jpg" for i in range(n_frames)]
    if not paths:
        return []
    result = client.storage.from_(STORAGE_BUCKET).create_signed_urls(paths, SIGNED_URL_TTL)
    return [item.get("signedURL", "") or "" for item in result]


# ---------------------------------------------------------------------------
# Image download / caching
# ---------------------------------------------------------------------------

def _url_cache_key(url: str) -> str:
    """Stable filename derived from the path portion of a signed URL."""
    path = url.split("?")[0].rsplit("/", 1)[-1]
    h = hashlib.md5(url.split("?")[0].encode()).hexdigest()[:8]
    return f"{h}_{path}"


def _download_image(url: str, cache_dir: Path | None = None) -> Image.Image | None:
    if not url:
        return None
    if cache_dir:
        cached = cache_dir / _url_cache_key(url)
        if cached.exists():
            return Image.open(cached).convert("RGB")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            img.save(cached, "JPEG", quality=90)
        return img
    except Exception:
        return None


def _download_batch(
    urls: list[str],
    cache_dir: Path | None = None,
    max_workers: int = 16,
) -> list[Image.Image | None]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda u: _download_image(u, cache_dir), urls))


# ---------------------------------------------------------------------------
# Crop helpers (mirror the JS crops in analyzer-engine.ts)
# ---------------------------------------------------------------------------

def crop_bottom_right(img: Image.Image, frac: float = 0.3) -> Image.Image:
    """Bottom-right 30 % crop (used for position reading)."""
    w, h = img.size
    cw, ch = round(w * frac), round(h * frac)
    return img.crop((w - cw, h - ch, w, h))


def crop_bottom_left(img: Image.Image, frac: float = 0.3) -> Image.Image:
    """Bottom-left 30 % crop (used for coin reading)."""
    w, h = img.size
    cw, ch = round(w * frac), round(h * frac)
    return img.crop((0, h - ch, cw, h))


# ---------------------------------------------------------------------------
# Sample type
# ---------------------------------------------------------------------------

class FrameSample:
    __slots__ = ("image", "question", "answer", "session_id", "timestamp", "task")

    def __init__(
        self,
        image: Image.Image,
        question: str,
        answer: str,
        session_id: str,
        timestamp: float,
        task: Task,
    ):
        self.image = image
        self.question = question
        self.answer = answer
        self.session_id = session_id
        self.timestamp = timestamp
        self.task = task


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class KartSceneDataset(Dataset):
    """
    Each item is a dict:
        image     – PIL.Image.Image (RGB)
        question  – str  (the prompt sent to Moondream)
        answer    – str  (the expected response)
        session_id, timestamp, task – metadata
    """

    def __init__(
        self,
        samples: list[FrameSample],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = self.transform(s.image) if self.transform else s.image
        return {
            "image": img,
            "question": s.question,
            "answer": s.answer,
            "session_id": s.session_id,
            "timestamp": s.timestamp,
            "task": s.task,
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """Custom collate that keeps PIL images as a list (no stacking)."""
        return {
            "images": [b["image"] for b in batch],
            "questions": [b["question"] for b in batch],
            "answers": [b["answer"] for b in batch],
            "session_ids": [b["session_id"] for b in batch],
            "timestamps": [b["timestamp"] for b in batch],
            "tasks": [b["task"] for b in batch],
        }

    # ------------------------------------------------------------------
    # Factory: build dataset straight from Supabase
    # ------------------------------------------------------------------
    @classmethod
    def from_supabase(
        cls,
        email: str | None = None,
        password: str | None = None,
        *,
        url: str | None = None,
        key: str | None = None,
        tasks: list[Task] | Task = "scene",
        image_kind: Literal["thumb", "hires"] = "hires",
        cache_dir: str | Path | None = "./frame_cache",
        session_ids: list[str] | None = None,
        skip_unlabeled: bool = True,
        transform=None,
        verbose: bool = True,
    ) -> "KartSceneDataset":
        """
        Download sessions from Supabase and build the dataset.

        Parameters
        ----------
        email, password : Supabase auth credentials.
        url, key        : Override NEXT_PUBLIC_SUPABASE_URL / _ANON_KEY.
        tasks           : Which task(s) to include — "scene", "position", "coins", or a list.
        image_kind      : "hires" (768 px) or "thumb".
        cache_dir       : Local directory to cache downloaded images (None to disable).
        session_ids     : Limit to specific session UUIDs (None = all).
        skip_unlabeled  : Drop frames whose label is None for the requested task.
        transform       : Optional torchvision transform applied to images.
        verbose         : Print progress.
        """
        if isinstance(tasks, str):
            tasks = [tasks]
        cache_path = Path(cache_dir) if cache_dir else None

        client, user_id = _connect(url, key, email, password)
        if verbose:
            print(f"Authenticated as {user_id[:8]}…")

        all_sessions = _fetch_all_sessions(client, user_id)
        if session_ids:
            sid_set = set(session_ids)
            all_sessions = [s for s in all_sessions if s["id"] in sid_set]
        if verbose:
            print(f"Loading {len(all_sessions)} session(s)…")

        samples: list[FrameSample] = []

        for sess in all_sessions:
            sid = sess["id"]
            frames_meta = sess["frame_annotations"].get("frames", [])
            n = len(frames_meta)
            if n == 0:
                continue

            if verbose:
                print(f"  {sess['video_name']}: {n} frames — downloading {image_kind} images…")

            urls = _sign_urls(client, user_id, sid, n, kind=image_kind)
            imgs = _download_batch(urls, cache_dir=cache_path)

            for i, (meta, img) in enumerate(zip(frames_meta, imgs)):
                if img is None:
                    continue

                ts = meta["timestamp"]
                scene_label = meta.get("scene")
                pos_label = meta.get("position")
                coin_label = meta.get("coins")

                if "scene" in tasks:
                    if scene_label is not None or not skip_unlabeled:
                        answer = "yes" if scene_label == "in_race" else "no"
                        samples.append(
                            FrameSample(img, SCENE_QUESTION, answer, sid, ts, "scene")
                        )

                if "position" in tasks and scene_label == "in_race":
                    if pos_label is not None or not skip_unlabeled:
                        cropped = crop_bottom_right(img)
                        if pos_label is not None and pos_label != "x":
                            ans = str(int(pos_label))
                        else:
                            ans = "n/a"
                        samples.append(
                            FrameSample(cropped, POSITION_QUESTION, ans, sid, ts, "position")
                        )

                if "coins" in tasks and scene_label == "in_race":
                    if coin_label is not None or not skip_unlabeled:
                        cropped = crop_bottom_left(img)
                        ans = str(int(coin_label)) if coin_label is not None else "n/a"
                        samples.append(
                            FrameSample(cropped, COINS_QUESTION, ans, sid, ts, "coins")
                        )

        if verbose:
            task_counts = {}
            for s in samples:
                task_counts[s.task] = task_counts.get(s.task, 0) + 1
            print(f"Dataset ready: {len(samples)} samples {task_counts}")

        return cls(samples, transform=transform)
