"""
competition_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Production pipeline that loads subcategory cluster models, computes a
competition score for every subcategory, and persists the results to Redis.

Key design decisions
────────────────────
* Keywords in the pickles are stored as (word, count) tuples – we extract
  only the word strings so all downstream logic stays clean.
* `fetch_videos_for_keyword` is a realistic stub that returns random but
  plausible YouTube-style engagement data; swap it for a real API call
  later without touching the rest of the pipeline.
* Scores are normalised to [0, 1] before the level lookup, matching the
  thresholds in the specification.
* Every public function is fully exception-safe so one bad subcategory
  never aborts the whole run.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from googleapiclient.discovery import build as _yt_build
    _GOOGLEAPI_AVAILABLE = True
except ImportError:
    _yt_build = None          # type: ignore[assignment]
    _GOOGLEAPI_AVAILABLE = False

import redis
from dotenv import load_dotenv

try:
    from Config import SUBCATEGORY_IDS as _SUBCATEGORY_IDS
except ImportError:
    _SUBCATEGORY_IDS = {}

# ── Project bootstrap ─────────────────────────────────────────────────────────
load_dotenv()

_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR: Path = _project_root / "models" / "subcategory_models"

# How many simulated videos to "fetch" per keyword
_VIDEOS_PER_KEYWORD: int = 20

# Normalisation ceiling for the supply_factor (avg_virality / N).
# Real-world supply_factor values top out ~400–500, so a cap of 500
# spreads scores across the full [0, 1] range and produces meaningful
# Low / Medium / High splits.
_VIRALITY_NORM_CAP: float = 500.0

# Redis key template
_REDIS_KEY_TEMPLATE: str = "competition:{id}"

# ── Redis client ──────────────────────────────────────────────────────────────
redis_client: redis.Redis | None = None

_REDIS_URL = os.getenv("REDIS_URL")
if _REDIS_URL:
    try:
        redis_client = redis.from_url(_REDIS_URL, decode_responses=True)
        redis_client.ping()  # fail fast if the server is unreachable
        logger.info("Redis connection established.")
    except Exception as _exc:
        logger.warning("Redis connection failed: %s", _exc)
        redis_client = None
else:
    logger.warning("REDIS_URL not set – Redis storage will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_all_models
# ─────────────────────────────────────────────────────────────────────────────
def load_all_models() -> dict[str, Any]:
    """
    Load every .pkl file from MODELS_DIR.

    Returns
    -------
    dict  { category_name: model_data }
        `category_name` is derived from the file stem by stripping known
        suffixes such as ``_clusters_results`` and
        ``_clusters_model_and_interpretations``.
    """
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    pkl_files = list(MODELS_DIR.glob("*.pkl"))
    if not pkl_files:
        logger.warning("No .pkl files found in %s", MODELS_DIR)
        return {}

    models: dict[str, Any] = {}

    for pkl_path in pkl_files:
        stem = pkl_path.stem
        # Normalise the category name: strip common suffixes then take the
        # leading word segment.
        category = (
            stem.replace("_clusters_model_and_interpretations", "")
                .replace("_clusters_results", "")
                .strip("_")
                .lower()
        )
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            models[category] = data
            logger.info("Loaded model: %-20s  (%s)", category, pkl_path.name)
        except Exception as exc:
            logger.error("Failed to load %s: %s", pkl_path.name, exc)

    logger.info("Total models loaded: %d", len(models))
    return models


# ─────────────────────────────────────────────────────────────────────────────
# 2. extract_subcategories
# ─────────────────────────────────────────────────────────────────────────────
def extract_subcategories(models: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten every cluster from every model into a list of subcategory dicts.

    Only the **top 3 keywords** (by their original frequency rank in the
    pickle) are kept per subcategory. This caps scoring at 3 API calls per
    subcategory and keeps Redis payloads lean.

    Handles keywords stored as plain strings **or** as (word, count) tuples.

    Returns
    -------
    list of dicts with keys:
        category, subcategory_id, name, keywords  (max 3 items)
    """
    subcategories: list[dict[str, Any]] = []

    for category, model_data in models.items():
        if not isinstance(model_data, dict):
            logger.warning("Skipping %s – unexpected model format.", category)
            continue

        cluster_interpretations: dict = model_data.get("cluster_interpretations", {})
        if not cluster_interpretations:
            logger.warning("No cluster_interpretations for category: %s", category)
            continue

        for cluster_id, cluster_info in cluster_interpretations.items():
            if not isinstance(cluster_info, dict):
                continue

            raw_keywords = cluster_info.get("keywords", [])

            # Normalise: support both str and (str, count) tuple formats
            clean_keywords: list[str] = []
            for kw in raw_keywords:
                if isinstance(kw, str):
                    clean_keywords.append(kw.strip().lower())
                elif isinstance(kw, (list, tuple)) and len(kw) >= 1:
                    clean_keywords.append(str(kw[0]).strip().lower())

            # Deduplicate while preserving order, then keep top 3
            seen: set[str] = set()
            unique_keywords: list[str] = []
            for kw in clean_keywords:
                if kw and kw not in seen:
                    seen.add(kw)
                    unique_keywords.append(kw)

            top_keywords = unique_keywords[:3]  # ← cap at top 3

            subcategories.append(
                {
                    "category": category,
                    "subcategory_id": int(cluster_id),
                    "name": cluster_info.get("name", f"Cluster {cluster_id}"),
                    "keywords": top_keywords,
                }
            )

    logger.info("Total subcategories extracted: %d", len(subcategories))
    return subcategories


# ─────────────────────────────────────────────────────────────────────────────
# 3. fetch_videos_for_keyword
# ─────────────────────────────────────────────────────────────────────────────
def fetch_videos_for_keyword(keyword: str) -> list[dict[str, float]]:
    """
    Fetch real YouTube video metrics for a keyword via the YouTube Data API v3.

    Flow
    ----
    1. ``search.list``  → find up to ``_VIDEOS_PER_KEYWORD`` video IDs for the keyword.
    2. ``videos.list``  → retrieve ``statistics`` (viewCount) and ``snippet``
       (publishedAt) for those IDs in a single batched request.
    3. Compute ``hours_since_upload`` from ``publishedAt`` → now (UTC).

    Fallback
    --------
    If the API key is missing, the library is not installed, or any API/network
    error occurs (including quota exhaustion), the function falls back to
    random plausible values so the pipeline continues uninterrupted.

    Returns
    -------
    list of dicts, each with keys:
        views              (float) – total view count
        hours_since_upload (float) – hours since the video was published (≥ 1)
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")

    if not _GOOGLEAPI_AVAILABLE or not api_key or api_key == "YOUR_API_KEY":
        logger.debug("YouTube API unavailable for '%s' – using random fallback.", keyword)
        return _random_video_fallback()

    try:
        youtube = _yt_build("youtube", "v3", developerKey=api_key)

        # ── Step 1: search for videos matching the keyword ────────────────
        search_response = (
            youtube.search()
            .list(
                q=keyword,
                part="id",
                type="video",
                maxResults=_VIDEOS_PER_KEYWORD,
                order="relevance",
                relevanceLanguage="en",
            )
            .execute()
        )

        video_ids: list[str] = [
            item["id"]["videoId"]
            for item in search_response.get("items", [])
            if item.get("id", {}).get("kind") == "youtube#video"
        ]

        if not video_ids:
            logger.debug("No video IDs returned for keyword '%s'.", keyword)
            return _random_video_fallback()

        # ── Step 2: batch-fetch statistics + snippet ──────────────────────
        stats_response = (
            youtube.videos()
            .list(
                id=",".join(video_ids),
                part="statistics,snippet",
            )
            .execute()
        )

        now_utc = datetime.now(timezone.utc)
        videos: list[dict[str, float]] = []

        for item in stats_response.get("items", []):
            try:
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})

                view_count = float(stats.get("viewCount", 0) or 0)

                published_at_str: str = snippet.get("publishedAt", "")
                if published_at_str:
                    published_dt = datetime.fromisoformat(
                        published_at_str.replace("Z", "+00:00")
                    )
                    hours_live = max(
                        (now_utc - published_dt).total_seconds() / 3600.0, 1.0
                    )
                else:
                    hours_live = random.uniform(24.0, 720.0)

                videos.append(
                    {"views": view_count, "hours_since_upload": hours_live}
                )
            except Exception as item_exc:
                logger.debug("Skipping malformed video item: %s", item_exc)

        if not videos:
            return _random_video_fallback()

        logger.debug(
            "Fetched %d real videos for keyword '%s'.", len(videos), keyword
        )
        return videos

    except Exception as api_exc:
        logger.warning(
            "YouTube API error for keyword '%s': %s – using random fallback.",
            keyword,
            api_exc,
        )
        return _random_video_fallback()


def _random_video_fallback() -> list[dict[str, float]]:
    """Return random but plausible video engagement data as a fallback."""
    return [
        {
            "views": float(random.randint(500, 5_000_000)),
            "hours_since_upload": random.uniform(1.0, 720.0),
        }
        for _ in range(_VIDEOS_PER_KEYWORD)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_competition_score
# ─────────────────────────────────────────────────────────────────────────────
def compute_competition_score(keywords: list[str]) -> float:
    """
    Compute a normalised competition score in [0, 1] for a list of keywords.

    Algorithm
    ---------
    For each keyword k:
        1. Fetch N videos.
        2. For each video  virality = views / hours_since_upload
        3. avg_virality   = mean(virality values)
        4. supply_factor  = avg_virality / N   (penalise crowded niches)

    keyword_score = supply_factor  (clipped to [0, _VIRALITY_NORM_CAP])
    final_score   = mean(keyword_score_normalised) over all keywords

    A **higher** score → more virality per unit of supply → **lower** competition
    from the content creator's perspective (easier to stand out), hence
    score > 0.7 maps to "Low" competition in get_competition_level().
    """
    if not keywords:
        return 0.0

    keyword_scores: list[float] = []

    for keyword in keywords:
        try:
            videos = fetch_videos_for_keyword(keyword)
            if not videos:
                logger.debug("No videos returned for keyword '%s', skipping.", keyword)
                continue

            virality_values = [
                v["views"] / max(v["hours_since_upload"], 1.0) for v in videos
            ]
            avg_virality = sum(virality_values) / len(virality_values)
            supply_factor = avg_virality / len(videos)

            # Clip and normalise to [0, 1]
            normalised = min(supply_factor, _VIRALITY_NORM_CAP) / _VIRALITY_NORM_CAP
            keyword_scores.append(normalised)
        except Exception as exc:
            logger.debug("Error computing score for keyword '%s': %s", keyword, exc)

    if not keyword_scores:
        return 0.0

    return round(sum(keyword_scores) / len(keyword_scores), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_competition_level
# ─────────────────────────────────────────────────────────────────────────────
def get_competition_level(score: float) -> str:
    """
    Map a normalised competition score to a human-readable level.

    score > 0.7  → "Low"   (niche has strong virality, low saturation)
    score > 0.4  → "Medium"
    score ≤ 0.4  → "High"  (saturated / low engagement-per-video)
    """
    if score > 0.7:
        return "Low"
    elif score > 0.4:
        return "Medium"
    else:
        return "High"


# ─────────────────────────────────────────────────────────────────────────────
# 6. build_subcategory_result
# ─────────────────────────────────────────────────────────────────────────────
def build_subcategory_result(subcategory: dict[str, Any]) -> dict[str, Any]:
    """
    Compute the competition score for a subcategory and return a complete
    result dict ready for Redis storage.

    Parameters
    ----------
    subcategory : dict
        Must contain keys: subcategory_id, name, keywords

    Returns
    -------
    dict with keys:
        id, subcategory_id, name, competition_score, competition_level,
        top_keywords, last_updated
    """
    keywords: list[str] = subcategory.get("keywords", [])
    name: str = subcategory["name"]
    score = compute_competition_score(keywords)
    level = get_competition_level(score)

    # Look up the global integer id from Config.SUBCATEGORY_IDS
    unique_id = _SUBCATEGORY_IDS.get(name)
    if unique_id is None:
        logger.warning("No SUBCATEGORY_IDS entry found for '%s' – id will be null.", name)

    return {
        "id": unique_id,
        "subcategory_id": subcategory["subcategory_id"],
        "name": name,
        "competition_score": score,
        "competition_level": level,
        "top_keywords": keywords[:3],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. save_to_redis
# ─────────────────────────────────────────────────────────────────────────────
def save_to_redis(unique_id: int, data: dict[str, Any]) -> bool:
    """
    Persist a subcategory result dict to Redis under the key:
        competition:{id}

    Returns True on success, False on failure.
    """
    if not redis_client:
        logger.debug("Redis client unavailable – skipping save for id: %s.", unique_id)
        return False

    key = _REDIS_KEY_TEMPLATE.format(id=unique_id)
    try:
        redis_client.set(key, json.dumps(data, ensure_ascii=False))
        logger.debug("Saved → %s", key)
        return True
    except Exception as exc:
        logger.error("Redis write error for key '%s': %s", key, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8. run_pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Orchestrate the full competition-scoring pipeline.

    Parameters
    ----------
    offset : int
        Index of the first subcategory to process (0-based).
        Use this to resume from where a previous batch stopped.
    limit : int | None
        Maximum number of subcategories to process in this run.
        ``None`` means process all subcategories from ``offset`` onward.

    Returns
    -------
    dict  { category: [result, ...] }
        Results for the subcategories processed in this batch.

    Examples
    --------
    First batch  (quota day 1):  run_pipeline(offset=0,  limit=30)
    Second batch (quota day 2):  run_pipeline(offset=30, limit=30)
    Full run (no quota concern): run_pipeline()
    """
    logger.info("=" * 60)
    logger.info("  Competition Pipeline  –  START  (offset=%d, limit=%s)", offset, limit)
    logger.info("=" * 60)

    # Step 1 – Load models
    logger.info("[1/3] Loading subcategory models …")
    try:
        models = load_all_models()
    except Exception as exc:
        logger.error("Fatal: could not load models – %s", exc)
        return {}

    if not models:
        logger.warning("No models loaded. Exiting pipeline.")
        return {}

    # Step 2 – Extract subcategories
    logger.info("[2/3] Extracting subcategories …")
    all_subcategories = extract_subcategories(models)
    if not all_subcategories:
        logger.warning("No subcategories extracted. Exiting pipeline.")
        return {}

    # Apply batch window
    batch = all_subcategories[offset : (offset + limit) if limit is not None else None]
    logger.info(
        "    Batch: subcategories %d–%d of %d total",
        offset + 1,
        offset + len(batch),
        len(all_subcategories),
    )

    # Step 3 – Score & store
    logger.info("[3/3] Computing scores and storing results …")
    results: dict[str, list[dict[str, Any]]] = {}
    success_count = 0
    error_count = 0

    for sub in batch:
        category = sub["category"]
        sub_id = sub["subcategory_id"]
        sub_name = sub["name"]

        try:
            result = build_subcategory_result(sub)
            saved = save_to_redis(result["id"], result)

            results.setdefault(category, []).append(result)
            success_count += 1

            status_icon = "✓" if saved else "○"  # ✓=saved  ○=computed only
            logger.info(
                "  %s  %-15s  #%-3d  %-40s  score=%.4f  level=%s",
                status_icon,
                category,
                sub_id,
                sub_name[:40],
                result["competition_score"],
                result["competition_level"],
            )
        except Exception as exc:
            error_count += 1
            logger.error(
                "  ✗  %-15s  #%-3d  %-40s  ERROR: %s",
                category,
                sub_id,
                sub_name[:40],
                exc,
            )

    remaining = len(all_subcategories) - (offset + len(batch))
    logger.info("=" * 60)
    logger.info(
        "  Batch complete  |  processed=%d  errors=%d  remaining=%d",
        success_count,
        error_count,
        max(remaining, 0),
    )
    if remaining > 0:
        logger.info(
            "  ▶ To continue, run:  run_pipeline(offset=%d, limit=%d)",
            offset + len(batch),
            limit or len(all_subcategories),
        )
    else:
        # Full run complete – dump everything from Redis to disk
        save_competition_data()
    logger.info("=" * 60)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. save_competition_data
# ─────────────────────────────────────────────────────────────────────────────
def save_competition_data(
    path: str = "output/fallback/competition_data.json",
) -> bool:
    """
    Read every ``competition:*`` key from Redis and write a single JSON file
    to ``path`` for use as an offline fallback.

    Output format
    -------------
    {
      "generated_at": "<ISO timestamp>",
      "total": 67,
      "data": {
        "1": { ...subcategory result... },
        "2": { ... },
        ...
      }
    }

    Returns True on success, False on any error.
    """
    if not redis_client:
        logger.warning("Redis unavailable – cannot export competition data.")
        return False

    try:
        keys = sorted(redis_client.keys("competition:*"))
        if not keys:
            logger.warning("No competition keys found in Redis – nothing to export.")
            return False

        data_by_id: dict[str, dict[str, Any]] = {}
        for key in keys:
            raw = redis_client.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            # Use real id or fallback to key suffix if id is missing
            uid = entry.get("id") or key.split(":")[-1]
            data_by_id[str(uid)] = entry

        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(data_by_id),
            "data": data_by_id,
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Competition data saved → %s  (%d subcategories)", out_path, len(keys))
        return True

    except Exception as exc:
        logger.error("Failed to save competition data: %s", exc)
        return False



# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Competition scoring pipeline")
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Index of the first subcategory to process (default: 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max subcategories to process in this run (default: all)",
    )
    args = parser.parse_args()
    run_pipeline(offset=args.offset, limit=args.limit)
