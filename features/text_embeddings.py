from sentence_transformers import SentenceTransformer
from typing import Any, Union, List
import json
import logging
import numpy as np

# Global setup
_TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_TEXT_EMBEDDING_DIM = 384
_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        logging.info(f"Loading text embedding model: {_TEXT_EMBEDDING_MODEL}")
        _st_model = SentenceTransformer(_TEXT_EMBEDDING_MODEL)
    return _st_model

def extract_text_metadata_features(
    title: str,
    description: str,
    tags: Union[str, List[str]]
) -> dict[str, Any]:
    """
    Encode text (title + description + tags) for a single video using all-MiniLM-L6-v2.

    Returns a dict with:
        text_emb_0 ... text_emb_383 (384 floats)
        title_length                (int)
        num_tags                    (int)
    """
    model = _get_st_model()

    title_safe = title if title else ""
    desc_safe = description if description else ""

    if isinstance(tags, str):
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list):
                tags_list = [tags]
        except:
            tags_list = tags.split() if tags else []
    else:
        tags_list = tags if tags else []

    tags_str = " ".join(tags_list)
    corpus_text = f"{title_safe} {desc_safe} {tags_str}".strip()

    title_length = len(title_safe)
    num_tags = len(tags_list)

    # Encode text
    emb = model.encode(
        corpus_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Output formatting
    row: dict[str, Any] = {
        f"text_emb_{j}": float(emb[j]) for j in range(_TEXT_EMBEDDING_DIM)
    }
    row["title_length"] = title_length
    row["num_tags"] = num_tags

    return row
