"""Embedding utilities for similarity-based retrieval.

高速な類似度検索のための埋め込み計算・管理ユーティリティ。
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# グローバルキャッシュ（モデルの遅延読み込み）
_embedding_model: "SentenceTransformer | None" = None
_model_name: str = "all-MiniLM-L6-v2"  # 384次元、高速


def get_embedding_model() -> "SentenceTransformer":
    """埋め込みモデルを取得（遅延読み込み）."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {_model_name}")
        _embedding_model = SentenceTransformer(_model_name)
    return _embedding_model


def compute_embedding(text: str) -> np.ndarray:
    """単一テキストの埋め込みを計算.

    Args:
        text: 入力テキスト

    Returns:
        正規化された埋め込みベクトル (384,)
    """
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True)


def compute_embeddings_batch(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """バッチで埋め込みを計算.

    Args:
        texts: テキストリスト
        batch_size: バッチサイズ
        show_progress: プログレスバー表示

    Returns:
        正規化された埋め込み行列 (n, 384)
    """
    model = get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    """埋め込みを .npy ファイルに保存.

    Args:
        embeddings: 埋め込み行列
        path: 保存先パス
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info(f"Saved embeddings to {path} (shape: {embeddings.shape})")


def load_embeddings(path: str | Path) -> np.ndarray:
    """埋め込みを .npy ファイルから読み込み.

    Args:
        path: ファイルパス

    Returns:
        埋め込み行列
    """
    path = Path(path)
    embeddings = np.load(path)
    logger.info(f"Loaded embeddings from {path} (shape: {embeddings.shape})")
    return embeddings


def cosine_similarity_batch(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """クエリと候補群のコサイン類似度を計算.

    埋め込みは正規化済みを前提とし、内積で計算（高速）。

    Args:
        query: クエリ埋め込み (d,) or (1, d)
        candidates: 候補埋め込み (n, d)

    Returns:
        類似度スコア (n,)
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)
    # 正規化済みなら内積 = コサイン類似度
    return (candidates @ query.T).flatten()


def top_k_similar(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k 類似候補を取得.

    Args:
        query: クエリ埋め込み (d,)
        candidates: 候補埋め込み (n, d)
        k: 取得数

    Returns:
        (top_k_indices, top_k_scores) のタプル
    """
    scores = cosine_similarity_batch(query, candidates)
    k = min(k, len(scores))
    # argpartition は O(n) で top-k を取得（argsort より高速）
    top_k_idx = np.argpartition(scores, -k)[-k:]
    # top-k 内でソート
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    return top_k_idx, scores[top_k_idx]
