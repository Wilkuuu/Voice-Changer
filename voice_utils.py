"""Shared voice conversion utilities used by app.py and translate.py."""

import torch

SAMPLE_RATE = 16000
CHUNK_SECONDS = 15  # WavLM attention is O(N²); 15s chunks keep it manageable on CPU


def extract_features_chunked(
    knn_vc,
    wav: torch.Tensor,
    chunk_seconds: int = CHUNK_SECONDS,
    progress_cb=None,
) -> torch.Tensor:
    """Extract WavLM features in chunks to avoid O(N²) attention hang on long audio."""
    chunk_samples = chunk_seconds * SAMPLE_RATE
    total_samples = wav.shape[1]

    if total_samples <= chunk_samples:
        return knn_vc.get_features(wav)

    chunks = []
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        feats = knn_vc.get_features(wav[:, start:end])
        chunks.append(feats)
        msg = f"Features: chunk {i+1}/{n_chunks} ({end//SAMPLE_RATE}s / {total_samples//SAMPLE_RATE}s)"
        print(msg)
        if progress_cb:
            progress_cb(msg)

    return torch.cat(chunks, dim=0)
