"""Label-free cosine retrieval and spherical k-means."""
import numpy as np
def spherical_kmeans(features, k, seed, max_iter=100, tol=1e-6):
    """GPU fitting, with no labels, spatial coordinates, or queries accepted."""
    import torch
    if features.ndim != 2 or len(features) < k or k < 1:
        raise ValueError("Invalid cluster count")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    n, d = features.shape
    first = int(torch.randint(n, (1,), generator=generator, device=features.device).item())
    centers = [features[first].clone()]
    selected = {first}
    minimum = (1 - features @ centers[0]).clamp_min(0)
    for _ in range(1, k):
        weights = minimum.square()
        if float(weights.sum().item()) <= 1e-12:
            index = next(i for i in range(n) if i not in selected)
        else:
            index = int(torch.multinomial(weights, 1, generator=generator).item())
        selected.add(index)
        centers.append(features[index].clone())
        minimum = torch.minimum(minimum, (1 - features @ centers[-1]).clamp_min(0))
    centers = torch.stack(centers)
    previous = None
    reinitializations = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        scores = features @ centers.T
        best, labels = scores.max(dim=1)
        updated = torch.zeros((k, d), device=features.device, dtype=features.dtype)
        updated.index_add_(0, labels, features)
        norms = updated.norm(dim=1)
        empty = torch.nonzero(norms <= 1e-8).flatten()
        if len(empty):
            farthest = torch.argsort(best, stable=True)
            for position, cluster in enumerate(empty.tolist()):
                updated[cluster] = features[farthest[position]]
                reinitializations += 1
            norms = updated.norm(dim=1)
        updated /= norms[:, None]
        movement = float((1 - (centers * updated).sum(dim=1)).clamp_min(0).max().item())
        unchanged = previous is not None and torch.equal(labels, previous)
        centers = updated
        if unchanged or movement <= tol:
            converged = True
            break
        previous = labels
    best, labels = (features @ centers.T).max(dim=1)
    record = {"seed": seed, "iterations": iteration, "converged": converged,
              "mean_cosine_to_center": float(best.double().mean().item()),
              "mean_cosine_distance": float((1 - best.double()).mean().item()),
              "empty_cluster_reinitializations": reinitializations,
              "cluster_token_counts": torch.bincount(labels, minlength=k).cpu().tolist(),
              "fit_uses_queries": False, "fit_uses_labels": False,
              "fit_scope": "all native tokens, including background",
              "update_precision": str(features.dtype) + " " + features.device.type + " sums", "k": k}
    return labels.cpu().numpy().astype(np.int32), centers.cpu().numpy(), record


def normalize_features(features):
    import torch
    if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
        raise ValueError("Features must be a finite nonempty [N,C] matrix")
    features = features.float()
    norms = features.norm(dim=1, keepdim=True)
    if torch.any(norms <= 1e-8):
        raise ValueError("Cosine similarity is undefined for a zero feature row")
    return features / norms


def cosine_segmentation(features, query_token_indices):
    import torch
    features = normalize_features(features)
    queries = torch.as_tensor(query_token_indices, device=features.device, dtype=torch.long)
    if queries.ndim != 1 or not len(queries) or queries.min() < 0 or queries.max() >= len(features):
        raise ValueError("Queries must index native feature rows")
    scores = features @ features[queries].T
    similarity, labels = scores.max(dim=1)
    return labels, similarity


def fit_kmeans(features, k, *, seed=42, restarts=5, max_iter=100):
    """Choose restarts by feature objective only; no labels enter fitting."""
    import numpy as np
    features = normalize_features(features)
    if restarts <= 0 or max_iter <= 0:
        raise ValueError("restarts and max_iter must be positive")
    fits = [spherical_kmeans(features, k, seed + index, max_iter=max_iter) for index in range(restarts)]
    best = max(range(len(fits)), key=lambda index: fits[index][2]["mean_cosine_to_center"])
    labels, centers, record = fits[best]
    return labels, centers, {**record, "restarts": [fit[2] for fit in fits], "selected_restart": best, "k_source": "user"}
