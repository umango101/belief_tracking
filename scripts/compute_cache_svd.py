import os

import fire
import torch
from tqdm import tqdm


def find_cache_files(root_dir):
    """Find all .pt files in the directory structure."""
    cache_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pt") and "lens.pt" not in file:
                cache_files.append(os.path.join(root, file))
    return cache_files


def compute_svd(cache_path):
    """Compute SVD of a cache file and return results."""

    try:
        cache = torch.load(cache_path)

        # Convert to tensor if it's not already
        if not isinstance(cache, torch.Tensor):
            cache = torch.tensor(cache)

        if "visibility_sent_acts.pt" in cache_path:
            token_lens_bigtom = torch.load(
                f"{os.path.dirname(cache_path)}/visibility_sent_lens.pt"
            ).to(cache.device)
            max_tokens = cache.size(2)
            mask = torch.arange(max_tokens).unsqueeze(0) < token_lens_bigtom.unsqueeze(
                1
            )
            mask = mask.to(cache.device)

        results = {}
        for l in tqdm(
            range(80), desc=f"Processing layers for {os.path.basename(cache_path)}"
        ):
            try:
                if "visibility_sent_acts.pt" in cache_path:
                    layer_cache = cache[:, l][mask]
                else:
                    layer_cache = cache[:, l, :, :].cuda()
                    layer_cache = layer_cache.reshape(
                        layer_cache.size(0) * layer_cache.size(1), layer_cache.size(-1)
                    )

                # Compute SVD
                U, S, Vh = torch.linalg.svd(layer_cache)

                results[l] = {
                    "U": U.cpu().numpy(),
                    "S": S.cpu().numpy(),
                    "Vh": Vh.cpu().numpy(),
                    "shape": layer_cache.shape,
                }
            except Exception as e:
                print(f"Error processing layer {l} in {cache_path}: {str(e)}")
                continue

        return results
    except Exception as e:
        print(f"Error processing {cache_path}: {str(e)}")
        return None


def main(dataset_type: str = "causalToM"):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(root_dir, "caches", "llama-3-70B-Instruct", dataset_type)
    output_dir = os.path.join("svd_results", dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # Find all cache files
    cache_files = find_cache_files(cache_dir)
    print(f"Found {len(cache_files)} cache files")

    # Process each cache file
    results = {}
    for cache_path in tqdm(cache_files, desc="Computing SVDs"):
        rel_path = os.path.relpath(cache_path, cache_dir)
        svd_results = compute_svd(cache_path)

        if svd_results is not None:
            # Create directory structure for this cache
            cache_output_dir = os.path.join(output_dir, os.path.splitext(rel_path)[0])
            singular_vecs_dir = os.path.join(cache_output_dir, "singular_vecs")
            singular_values_dir = os.path.join(cache_output_dir, "singular_values")

            os.makedirs(singular_vecs_dir, exist_ok=True)
            os.makedirs(singular_values_dir, exist_ok=True)

            # Save results for each layer
            for layer_idx, layer_results in svd_results.items():
                # Save singular vectors
                torch.save(
                    torch.from_numpy(layer_results["Vh"]),
                    os.path.join(singular_vecs_dir, f"{layer_idx}.pt"),
                )

                # Save singular values
                torch.save(
                    torch.from_numpy(layer_results["S"]),
                    os.path.join(singular_values_dir, f"{layer_idx}.pt"),
                )

            # Store metadata
            results[rel_path] = {
                "shape": layer_results["shape"],
                "output_dir": cache_output_dir,
                "num_layers": len(svd_results),
            }


if __name__ == "__main__":
    fire.Fire(main)
