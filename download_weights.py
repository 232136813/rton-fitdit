import argparse
import os
import subprocess
import sys

# Disable hf_transfer to avoid "hf_transfer package not available" errors
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def run(cmd, **kwargs):
    print(f"> {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description="Download FitDiT weights and source code.")
    parser.add_argument("--output", type=str, default="/runpod-volume/FitDiT",
                        help="Target directory for weights and source code.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (if model is gated).")
    args = parser.parse_args()

    out = args.output
    os.makedirs(out, exist_ok=True)

    # Step 1: Clone FitDiT source code (for src/ and preprocess/ directories)
    src_marker = os.path.join(out, "src", "pipeline_stable_diffusion_3_tryon.py")
    if not os.path.exists(src_marker):
        print("[1/2] Cloning FitDiT source code ...")
        tmp_clone = out + "_clone_tmp"
        if os.path.exists(tmp_clone):
            run(f"rm -rf {tmp_clone}")
        run(f"git clone --depth 1 https://github.com/BoyuanJiang/FitDiT.git {tmp_clone}")

        # Copy source directories into the output (weights) directory
        for d in ["src", "preprocess", "examples"]:
            src_path = os.path.join(tmp_clone, d)
            dst_path = os.path.join(out, d)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                run(f"cp -r {src_path} {dst_path}")

        # Copy gradio_sd3.py
        for f in ["gradio_sd3.py"]:
            src_path = os.path.join(tmp_clone, f)
            dst_path = os.path.join(out, f)
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                run(f"cp {src_path} {dst_path}")
        run(f"rm -rf {tmp_clone}")
        print("Source code ready.")
    else:
        print("[1/2] Source code already present, skipping clone.")

    # Step 2: Download model weights from HuggingFace
    weight_marker = os.path.join(out, "transformer_garm", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(weight_marker):
        print("[2/2] Downloading model weights from HuggingFace ...")
        token_arg = f"--token {args.hf_token}" if args.hf_token else ""
        try:
            run(f"huggingface-cli download BoyuanJiang/FitDiT --local-dir {out} {token_arg}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: Use Python API (no hf CLI / no git-lfs needed)
            print("\nhuggingface-cli failed. Falling back to Python API ...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                "BoyuanJiang/FitDiT",
                local_dir=out,
                token=args.hf_token,
            )
        print("Model weights ready.")
    else:
        print("[2/2] Model weights already present, skipping download.")

    # Step 3: Download CLIP vision encoders
    # NOTE: Only download the files needed by CLIPVisionModelWithProjection,
    # NOT the full repo (laion/CLIP-ViT-bigG-14 full repo is ~30GB+).
    CLIP_MODELS = {
        "clip-vit-large-patch14": {
            "repo": "openai/clip-vit-large-patch14",
            "files": [
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        },
        "clip-vit-bigG-14": {
            "repo": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            # bigG weights are sharded - no single model.safetensors / pytorch_model.bin
            "files": [
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin.index.json",
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
            ],
        },
    }

    def download_hf_file(repo_id, filename, dest_dir, token=None):
        """Download a single file, trying hf_hub_download first, then raw URL."""
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"  {filename} already exists, skipping.")
            return True

        # Method 1: huggingface_hub API
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=repo_id, filename=filename,
                            local_dir=dest_dir, token=token)
            print(f"  downloaded {filename} (via hf_hub)")
            return True
        except Exception as e:
            print(f"  hf_hub failed for {filename}: {e}")

        # Method 2: direct URL download via urllib
        try:
            import urllib.request
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            print(f"  trying direct download: {url}")
            urllib.request.urlretrieve(url, dest_path)
            print(f"  downloaded {filename} (via urllib)")
            return True
        except Exception as e:
            print(f"  urllib failed for {filename}: {e}")

        # Method 3: wget
        try:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            run(f"wget -q -O '{dest_path}' '{url}'")
            print(f"  downloaded {filename} (via wget)")
            return True
        except Exception:
            pass

        print(f"  FAILED to download {filename}")
        return False

    for local_name, info in CLIP_MODELS.items():
        clip_dir = os.path.join(out, local_name)
        marker = os.path.join(clip_dir, "config.json")
        if not os.path.exists(marker):
            print(f"[3/3] Downloading {info['repo']} ...")
            os.makedirs(clip_dir, exist_ok=True)
            for fname in info["files"]:
                download_hf_file(info["repo"], fname, clip_dir, args.hf_token)
            print(f"  {local_name} done.")
        else:
            print(f"[3/3] {local_name} already present, skipping.")

    # Verify
    print("\n--- Verification ---")
    required = [
        "src/pipeline_stable_diffusion_3_tryon.py",
        "preprocess/dwpose/__init__.py",
        "preprocess/humanparsing/run_parsing.py",
        "transformer_garm/diffusion_pytorch_model.safetensors",
        "transformer_vton/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "pose_guider/diffusion_pytorch_model.bin",
        "dwpose/yolox_l.onnx",
        "humanparsing/parsing_atr.onnx",
        "model_index.json",
        "clip-vit-large-patch14/config.json",
        "clip-vit-large-patch14/model.safetensors",
        "clip-vit-bigG-14/config.json",
        "clip-vit-bigG-14/pytorch_model-00001-of-00002.bin",
    ]

    all_ok = True
    for f in required:
        path = os.path.join(out, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"[{status}] {f}")

    if all_ok:
        print(f"\nAll files ready at {out}!")
    else:
        print("\nSome files are missing. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()