"""
Download FitDiT model weights, CLIP encoders, and source code to a local directory.

Usage:
    python download_weights.py --output /runpod-volume/FitDiT

This script:
    1. Clones the FitDiT GitHub repo (source code + preprocessing).
    2. Downloads model weights from HuggingFace (BoyuanJiang/FitDiT).
    3. Downloads CLIP vision encoders (ViT-Large + ViT-bigG).

Total download size: ~9.7 GB.
"""

import argparse
import os
import subprocess
import sys


def run(cmd, **kwargs):
    print(f" $ {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Download FitDiT weights and source code."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/runpod-volume/FitDiT",
        help="Target directory for weights and source code.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (if model is gated).",
    )
    args = parser.parse_args()

    out = args.output
    os.makedirs(out, exist_ok=True)

    # Step 1: Clone FitDiT source code (for src/ and preprocess/ directories)
    src_marker = os.path.join(
        out, "src", "pipeline_stable_diffusion_3_tryon.py"
    )
    if not os.path.exists(src_marker):
        print("[1/2] Cloning FitDiT source code ...")
        tmp_clone = out + "_clone_tmp"
        if os.path.exists(tmp_clone):
            run(f"rm -rf {tmp_clone}")
        run(
            f"git clone --depth 1 https://github.com/BoyuanJiang/FitDiT.git {tmp_clone}"
        )

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
        print(" Source code ready.")
    else:
        print("[1/2] Source code already present, skipping clone.")

    # Step 2: Download model weights from HuggingFace
    weight_marker = os.path.join(
        out, "transformer_garm", "diffusion_pytorch_model.safetensors"
    )
    if not os.path.exists(weight_marker):
        print("[2/2] Downloading model weights from HuggingFace ...")
        token_arg = f"--token {args.hf_token}" if args.hf_token else ""

        # Use huggingface-cli to download
        try:
            run(
                f"hf download BoyuanJiang/FitDiT --local-dir {out} {token_arg}"
            )
        except subprocess.CalledProcessError:
            print("\n huggingface-cli failed. Trying git git-lfs clone ...")
            run("git lfs install")
            run(f"git clone https://huggingface.co/BoyuanJiang/FitDiT {out}_hf_tmp")

            # Move weight directories
            hf_tmp = out + "_hf_tmp"
            for d in [
                "transformer_garm",
                "transformer_vton",
                "vae",
                "pose_guider",
                "scheduler",
                "dwpose",
                "humanparsing",
            ]:
                src_path = os.path.join(hf_tmp, d)
                dst_path = os.path.join(out, d)
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    run(f"mv {src_path} {dst_path}")

            # Copy model_index.json
            idx_src = os.path.join(hf_tmp, "model_index.json")
            idx_dst = os.path.join(out, "model_index.json")
            if os.path.exists(idx_src):
                run(f"cp {idx_src} {idx_dst}")
            run(f"rm -rf {hf_tmp}")

        print(" Model weights ready.")
    else:
        print("[2/2] Model weights already present, skipping download.")

    # Step 3: Download CLIP vision encoders
    CLIP_MODELS = {
        "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
        "clip-vit-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-s39B-b160k",
    }
    for local_name, hf_repo in CLIP_MODELS.items():
        clip_dir = os.path.join(out, local_name)
        marker = os.path.join(clip_dir, "config.json")
        if not os.path.exists(marker):
            print(f"[3/3] Downloading {hf_repo} ...")
            token_arg = f"--token {args.hf_token}" if args.hf_token else ""
            run(
                f"hf download {hf_repo} --local-dir {clip_dir} {token_arg}"
            )
            print(f" {local_name} ready.")
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
        "clip-vit-bigG-14/config.json",
    ]
    all_ok = True
    for f in required:
        path = os.path.join(out, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"[{status}] {f}")

    if all_ok:
        print(f"\nAll files ready at {out}")
    else:
        print(f"\nSome files are missing. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()