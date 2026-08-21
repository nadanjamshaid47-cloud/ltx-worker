import runpod
import subprocess
import os
import requests
import boto3

MODELS = "/workspace/models/ltx-2.5"

TIER_CONFIG = {
    "free": {
        "num_frames": 121,
        "height": 512, "width": 768,
        "quantization": None,
    },
    "standard": {
        "num_frames": 121,
        "height": 704, "width": 1280,
        "quantization": None,
    },
    "pro": {
        "num_frames": 121,
        "height": 1088, "width": 1920,
        "quantization": "fp8-scaled-mm",
    },
}

# Match this to the "Execution timeout" value set on the RunPod endpoint.
SUBPROCESS_TIMEOUT_SECONDS = 600
SEED_DOWNLOAD_TIMEOUT_SECONDS = 30


def handler(event):
    inp = event["input"]
    chunk_idx = inp.get("chunk_idx", 0)
    prompt = inp["prompt"]
    tier = inp.get("tier", "standard")
    seed_image_url = inp.get("seed_image_url")

    if tier not in TIER_CONFIG:
        return {"error": f"invalid tier '{tier}'", "chunk_idx": chunk_idx}
    cfg = TIER_CONFIG[tier]

    output_path = f"/tmp/chunk_{chunk_idx}.mp4"
    seed_path = f"/tmp/seed_{chunk_idx}.png"

    cmd = [
        "python3", "-m", "ltx_pipelines.ti2vid_two_stages",
        "--transformer-path", f"{MODELS}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "--text-encoder-path", f"{MODELS}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "--video-vae-path", f"{MODELS}/vae/ltx-2.5-video-vae-bf16.safetensors",
        "--audio-vae-path", f"{MODELS}/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "--spatial-upsampler-path", f"{MODELS}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "--distilled-lora", f"{MODELS}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        "--prompt", prompt,
        "--offload", "none",
        "--max-batch-size", "4",
        "--height", str(cfg["height"]),
        "--width", str(cfg["width"]),
        "--num-frames", str(cfg["num_frames"]),
        "--output-path", output_path,
    ]

    if cfg["quantization"]:
        cmd += ["--quantization", cfg["quantization"]]

    if seed_image_url:
        try:
            resp = requests.get(seed_image_url, timeout=SEED_DOWNLOAD_TIMEOUT_SECONDS)
            resp.raise_for_status()
        except requests.RequestException as e:
            return {"error": f"seed image download failed: {e}", "chunk_idx": chunk_idx}

        with open(seed_path, "wb") as f:
            f.write(resp.content)
        cmd += ["--image", seed_path, "0", "0.9"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        _cleanup(output_path, seed_path)
        return {"error": "generation timed out", "chunk_idx": chunk_idx}

    if result.returncode != 0:
        _cleanup(output_path, seed_path)
        return {
            "error": result.stderr[-2000:],
            "stdout": result.stdout[-2000:],
            "chunk_idx": chunk_idx,
        }

    try:
        url = upload_to_storage(output_path, chunk_idx, event["id"])
    except Exception as e:
        _cleanup(output_path, seed_path)
        return {"error": f"upload failed: {e}", "chunk_idx": chunk_idx}

    _cleanup(output_path, seed_path)
    return {"chunk_idx": chunk_idx, "url": url, "tier": tier}


def upload_to_storage(path, chunk_idx, job_id):
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["B2_ENDPOINT_URI"],
        aws_access_key_id=os.environ["B2_KEY_ID"],
        aws_secret_access_key=os.environ["B2_APPLICATION_KEY"],
    )
    bucket = os.environ["B2_BUCKET_NAME"]
    key = f"chunks/{job_id}/chunk_{chunk_idx}.mp4"
    s3.upload_file(path, bucket, key)

    public_url = os.environ.get("B2_PUBLIC_URL")
    if public_url:
        return f"{public_url}/{key}"
    # Fallback: construct a direct S3-style URL if no public/CDN URL is configured.
    return f"{os.environ['B2_ENDPOINT_URI']}/{bucket}/{key}"


def _cleanup(*paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
