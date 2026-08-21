import runpod
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== CONFIG ====================
WORKER_ENDPOINT_ID = os.environ.get("WORKER_ENDPOINT_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
MAX_POLL_TIME = 600

PLAN_CONFIG = {
    "free":     {"tier": "free",    "num_chunks": 1, "label": "512p",  "duration": 5},
    "standard": {"tier": "standard", "num_chunks": 3, "label": "720p",  "duration": 15},
    "pro":      {"tier": "pro",     "num_chunks": 5, "label": "1080p", "duration": 25},
}
# ================================================

def trigger_worker_job(chunk_idx, prompt, tier, seed_image_url):
    """Worker endpoint pe async job trigger karo"""
    if not WORKER_ENDPOINT_ID or not RUNPOD_API_KEY:
        raise ValueError("WORKER_ENDPOINT_ID ya RUNPOD_API_KEY missing hai")
    
    url = f"https://api.runpod.ai/v2/{WORKER_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    chunk_prompt = f"{prompt}. Temporal segment {chunk_idx + 1}."
    
    payload = {
        "input": {
            "chunk_idx": chunk_idx,
            "prompt": chunk_prompt,
            "tier": tier,
            "seed_image_url": seed_image_url
        }
    }
    
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["id"]

def get_job_result(job_id):
    """Job status check karo"""
    url = f"https://api.runpod.ai/v2/{WORKER_ENDPOINT_ID}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def handler(event):
    inp = event["input"]
    prompt = inp.get("prompt", "")
    plan = inp.get("plan", "free")
    seed_image_url = inp.get("seed_image_url")
    
    if not prompt:
        return {"error": "prompt required", "status": "failed"}
    
    if plan not in PLAN_CONFIG:
        return {"error": f"invalid plan '{plan}'. Use: free, standard, pro", "status": "failed"}
    
    cfg = PLAN_CONFIG[plan]
    tier = cfg["tier"]
    num_chunks = cfg["num_chunks"]
    total_duration = cfg["duration"]
    
    print(f"[LB] Plan: {plan} ({cfg['label']}), Chunks: {num_chunks}, Tier: {tier}, Duration: {total_duration}s")
    
    # ========== STEP 1: Parallel trigger ==========
    job_map = {}
    
    try:
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            future_to_chunk = {
                executor.submit(trigger_worker_job, i, prompt, tier, seed_image_url): i 
                for i in range(num_chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    job_id = future.result()
                    job_map[chunk_idx] = job_id
                    print(f"[LB] Chunk {chunk_idx} triggered -> JobID: {job_id}")
                except Exception as e:
                    print(f"[LB] Chunk {chunk_idx} trigger failed: {e}")
                    return {"error": f"chunk {chunk_idx} trigger failed: {str(e)}", "status": "failed"}
    except Exception as e:
        return {"error": f"parallel trigger failed: {str(e)}", "status": "failed"}
    
    # ========== STEP 2: Parallel polling ==========
    results = {}
    start_time = time.time()
    
    while len(results) < num_chunks:
        if time.time() - start_time > MAX_POLL_TIME:
            return {
                "status": "timeout",
                "plan": plan,
                "tier": tier,
                "total_duration_seconds": total_duration,
                "completed_chunks": results,
                "pending_chunks": [i for i in range(num_chunks) if i not in results]
            }
        
        for chunk_idx, job_id in job_map.items():
            if chunk_idx in results:
                continue
            
            try:
                status = get_job_result(job_id)
                job_status = status.get("status")
                output = status.get("output", {})
                
                if job_status == "COMPLETED":
                    results[chunk_idx] = {
                        "chunk_idx": chunk_idx,
                        "url": output.get("url"),
                        "tier": output.get("tier"),
                        "status": "completed"
                    }
                    print(f"[LB] Chunk {chunk_idx} COMPLETED")
                elif job_status == "FAILED":
                    results[chunk_idx] = {
                        "chunk_idx": chunk_idx,
                        "status": "failed",
                        "error": output.get("error", "Worker failed")
                    }
                    print(f"[LB] Chunk {chunk_idx} FAILED")
            except Exception as e:
                print(f"[LB] Status check error chunk {chunk_idx}: {e}")
        
        if len(results) < num_chunks:
            time.sleep(3)
    
    # ========== STEP 3: Final response ==========
    failed_chunks = [r for r in results.values() if r["status"] == "failed"]
    ordered_chunks = [results[i] for i in sorted(results.keys())]
    ordered_urls = [results[i]["url"] for i in sorted(results.keys()) if results[i]["status"] == "completed"]
    
    if failed_chunks:
        return {
            "status": "partial_failure",
            "plan": plan,
            "tier": tier,
            "total_chunks": num_chunks,
            "total_duration_seconds": len(ordered_urls) * 5,
            "chunks": ordered_chunks,
            "failed_count": len(failed_chunks)
        }
    
    return {
        "status": "completed",
        "plan": plan,
        "tier": tier,
        "total_chunks": num_chunks,
        "total_duration_seconds": total_duration,
        "chunks": ordered_chunks,
        "concat_urls": ordered_urls,
        "message": f"{total_duration} second video ready. FFmpeg se concat karo."
    }

runpod.serverless.start({"handler": handler})
