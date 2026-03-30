"""
generate_traces.py - Main reasoning trace generation pipeline.

Supports THREE backends:
  1. api  (RECOMMENDED): DeepSeek API (pay-as-you-go)
  2. hf   : HuggingFace Transformers (local GPU on HPC)
  3. vllm : vLLM server (local GPU on HPC, faster batching)

Usage:
    # API backend
    DEEPSEEK_API_KEY=sk-xxx python src/generation/generate_traces.py \
        --dataset math500 --output data/traces/math500_r1.jsonl \
        --backend api --model deepseek-r1

    # HPC local (Qwen-7B or Llama-8B)
    python src/generation/generate_traces.py \
        --dataset math500 --output data/traces/math500_qwen7b.jsonl \
        --backend hf --model r1-distill-qwen-7b
"""

import argparse, json, os, re, time, logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _load_dotenv():
    """
    Load environment variables from .env file.

    Searches for .env in the current directory and parent directories
    (up to 3 levels). Only sets variables that are not already set,
    so explicit environment variables always take precedence.
    """
    # Search for .env file
    env_path = None
    search_dir = os.getcwd()
    for _ in range(4):  # Current + 3 parent levels
        candidate = os.path.join(search_dir, ".env")
        if os.path.isfile(candidate):
            env_path = candidate
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent

    if env_path is None:
        return

    loaded = 0
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Only set if not already in environment
            if key and value and key not in os.environ:
                os.environ[key] = value
                loaded += 1

    if loaded:
        logger.debug(f"Loaded {loaded} variables from {env_path}")


def extract_think_block(full_text: str) -> tuple[str, str]:
    """Extract <think>...</think> trace and answer. Returns (trace, answer)."""
    m = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
    if m:
        return m.group(1).strip(), full_text[m.end():].strip()
    return "", full_text.strip()


def build_record(item, gen_result, score_result, model_name, model_short_name, backend, temperature):
    """Build standardized output record with ALL required fields."""
    trace = gen_result.get("reasoning_trace", "")
    return {
        "item_id": item["item_id"],
        "dataset": item["dataset"],
        "problem": item["problem"],
        "prompt": item["prompt"],
        "ground_truth": item["ground_truth"],
        "answer_type": item["answer_type"],
        "answer_extraction": item.get("answer_extraction"),
        "metadata": item.get("metadata", {}),
        "full_response": gen_result.get("full_response", ""),
        "reasoning_trace": trace,
        "answer_text": gen_result.get("answer_text", ""),
        "trace_token_count": gen_result.get("trace_token_count", 0),
        "trace_length": len(trace),
        "total_generated_tokens": gen_result.get("total_generated_tokens", 0),
        "prompt_tokens": gen_result.get("prompt_tokens", 0),
        "reasoning_tokens": gen_result.get("reasoning_tokens", 0),
        "generation_time_seconds": gen_result.get("generation_time_seconds", 0),
        "mean_log_prob": gen_result.get("mean_log_prob"),
        "is_correct": score_result["is_correct"],
        "extracted_answer": score_result["extracted_answer"],
        "comparison_method": score_result["comparison_method"],
        "model_name": model_name,
        "model_short_name": model_short_name,
        "backend": backend,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }


def _load_checkpoint(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try: done.add(json.loads(line.strip()).get("item_id", ""))
                except: pass
    if done: logger.info(f"Resuming: {len(done)} items completed")
    return done


def _print_summary(results, start_time, output_path):
    t = time.time() - start_time
    n = max(len(results), 1)
    c = sum(1 for r in results if r.get("is_correct", False))
    mt = sum(r.get("trace_token_count", 0) for r in results) / n
    mc = sum(r.get("trace_length", 0) for r in results) / n
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete: {len(results)} items in {t/60:.1f}min | Acc={c/n:.1%}")
    logger.info(f"Mean trace: {mt:.0f} tokens, {mc:.0f} chars → {output_path}")


# --- API Backend ---
def _run_api_pipeline(items, output_path, model_name, model_short_name,
                      api_model, base_url, temperature, max_tokens,
                      checkpoint_interval, delay):
    from src.generation.api_client import DeepSeekClient
    from src.generation.scoring import score_answer
    client = DeepSeekClient(base_url=base_url if base_url else None)
    results, t0 = [], time.time()
    with open(output_path, 'a') as f:
        for i, item in enumerate(items):
            try:
                g = client.generate(item["prompt"], model=api_model,
                                    temperature=temperature, max_tokens=max_tokens)
                s = score_answer(g["answer_text"], item["ground_truth"],
                                 item["answer_type"], item.get("answer_extraction"))
                rec = build_record(item, g, s, model_name, model_short_name, "api", temperature)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                results.append(rec)
                if (i+1) % checkpoint_interval == 0 or i+1 == len(items):
                    u = client.get_usage_summary()
                    acc = sum(1 for r in results if r["is_correct"]) / len(results)
                    logger.info(f"[{i+1}/{len(items)}] Acc={acc:.1%} Trace={g['trace_token_count']}tok ${u['estimated_cost_usd']:.3f}")
            except Exception as e:
                logger.error(f"Failed {item['item_id']}: {e}")
                f.write(json.dumps({"item_id": item["item_id"], "error": str(e),
                                    "model_name": model_name, "backend": "api",
                                    "timestamp": datetime.now().isoformat()}) + '\n'); f.flush()
            if delay > 0 and i < len(items)-1: time.sleep(delay)
    _print_summary(results, t0, output_path)
    logger.info(f"API cost: ${client.get_usage_summary()['estimated_cost_usd']:.3f}")


# --- HF Backend ---
def _load_hf_model(model_name, quantize_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name} (4bit={quantize_4bit})")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kw = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except ImportError: pass
    mdl = AutoModelForCausalLM.from_pretrained(model_name, **kw); mdl.eval()
    logger.info(f"Loaded on {next(mdl.parameters()).device}")
    return mdl, tok

def _gen_single_hf(prompt, model, tokenizer, temperature=0.6, top_p=0.95,
                    max_new_tokens=32768, extract_logprobs=True):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    il = ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, do_sample=True, output_scores=extract_logprobs,
                              return_dict_in_generate=True)
    gt = time.time() - t0
    gen_ids = out.sequences[0][il:]
    ft = tokenizer.decode(gen_ids, skip_special_tokens=True)
    mlp = None
    if extract_logprobs and hasattr(out, 'scores') and out.scores:
        try:
            lps = [torch.log_softmax(sc[0], dim=-1)[gen_ids[j].item()].item() for j, sc in enumerate(out.scores)]
            mlp = sum(lps)/len(lps) if lps else None
        except: pass
    rt, at = extract_think_block(ft)
    ttc = len(tokenizer.encode(rt)) if rt else 0
    return {"full_response": ft, "reasoning_trace": rt, "answer_text": at,
            "trace_token_count": ttc, "total_generated_tokens": len(gen_ids),
            "prompt_tokens": il, "reasoning_tokens": ttc,
            "generation_time_seconds": round(gt, 2), "mean_log_prob": mlp}

def _run_hf_pipeline(items, output_path, model_name, model_short_name,
                      temperature, top_p, max_new_tokens, quantize_4bit, checkpoint_interval):
    from src.generation.scoring import score_answer
    model, tokenizer = _load_hf_model(model_name, quantize_4bit)
    results, t0 = [], time.time()
    with open(output_path, 'a') as f:
        for i, item in enumerate(items):
            try:
                g = _gen_single_hf(item["prompt"], model, tokenizer,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
                s = score_answer(g["answer_text"], item["ground_truth"],
                                 item["answer_type"], item.get("answer_extraction"))
                rec = build_record(item, g, s, model_name, model_short_name, "hf", temperature)
                f.write(json.dumps(rec, ensure_ascii=False)+'\n'); f.flush()
                results.append(rec)
                if (i+1) % checkpoint_interval == 0 or i+1 == len(items):
                    el = time.time()-t0; eta = (len(items)-i-1)*el/max(i+1,1)
                    acc = sum(1 for r in results if r["is_correct"])/len(results)
                    logger.info(f"[{i+1}/{len(items)}] Acc={acc:.1%} Trace={g['trace_token_count']}tok ETA={eta/60:.0f}min")
            except Exception as e:
                logger.error(f"Failed {item['item_id']}: {e}")
                f.write(json.dumps({"item_id": item["item_id"], "error": str(e),
                                    "model_name": model_name, "backend": "hf",
                                    "timestamp": datetime.now().isoformat()})+'\n'); f.flush()
    _print_summary(results, t0, output_path)


# --- SC Generation ---
def run_self_consistency_generation(dataset_name, output_path, model_name, model_short_name,
                                     backend="api", api_model="deepseek-reasoner", base_url=None,
                                     num_samples=8, temperature=0.8, quantize_4bit=False, limit=None):
    from src.generation.dataset_loader import load_dataset_items
    from src.generation.scoring import score_answer
    items = load_dataset_items(dataset_name, limit=limit)
    done = _load_checkpoint(output_path)
    items = [it for it in items if it["item_id"] not in done]
    logger.info(f"SC: {len(items)} items × {num_samples} samples, backend={backend}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if backend == "api":
        from src.generation.api_client import DeepSeekClient
        client = DeepSeekClient(base_url=base_url if base_url else None)
        gfn = lambda p: client.generate(p, model=api_model, temperature=temperature)
    elif backend == "hf":
        mdl, tok = _load_hf_model(model_name, quantize_4bit)
        gfn = lambda p: _gen_single_hf(p, mdl, tok, temperature=temperature, extract_logprobs=False)
    else:
        raise ValueError(f"SC supports 'api'/'hf', got '{backend}'")
    with open(output_path, 'a') as f:
        for i, item in enumerate(items):
            samples = []
            for s in range(num_samples):
                try:
                    g = gfn(item["prompt"])
                    sc = score_answer(g["answer_text"], item["ground_truth"],
                                      item["answer_type"], item.get("answer_extraction"))
                    samples.append({"sample_idx": s, "answer_text": g["answer_text"],
                                    "extracted_answer": sc["extracted_answer"],
                                    "is_correct": sc["is_correct"],
                                    "trace_token_count": g.get("trace_token_count", 0)})
                except Exception as e:
                    samples.append({"sample_idx": s, "error": str(e)})
                if backend == "api" and s < num_samples-1: time.sleep(0.3)
            rec = {"item_id": item["item_id"], "dataset": item["dataset"],
                   "ground_truth": item["ground_truth"], "answer_type": item["answer_type"],
                   "answer_extraction": item.get("answer_extraction"),
                   "model_name": model_name, "model_short_name": model_short_name,
                   "backend": backend, "num_samples": num_samples,
                   "samples": samples, "timestamp": datetime.now().isoformat()}
            f.write(json.dumps(rec, ensure_ascii=False)+'\n'); f.flush()
            if (i+1) % 10 == 0: logger.info(f"SC: {i+1}/{len(items)}")
    logger.info(f"SC complete → {output_path}")


# --- Main pipeline ---
def run_generation_pipeline(dataset_name, output_path, model_key="r1-distill-qwen-7b",
                             backend="api", temperature=0.6, top_p=0.95, max_tokens=32768,
                             quantize_4bit=False, limit=None, checkpoint_interval=10,
                             api_base_url=None, delay=0.5):
    from src.generation.api_client import get_model_info
    from src.generation.dataset_loader import load_dataset_items
    mi = get_model_info(model_key)
    mn = mi.get("hf_model", mi.get("api_model", model_key))
    msn = mi.get("display_name", model_key)
    am = mi.get("api_model", model_key)
    logger.info(f"Model: {msn} | Backend: {backend} | Dataset: {dataset_name}")
    items = load_dataset_items(dataset_name, limit=limit)
    done = _load_checkpoint(output_path)
    items = [it for it in items if it["item_id"] not in done]
    logger.info(f"Items: {len(items)} remaining")
    if not items: logger.info("All done!"); return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if backend == "api":
        _run_api_pipeline(items, output_path, mn, msn, am,
                          api_base_url or mi.get("base_url"), temperature, max_tokens,
                          checkpoint_interval, delay)
    elif backend == "hf":
        _run_hf_pipeline(items, output_path, mn, msn, temperature, top_p,
                          max_tokens, quantize_4bit, checkpoint_interval)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    # Load .env file FIRST, before any code reads environment variables
    _load_dotenv()

    from src.generation.api_client import list_models
    p = argparse.ArgumentParser(description="Generate reasoning traces",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Models: " + ", ".join(m["short_name"] for m in list_models()))
    p.add_argument("--dataset", required=True, choices=["math500","gsm8k","gpqa_diamond","arc_challenge"])
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="r1-distill-qwen-7b")
    p.add_argument("--backend", default="api", choices=["api","hf","vllm"])
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--api-base-url", default=None)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--self-consistency", action="store_true")
    p.add_argument("--num-samples", type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    from src.generation.api_client import get_model_info
    mi = get_model_info(a.model)
    if a.self_consistency:
        run_self_consistency_generation(a.dataset, a.output,
            mi.get("hf_model", mi.get("api_model", a.model)), mi.get("display_name", a.model),
            a.backend, mi.get("api_model", a.model), a.api_base_url or mi.get("base_url"),
            a.num_samples, 0.8, a.quantize_4bit, a.limit)
    else:
        run_generation_pipeline(a.dataset, a.output, a.model, a.backend, a.temperature,
                                 max_tokens=a.max_tokens, quantize_4bit=a.quantize_4bit,
                                 limit=a.limit, checkpoint_interval=a.checkpoint_interval,
                                 api_base_url=a.api_base_url, delay=a.delay)

if __name__ == "__main__":
    main()
