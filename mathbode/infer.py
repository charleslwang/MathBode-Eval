# mathbode/infer.py
import os, math, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from .clients import make_client
from .utils import build_prompt, call_with_retries, coerce_to_fixed_decimals, force_final_line

def _load_existing(pred_path:str)->Optional[pd.DataFrame]:
    return pd.read_parquet(pred_path) if os.path.exists(pred_path) else None

def _merge_and_save(base:pd.DataFrame, new_rows:pd.DataFrame, pred_path:str):
    if base is None:
        out=new_rows
    else:
        # outer-join on row_id and prefer new non-null fields
        out=(base.set_index("row_id")
                .combine_first(new_rows.set_index("row_id"))
                .reset_index())
        # If both have values, keep the latest non-empty final_line/raw_text/y_hat
        for col in ["raw_text","final_line","y_hat"]:
            if col in new_rows.columns:
                out[col]=out[col+"_x"].where(out[col+"_y"].isna(), out[col+"_y"])
                out=out.drop(columns=[c for c in out.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    out.to_parquet(pred_path, index=False)

def run_inference(
    df: pd.DataFrame,
    provider: str,
    model: str,
    outdir: str,
    temperature: float = 0.0,
    max_tokens: int = 32,
    api_base: Optional[str] = None,
    workers: int = 4,
    provider_rps: float = 0.0
) -> str:
    os.makedirs(outdir, exist_ok=True)
    tag=f"{provider}_{model}".replace("/","_")
    pred_path=os.path.join(outdir, f"preds_{tag}.parquet")

    existing=_load_existing(pred_path)
    if existing is not None and len(existing):
        good = existing.copy()
        if "final_line" in good.columns:
            mask = good["final_line"].notna() & (good["final_line"] != "")
            have = set(good.loc[mask, "row_id"].tolist())
        else:
            have = set()  # no valid outputs yet
        todo = df[~df["row_id"].isin(have)].copy().reset_index(drop=True)
    else:
        todo=df.copy()

    if todo.empty:
        print("✅ Nothing to do; all rows already inferred.")
        return pred_path

    client = make_client(provider, model, temperature, max_tokens, api_base)

    rows=todo.to_dict("records")
    raw_text=[None]*len(rows); final_line=[None]*len(rows); y_hat=[math.nan]*len(rows)

    def work(i):
        r=rows[i]
        prompt=build_prompt(r["prompt"])
        txt=call_with_retries(client, prompt, retries=3, backoff=0.8)
        number_str=None
        # OpenAI JSON mode: parse {"final": "..."} if present
        if provider.lower()=="openai" and txt.strip().startswith("{"):
            try:
                j=json.loads(txt); number_str,_=coerce_to_fixed_decimals(j.get("final",""),6)
            except Exception:
                pass
        if number_str is None:
            number_str,_=coerce_to_fixed_decimals(txt,6)
        raw_text[i]=txt
        final_line[i]=force_final_line(number_str,6)
        if number_str is not None:
            y_hat[i]=float(number_str)
        if provider_rps>0:
            import time; time.sleep(1.0/provider_rps)

    from tqdm import tqdm
    import time
    
    start_time = time.time()
    total_requests = len(rows)
    completed_requests = 0
    last_print = 0
    
    print(f"🚀 Starting inference with {workers} workers for {total_requests} requests...")
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Submit all tasks
        futs = {ex.submit(work, i): i for i in range(len(rows))}
        
        # Create progress bar
        with tqdm(total=total_requests, desc=f"Inferring ({provider}/{model})", unit='req') as pbar:
            for future in as_completed(futs):
                completed_requests += 1
                pbar.update(1)
                
                # Calculate and display ETA and rate
                elapsed = time.time() - start_time
                rate = completed_requests / elapsed if elapsed > 0 else 0
                eta = (total_requests - completed_requests) / rate if rate > 0 else 0
                
                # Update progress bar description with rate and ETA
                pbar.set_postfix({
                    'rate': f"{rate:.1f} req/s",
                    'eta': f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                })
                
                # Print detailed status every 10% progress
                progress = (completed_requests / total_requests) * 100
                if progress - last_print >= 10:
                    last_print = (progress // 10) * 10
                    print(f"\n📊 Progress: {progress:.0f}% ({completed_requests}/{total_requests}) "
                          f"| Rate: {rate:.1f} req/s | ETA: {eta/60:.1f} minutes")
                
                # Check for exceptions
                try:
                    future.result()  # This will raise any exceptions from the worker
                except Exception as e:
                    print(f"\n⚠️ Error in worker for row {futs[future]}: {str(e)}")
                    raise

    part=todo.copy()
    part["raw_text"]=raw_text
    part["final_line"]=final_line
    part["y_hat"]=y_hat

    _merge_and_save(existing, part, pred_path)
    print(f"💾 Saved predictions → {pred_path}")
    print(f"Requests OK/Total: {getattr(client,'ok',0)}/{getattr(client,'total',0)}; last_error={getattr(client,'last_error',None)}")
    return pred_path
