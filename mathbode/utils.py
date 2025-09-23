# mathbode/utils.py
import re, time, json, math
from decimal import Decimal, ROUND_HALF_UP

NUMBER_RE = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

def strict_rules() -> str:
    return ("Output EXACTLY one line: 'FINAL: -0.000000' with a dot decimal and six digits. "
            "No scientific notation, commas, spaces, or units. Round HALF-UP.")

def build_prompt(raw_prompt:str)->str:
    # keep math content intact, append rules for recency
    return f"{raw_prompt}\n\n{strict_rules()}"

def coerce_to_fixed_decimals(text:str, places:int=6):
    if not text: return None, None
    m=NUMBER_RE.search(text)
    if not m: return None, None
    val=Decimal(m.group(0))
    q=Decimal(10) ** -places
    fixed=val.quantize(q, rounding=ROUND_HALF_UP)
    return f"{fixed}", float(fixed)

def force_final_line(text_num, places:int=6)->str:
    return f"FINAL: {text_num}" if text_num is not None else ""

def call_with_retries(client, prompt:str, retries:int=3, backoff:float=0.8)->str:
    for i in range(retries):
        out=client.generate(prompt)
        if out: return out
        time.sleep(backoff*(2**i))
    return ""
