# mathbode/utils.py
import re, time, json, math
from decimal import Decimal, ROUND_HALF_UP

FINAL_RE = re.compile(r'^\s*FINAL:\s*([+-]?\d+(?:\.\d+))\s*$', re.MULTILINE)


def strict_rules() -> str:
    return (
        "Output EXACTLY one line: 'FINAL: -0.000000' with a dot decimal and six digits. "
        "No scientific notation, commas, spaces, or units. Round HALF-UP. "
        "Output ONLY that single line and NOTHING ELSE."
    )

def build_prompt(raw_prompt:str)->str:
    # keep math content intact, append rules for recency
    return f"{raw_prompt}\n\n{strict_rules()}"

def coerce_to_fixed_decimals(text:str, places:int=6):
    if not text:
        return None, None

    # Find the last valid FINAL line
    last = None
    for m in FINAL_RE.finditer(text):
        last = m
    if not last:
        return None, None

    # Ensure the ENTIRE output (ignoring surrounding whitespace) is just that one line
    only_line = last.group(0).strip()
    if text.strip() != only_line:
        return None, None  # extra chatter → invalid → trigger retry

    # Quantize HALF-UP to N decimals
    val = Decimal(last.group(1))
    q = Decimal(10) ** -places
    fixed = val.quantize(q, rounding=ROUND_HALF_UP)
    return f"{fixed}", float(fixed)

def force_final_line(text_num, places:int=6)->str:
    return f"FINAL: {text_num}" if text_num is not None else ""

def call_with_retries(client, prompt:str, retries:int=3, backoff:float=0.8)->str:
    for i in range(retries):
        out=client.generate(prompt)
        if out: return out
        time.sleep(backoff*(2**i))
    return ""
