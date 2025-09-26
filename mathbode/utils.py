# import re, time
# from decimal import Decimal, InvalidOperation, ROUND_DOWN

# def call_with_retries(client, prompt: str, retries: int = 3, backoff: float = 0.8) -> str:
#     for i in range(max(0, int(retries))):
#         out = client.generate(prompt)
#         if isinstance(out, str) and out.strip():
#             return out
#         time.sleep(max(0.0, backoff) * (2 ** i))
#     return ""

# # Sentinels (must match what you use in examples & stops)
# ANSWER_START = "[answer_start]"
# ANSWER_END   = "[answer_end]"

# # Normalize problematic unicode/newlines before parsing
# _ZW = "\u200b\u200c\u200d\u2060\ufeff"   # zero-width chars
# _BAD_MINUS = "−"                          # U+2212
# _NBSP = "\xa0"

# def _norm(s: str) -> str:
#     if not isinstance(s, str): return ""
#     s = s.replace("\r\n", "\n").replace("\r", "\n")
#     for ch in _ZW: s = s.replace(ch, "")
#     s = s.replace(_BAD_MINUS, "-").replace(_NBSP, " ")
#     return s

# # Exactly 6 decimals between tags; allow whitespace/newlines
# ANSWER_RE = re.compile(
#     rf"{re.escape(ANSWER_START)}\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*{re.escape(ANSWER_END)}",
#     flags=re.DOTALL
# )

# def strict_rules() -> str:
#     return (
#         f"Do not output anything except the numerical final answer, in the following format example \n"
#         f"{ANSWER_START} 1.204239 {ANSWER_END}\n"
#         "Show exactly six digits after the decimal (no scientific notation). "
#         "There should be no other text in this response other than the final answer and tags."
#     )

# def build_prompt(raw_prompt: str) -> str:
#     return raw_prompt  # user content = just the problem text

# def coerce_to_fixed_decimals(text: str, places: int = 6):
#     if not text:
#         return None, None
#     text = _norm(text)

#     last = None
#     for m in ANSWER_RE.finditer(text):
#         last = m
#     if not last:
#         return None, None

#     raw = last.group(1).strip()
#     try:
#         dec = Decimal(raw)  # handles "1", "3.2", ".5", "1e-3", etc.
#         quant = Decimal("0." + "0"*places)  # e.g., 0.000001
#         # Truncate (toward zero), not round:
#         dec_trunc = dec.quantize(quant, rounding=ROUND_DOWN)
#         fixed_str = format(dec_trunc, "f")  # exact decimal string with 6 places
#         return fixed_str, float(dec_trunc)
#     except (InvalidOperation, ValueError):
#         return None, None

# def force_final_line(text_num: str, places: int = 6) -> str:
#     return f"{ANSWER_START} {text_num} {ANSWER_END}" if text_num is not None else ""

import re, time
from decimal import Decimal, InvalidOperation, ROUND_DOWN

def call_with_retries(client, prompt: str, retries: int = 3, backoff: float = 0.8) -> str:
    for i in range(max(0, int(retries))):
        out = client.generate(prompt)
        if isinstance(out, str) and out.strip():
            return out
        time.sleep(max(0.0, backoff) * (2 ** i))
    return ""

ANSWER_START = "[answer_start]"
ANSWER_END   = "[answer_end]"

_ZW = "\u200b\u200c\u200d\u2060\ufeff"
_BAD_MINUS = "−"
_NBSP = "\xa0"

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for ch in _ZW: s = s.replace(ch, "")
    s = s.replace(_BAD_MINUS, "-").replace(_NBSP, " ")
    return s

# NEW: capture the WHOLE content between tags (anything), we’ll pick the last number inside it.
_TAG_RE = re.compile(
    rf"{re.escape(ANSWER_START)}\s*(.*?)\s*{re.escape(ANSWER_END)}",
    flags=re.DOTALL
)

# Numeric token finder (accepts +/-, decimals, optional exponent; we can parse it even if rules say “no sci”)
_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def strict_rules() -> str:
    return (
        f"Do not output anything except the numerical final answer, in the following format example \n"
        f"{ANSWER_START} 1.204239 {ANSWER_END}\n"
        "Show exactly six digits after the decimal (no scientific notation). "
        "There should be no other text in this response other than the final answer and tags."
    )

def build_prompt(raw_prompt: str) -> str:
    return raw_prompt

def coerce_to_fixed_decimals(text: str, places: int = 6):
    if not text:
        return None, None
    text = _norm(text)

    last = None
    for m in _TAG_RE.finditer(text):
        last = m
    if not last:
        return None, None

    inner = last.group(1)  # everything between tags, possibly like "x = -3.1415926"
    nums = list(_NUM_RE.finditer(inner))
    if not nums:
        return None, None

    raw = nums[-1].group(0).strip()  # take the LAST numeric token inside the tags
    try:
        dec = Decimal(raw)
        quant = Decimal("0." + "0"*places)  # e.g., 0.000001
        dec_trunc = dec.quantize(quant, rounding=ROUND_DOWN)  # truncate, don’t round
        fixed_str = format(dec_trunc, "f")  # exact string with the requested places
        return fixed_str, float(dec_trunc)
    except (InvalidOperation, ValueError):
        return None, None

def force_final_line(text_num: str, places: int = 6) -> str:
    return f"{ANSWER_START} {text_num} {ANSWER_END}" if text_num is not None else ""
