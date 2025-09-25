import re, time

def call_with_retries(client, prompt: str, retries: int = 3, backoff: float = 0.8) -> str:
    for i in range(max(0, int(retries))):
        out = client.generate(prompt)
        if isinstance(out, str) and out.strip():
            return out
        time.sleep(max(0.0, backoff) * (2 ** i))
    return ""

# Sentinels (must match what you use in examples & stops)
ANSWER_START = "[answer_start]"
ANSWER_END   = "[answer_end]"

# Normalize problematic unicode/newlines before parsing
_ZW = "\u200b\u200c\u200d\u2060\ufeff"   # zero-width chars
_BAD_MINUS = "−"                          # U+2212
_NBSP = "\xa0"

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for ch in _ZW: s = s.replace(ch, "")
    s = s.replace(_BAD_MINUS, "-").replace(_NBSP, " ")
    return s

# Exactly 6 decimals between tags; allow whitespace/newlines
ANSWER_RE = re.compile(
    rf"{re.escape(ANSWER_START)}\s*([+-]?\d+\.\d{{6}})\s*{re.escape(ANSWER_END)}",
    flags=re.DOTALL  # allow newline between start/number/end
)

def strict_rules() -> str:
    return (
        f"You may reason a little, but you MUST end with exactly one final line:\n"
        f"{ANSWER_START} -0.000000 {ANSWER_END}\n"
        "The number must have exactly six digits after the decimal (no scientific notation). "
        "No other text after the END tag."
    )

def build_prompt(raw_prompt: str) -> str:
    return raw_prompt  # user content = just the problem text

def coerce_to_fixed_decimals(text: str, places: int = 6):
    if not text:
        return None, None
    text = _norm(text)
    last = None
    for m in ANSWER_RE.finditer(text):
        last = m
    if not last:
        return None, None
    num_str = last.group(1)  # already exactly 6 decimals; NO rounding
    try:
        return num_str, float(num_str)
    except Exception:
        return None, None

def force_final_line(text_num: str, places: int = 6) -> str:
    return f"{ANSWER_START} {text_num} {ANSWER_END}" if text_num is not None else ""
