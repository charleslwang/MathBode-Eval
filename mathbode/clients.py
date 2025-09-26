# mathbode/clients.py
import os, time
from collections import deque
from typing import Optional
from mathbode.utils import strict_rules, ANSWER_START, ANSWER_END

# ---------- helpers ----------

def _est_tokens_from_text(s: str) -> int:
    # very rough fallback (~4 chars/token)
    return max(0, int(len(s) / 4)) if s else 0

class RequestRateLimiter:
    """Simple sliding-window RPM limiter."""
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._times = deque()  # seconds

    def wait(self):
        now = time.time()
        window = now - 60.0
        while self._times and self._times[0] < window:
            self._times.popleft()
        if len(self._times) >= self.rpm:
            sleep_s = 60.0 - (now - self._times[0]) + 0.01
            if sleep_s > 0:
                time.sleep(sleep_s)
        # record after any sleep
        self._times.append(time.time())

class TokenRateLimiter:
    """Sliding-window TPM limiter (for OpenAI)."""
    def __init__(self, tpm: int):
        self.tpm = max(1, int(tpm))
        self._events = deque()  # (timestamp, tokens)
        self._sum = 0

    def _prune(self):
        now = time.time()
        window = now - 60.0
        while self._events and self._events[0][0] < window:
            t, tok = self._events.popleft()
            self._sum -= tok

    def acquire(self, tokens_needed: int):
        """Block until tokens_needed fits under TPM window."""
        tokens_needed = max(0, int(tokens_needed))
        while True:
            self._prune()
            if self._sum + tokens_needed <= self.tpm:
                return
            # wait until oldest event falls out of window
            oldest_time, _ = self._events[0]
            wait_s = 60.0 - (time.time() - oldest_time) + 0.01
            if wait_s > 0:
                time.sleep(wait_s)

    def record(self, tokens_used: int):
        tokens_used = max(0, int(tokens_used))
        self._prune()
        self._events.append((time.time(), tokens_used))
        self._sum += tokens_used

# ---------- base client ----------

class BaseClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1028):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total = 0
        self.ok = 0
        self.last_error = None
        # token accounting
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_total = 0
        # limiters (subclasses set these)
        self._rl_req: Optional[RequestRateLimiter] = None
        self._rl_tok: Optional[TokenRateLimiter] = None

    def _add_tokens(self, prompt_tokens: int, completion_tokens: int):
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        self.tokens_in += prompt_tokens
        self.tokens_out += completion_tokens
        self.tokens_total += (prompt_tokens + completion_tokens)

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

# ---------- OpenAI ----------

class OpenAIClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=1028, api_base: Optional[str] = None):
        super().__init__(model, temperature, max_tokens)
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)
        # Simple TPM limiter (OpenAI docs recommend minding TPM/RPM)
        self._rl_tok = TokenRateLimiter(int(os.getenv("OPENAI_TPM", "20000")))

    def generate(self, prompt: str) -> str:
        self.total += 1
        if self._rl_tok:
            # best-effort: prompt tokens + output cap
            self._rl_tok.acquire(_est_tokens_from_text(prompt) + int(self.max_tokens))

        try:
            msgs = [
                {"role": "system", "content": strict_rules()},
                {"role": "user",   "content": prompt},
            ]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),  # classic param name for chat.completions
                stop=[ANSWER_END],                # hard stop at your sentinel
            )

            out = (resp.choices[0].message.content or "").strip()
            if out and not out.endswith(ANSWER_END):
                out = f"{out} {ANSWER_END}"

            # usage accounting (fallback to rough estimate)
            usage = getattr(resp, "usage", None)
            in_tok  = getattr(usage, "prompt_tokens", None)     if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            if in_tok  is None: in_tok  = _est_tokens_from_text(prompt)
            if out_tok is None: out_tok = _est_tokens_from_text(out)
            self._add_tokens(in_tok, out_tok)
            if self._rl_tok: self._rl_tok.record(in_tok + out_tok)

            if out:
                self.ok += 1
            return out
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._add_tokens(_est_tokens_from_text(prompt), 0)
            return ""


# ---------- Gemini (minimal, system+prompt, single call, hard stop) ----------

class GeminiClient(BaseClient):
    """
    Uses the new Google GenAI SDK if available (`google.genai`), otherwise falls back
    to the legacy `google.generativeai`. Both paths keep the same simple behavior:
    - one system instruction (strict_rules)
    - one user prompt
    - stop at ANSWER_END
    """
    def __init__(self, model, temperature=0.0, max_tokens=1028):
        super().__init__(model, temperature, max_tokens)

        self._use_new_sdk = False
        self._client = None
        self._model_obj = None

        key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY")

        # Try the new Google GenAI SDK first (ai.google.dev)
        try:
            from google import genai as new_genai
            from google.genai.types import GenerateContentConfig
            self._use_new_sdk = True
            self._new = {
                "genai": new_genai,
                "GenerateContentConfig": GenerateContentConfig,
            }
            # New SDK uses a client object
            self._client = new_genai.Client()
        except Exception:
            # Fallback to legacy google.generativeai
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._legacy = {"genai": genai}
            # Pre-bind system instruction at model construction (supported by legacy SDK)
            self._model_obj = genai.GenerativeModel(
                model_name=model,
                system_instruction=strict_rules()
            )

        # Lightweight RPM limiter to avoid 429s
        self._rl_req = RequestRateLimiter(int(os.getenv("GEMINI_RPM", "150")))
        self._api_key = key
        self._model = model

    def _extract_text_legacy(self, resp) -> str:
        # Legacy SDK sometimes lacks .text if MAX_TOKENS or empty candidates happen.
        try:
            t = getattr(resp, "text", None)
            if t: return t
        except Exception:
            pass
        # Fallback: fish into candidates/parts
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands and getattr(cands[0], "content", None):
                parts = getattr(cands[0].content, "parts", None) or []
                if parts and getattr(parts[0], "text", None):
                    return parts[0].text
        except Exception:
            pass
        return ""

    def generate(self, prompt: str) -> str:
        from mathbode.utils import strict_rules, ANSWER_END
        self.total += 1

        try:
            if self._rl_req:
                self._rl_req.wait()

            if self._use_new_sdk:
                # New Google GenAI SDK path
                from google.genai.types import GenerateContentConfig
                cfg = GenerateContentConfig(
                    system_instruction=[strict_rules()],
                    temperature=float(self.temperature),
                    max_output_tokens=int(self.max_tokens),
                    stop_sequences=[ANSWER_END],
                )
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=cfg,
                )
                # New SDK exposes .text (method or property depending on version)
                out = ""
                try:
                    out = resp.text if isinstance(getattr(resp, "text", None), str) else resp.text()
                except Exception:
                    # Fallback: unified accessor
                    try:
                        out = resp.text
                    except Exception:
                        out = ""
            else:
                # Legacy google.generativeai path
                # Minimal call: we already set system_instruction on the model object
                resp = self._model_obj.generate_content(
                    prompt,
                    generation_config={
                        "temperature": float(self.temperature),
                        "max_output_tokens": int(self.max_tokens),
                        "stop_sequences": [ANSWER_END],
                    },
                    # Keep safety defaults; fewer surprises across models
                )
                out = self._extract_text_legacy(resp)

            out = (out or "").strip()
            if out and not out.endswith(ANSWER_END):
                out = f"{out}{ANSWER_END}"

            # Gemini SDKs don't consistently report token usage → best-effort estimate
            in_tok  = _est_tokens_from_text(prompt)
            out_tok = _est_tokens_from_text(out)
            self._add_tokens(in_tok, out_tok)

            if out:
                self.ok += 1
            return out

        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self._add_tokens(_est_tokens_from_text(prompt), 0)
            return ""


# ---------- Anthropic ----------

class AnthropicClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=512):
        super().__init__(model, temperature, max_tokens)
        from anthropic import Anthropic
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=key)
        rpm = int(os.getenv("ANTHROPIC_RPM", "50"))
        self._rl_req = RequestRateLimiter(rpm)

    def generate(self, prompt: str) -> str:
        self.total += 1
        try:
            if self._rl_req:
                self._rl_req.wait()

            msg = self.client.messages.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                system=strict_rules(),
                messages=[
                    {"role": "user", "content": "Example: 2+3"},
                    {"role": "assistant", "content": f"{ANSWER_START} 5.000000 {ANSWER_END}"},
                    {"role": "user", "content": prompt},
                ],
                stop_sequences=[ANSWER_END],               # <— stop at END tag
            )

            # pull text out
            chunks = []
            for b in getattr(msg, "content", []) or []:
                if getattr(b, "type", "") == "text":
                    chunks.append(getattr(b, "text", ""))
                elif hasattr(b, "text"):
                    chunks.append(getattr(b, "text", ""))
            out = ("".join(chunks) or "").strip()

            # usage
            u = getattr(msg, "usage", None)
            in_tok = getattr(u, "input_tokens", None) if u else None
            out_tok = getattr(u, "output_tokens", None) if u else None
            if in_tok is None:
                in_tok = _est_tokens_from_text(prompt)
            if out_tok is None:
                out_tok = _est_tokens_from_text(out)

            self._add_tokens(in_tok, out_tok)

            if out:
                self.ok += 1
            return out
        except Exception as e:
            self.last_error = str(e)
            self._add_tokens(_est_tokens_from_text(prompt), 0)
            return ""

# ---------- Together ----------

class TogetherClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=1028, api_base: Optional[str] = None):
        super().__init__(model, temperature, max_tokens)
        from together import Together
        key = os.getenv("TOGETHER_API_KEY", "")
        if not key:
            raise RuntimeError("Missing TOGETHER_API_KEY")
        self.client = Together(api_key=key, base_url=api_base) if api_base else Together(api_key=key)
        rpm = int(os.getenv("TOGETHER_RPM", "600"))
        self._rl_req = RequestRateLimiter(rpm)

    def generate(self, prompt: str) -> str:
        self.total += 1
        try:
            if self._rl_req:
                self._rl_req.wait()

            from mathbode.utils import strict_rules, ANSWER_START, ANSWER_END

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=float(self.temperature),       # 0.0 recommended
                max_tokens=int(self.max_tokens),           # 24 is a good starting point
                messages=[
                    {"role": "system", "content": strict_rules()},
                    {"role": "user", "content": "Example: 175/67"},
                    {"role": "assistant", "content": f"{ANSWER_START} 2.611940 {ANSWER_END}"},
                    {"role": "user", "content": prompt},   # prompt = just the problem text
                ],
                stop=[ANSWER_END],                         # <— key: stop exactly at END tag
                top_p=1.0,
            )


            out = (response.choices[0].message.content or "").strip()
            if not out.endswith(ANSWER_END):
                out = f"{out} {ANSWER_END}"

            # usage
            u = getattr(response, "usage", None)
            in_tok = None
            out_tok = None
            if u is not None:
                # usage may be an object or dict
                if isinstance(u, dict):
                    in_tok = u.get("prompt_tokens")
                    out_tok = u.get("completion_tokens")
                else:
                    in_tok = getattr(u, "prompt_tokens", None)
                    out_tok = getattr(u, "completion_tokens", None)

            if in_tok is None:
                in_tok = _est_tokens_from_text(prompt)
            if out_tok is None:
                out_tok = _est_tokens_from_text(out)

            self._add_tokens(in_tok, out_tok)

            if out:
                self.ok += 1
            return out
        except Exception as e:
            self.last_error = str(e)
            self._add_tokens(_est_tokens_from_text(prompt), 0)
            return ""

# ---------- factory ----------

def make_client(provider: str, model: str, temperature: float, max_tokens: int, api_base: Optional[str] = None):
    provider = provider.lower()
    if provider == 'openai':
        return OpenAIClient(model, temperature, max_tokens, api_base)
    if provider in ('google', 'gemini'):
        return GeminiClient(model, temperature, max_tokens)
    if provider == 'anthropic':
        return AnthropicClient(model, temperature, max_tokens)
    if provider == 'together':
        return TogetherClient(model, temperature, max_tokens, api_base)
    raise ValueError(f'Unknown provider: {provider}')
