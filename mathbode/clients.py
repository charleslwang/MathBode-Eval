# mathbode/clients.py
import os, time
from collections import deque
from typing import Optional

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
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 32):
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
    def __init__(self, model, temperature=0.0, max_tokens=32, api_base: Optional[str] = None):
        super().__init__(model, temperature, max_tokens)
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)
        # token limit only (per user spec)
        tpm = int(os.getenv("OPENAI_TPM", "20000"))
        self._rl_tok = TokenRateLimiter(tpm)

    def generate(self, prompt: str) -> str:
        self.total += 1

        # acquire TPM capacity conservatively: prompt_est + max_tokens
        if self._rl_tok:
            est_in = _est_tokens_from_text(prompt)
            self._rl_tok.acquire(est_in + int(self.max_tokens))

        try:
            if hasattr(self, "_rl_req") and self._rl_req:
                self._rl_req.wait()

            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
            out = (resp.choices[0].message.content or "").strip()

            # usage fields
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None

            if in_tok is None:
                in_tok = _est_tokens_from_text(prompt)
            if out_tok is None:
                out_tok = _est_tokens_from_text(out)

            self._add_tokens(in_tok, out_tok)
            if self._rl_tok:
                # record actual tokens (not planned)
                self._rl_tok.record(in_tok + out_tok)

            if out:
                self.ok += 1
            return out
        except Exception as e:
            self.last_error = str(e)
            # on failure, don't record tokens against TPM (service may not have counted it),
            # but keep local prompt estimate in totals for transparency:
            if self._rl_tok:
                # nothing recorded to limiter (since request may not have consumed)
                pass
            self._add_tokens(_est_tokens_from_text(prompt), 0)
            return ""

# ---------- Gemini ----------

class GeminiClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=32):
        super().__init__(model, temperature, max_tokens)
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=key)
        self.model_obj = genai.GenerativeModel(model_name=model)
        rpm = int(os.getenv("GEMINI_RPM", "150"))
        self._rl_req = RequestRateLimiter(rpm)

    def generate(self, prompt: str) -> str:
        self.total += 1
        try:
            if self._rl_req:
                self._rl_req.wait()

            resp = self.model_obj.generate_content(
                prompt,
                generation_config={
                    "temperature": float(self.temperature),
                    "max_output_tokens": int(self.max_tokens),
                },
            )
            # Extract text robustly
            text = getattr(resp, "text", None)
            if not text and getattr(resp, "candidates", None):
                cand = resp.candidates[0]
                parts = getattr(getattr(cand, "content", {}), "parts", []) or getattr(cand, "content", {}).get("parts", [])
                text = "".join(getattr(p, "text", "") for p in parts)
            out = (text or "").strip()

            # usage
            um = getattr(resp, "usage_metadata", None)
            in_tok = getattr(um, "prompt_token_count", None) if um else None
            out_tok = getattr(um, "candidates_token_count", None) if um else None
            total_tok = getattr(um, "total_token_count", None) if um else None
            if out_tok is None and (in_tok is not None and total_tok is not None):
                out_tok = max(0, int(total_tok) - int(in_tok))

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

# ---------- Anthropic ----------

class AnthropicClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=64):
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
                messages=[{"role": "user", "content": prompt}],
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
    def __init__(self, model, temperature=0.0, max_tokens=32, api_base: Optional[str] = None):
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

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
            out = (response.choices[0].message.content or "").strip()

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
