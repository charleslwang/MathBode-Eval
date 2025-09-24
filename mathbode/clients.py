# mathbode/clients.py
import os, json, time
from typing import Optional, Dict, Any

class BaseClient:
    def __init__(self, model:str, temperature:float=0.0, max_tokens:int=32):
        self.model=model; self.temperature=temperature; self.max_tokens=max_tokens
        self.total=0; self.ok=0; self.last_error=None
    def generate(self, prompt:str)->str: raise NotImplementedError

class OpenAIClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=32, api_base:Optional[str]=None):
        super().__init__(model, temperature, max_tokens)
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY","")
        if not key: raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)
    def generate(self, prompt:str)->str:
        self.total+=1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            out = resp.choices[0].message.content or ""
            if out: self.ok+=1
            return out
        except Exception as e:
            self.last_error=str(e); return ""

class GeminiClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=32):
        super().__init__(model, temperature, max_tokens)
        import google.generativeai as genai
        key = os.getenv("GOOGLE_API_KEY", "")
        if not key: 
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=key)
        self.model_obj = genai.GenerativeModel(model_name=model)
    def generate(self, prompt:str)->str:
        self.total+=1
        try:
            resp=self.model_obj.generate_content(
                prompt,
                generation_config={"temperature":float(self.temperature),
                                 "max_output_tokens":int(self.max_tokens)}
            )
            text=getattr(resp,"text",None)
            if not text and getattr(resp,"candidates",None):
                cand=resp.candidates[0]; parts=getattr(cand,"content",{}).get("parts",[])
                text="".join(getattr(p,"text","") for p in parts)
            if text: self.ok+=1
            return text or ""
        except Exception as e:
            self.last_error=str(e); return ""

class AnthropicClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=64):
        super().__init__(model, temperature, max_tokens)
        from anthropic import Anthropic
        key=os.getenv("ANTHROPIC_API_KEY","")
        if not key: raise RuntimeError("Missing ANTHROPIC_API_KEY")
        self.client=Anthropic(api_key=key)
    def generate(self, prompt:str)->str:
        self.total+=1
        try:
            msg=self.client.messages.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                messages=[{"role":"user","content":prompt}]
            )
            text="".join(b.text for b in msg.content if getattr(b,"type","")=="text")
            if not text:
                text="".join(getattr(b, "text", "") for b in msg.content if hasattr(b, "text"))
            if text: self.ok+=1
            return text or ""
        except Exception as e:
            self.last_error=str(e); return ""

class TogetherClient(BaseClient):
    def __init__(self, model, temperature=0.0, max_tokens=32, api_base:Optional[str]=None):
        super().__init__(model, temperature, max_tokens)
        from together import Together
        key = os.getenv("TOGETHER_API_KEY", "")
        if not key:
            raise RuntimeError("Missing TOGETHER_API_KEY")
        self.client = Together(api_key=key, base_url=api_base) if api_base else Together(api_key=key)
    
    def generate(self, prompt: str) -> str:
        self.total += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            out = response.choices[0].message.content or ""
            if out: self.ok += 1
            return out
        except Exception as e:
            self.last_error = str(e)
            return ""

def make_client(provider:str, model:str, temperature:float, max_tokens:int, api_base:Optional[str]=None):
    if provider=='openai': return OpenAIClient(model, temperature, max_tokens, api_base)
    if provider=='google': return GeminiClient(model, temperature, max_tokens)
    if provider=='anthropic': return AnthropicClient(model, temperature, max_tokens)
    if provider=='together': return TogetherClient(model, temperature, max_tokens, api_base)
    raise ValueError(f'Unknown provider: {provider}')
