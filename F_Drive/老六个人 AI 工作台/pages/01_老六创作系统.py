from __future__ import annotations

import asyncio
import sys
import base64
import json
import re
import concurrent.futures
import os
import subprocess
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
try:
    from playwright.async_api import async_playwright
except Exception:
    async_playwright = None

from modules.obsidian_rw import ObsidianStore
from modules.skill_distiller import DistillConfig, SkillDistiller, _default_input_roots

DEFAULT_OLLAMA_MODEL = "qwen35:9b-q4_0"

st.set_page_config(page_title="老六创作系统", layout="wide")
st.write("Debug: 阶段 0（import + set_page_config）已通过")

GenerationConfig = None
PytorchEngineConfig = None
TurbomindEngineConfig = None
pipeline = None


@dataclass(frozen=True)
class WebProvider:
    key: str
    title: str
    url: str


WEB_PROVIDERS: list[WebProvider] = [
    WebProvider(key="gemini", title="Gemini", url="https://gemini.google.com/app"),
    WebProvider(key="grok", title="Grok", url="https://grok.com/"),
    WebProvider(key="doubao", title="豆包", url="https://www.doubao.com/"),
]


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _run_async(coro):
    if sys.platform.startswith("win"):
        try:
            loop = asyncio.ProactorEventLoop()
        except Exception:
            loop = asyncio.new_event_loop()
    else:
        loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            asyncio.set_event_loop(None)
            loop.close()
        except Exception:
            pass


async def _await_or_stop(coro, timeout_s: float, stop_event: threading.Event | None = None):
    main_task = asyncio.create_task(coro)
    stop_task = None
    try:
        if stop_event is None:
            return await asyncio.wait_for(main_task, timeout=timeout_s)
        stop_task = asyncio.create_task(asyncio.to_thread(stop_event.wait, timeout_s))
        done, pending = await asyncio.wait({main_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and stop_event.is_set():
            try:
                main_task.cancel()
            except Exception:
                pass
        if main_task in done:
            return await asyncio.wait_for(main_task, timeout=0.1)
        raise asyncio.TimeoutError()
    finally:
        if stop_task is not None:
            try:
                stop_task.cancel()
            except Exception:
                pass


def _call_blocking_with_stop(fn, stop_event: threading.Event) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        while True:
            if stop_event.is_set():
                raise RuntimeError("用户手动停止")
            try:
                return fut.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                continue


async def _poll_visible(page, selectors: list[str], timeout_s: float, step_s: float, stop_event: threading.Event | None = None):
    start = time.time()
    last_err = ""
    while time.time() - start < timeout_s:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.is_visible(timeout=500):
                    return loc
            except Exception as e:
                last_err = str(e)
                continue
        await asyncio.sleep(step_s)
    raise TimeoutError(last_err or "等待元素超时")


def _ensure_cuda_path_for_turbomind() -> None:
    try:
        import os
        if os.name != "nt":
            return
        if os.environ.get("CUDA_PATH"):
            return
        import shutil
        from pathlib import Path
        import torch
        root = Path(__file__).resolve().parent.parent
        stub = root / "llm_engine" / "cuda_stub"
        bin_dir = stub / "bin"
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if not torch_lib.is_dir():
            return
        bin_dir.mkdir(parents=True, exist_ok=True)
        for dll in torch_lib.glob("*.dll"):
            dst = bin_dir / dll.name
            if dst.exists():
                try:
                    if dst.stat().st_size == dll.stat().st_size:
                        continue
                except Exception:
                    pass
            try:
                shutil.copy2(dll, dst)
            except Exception:
                pass
        os.environ["CUDA_PATH"] = str(stub)
    except Exception:
        return


class LocalLMDeploy:
    def __init__(
        self,
        model_dir: str,
        cache_max_entry_count: float,
        session_len: int,
        offload: bool,
        conda_env: str = "lmdeploy-qwen35-27b-4bit",
        backend: str = "ollama",
        gpu_memory_utilization: float = 0.85,
        quant_policy: int = 4,
        flash_attn: bool = True,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_num_gpu: int = 999,
        ollama_num_batch: int = 16,
    ):
        self.model_dir = model_dir
        self.cache_max_entry_count = cache_max_entry_count
        self.session_len = session_len
        self.offload = offload
        self.conda_env = conda_env
        self.backend = (backend or "ollama").strip().lower()
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.quant_policy = int(quant_policy)
        self.flash_attn = bool(flash_attn)
        self.ollama_model = (ollama_model or "").strip() or DEFAULT_OLLAMA_MODEL
        self.ollama_num_gpu = int(ollama_num_gpu)
        self.ollama_num_batch = int(ollama_num_batch)
        self._pipe = None
        self._hf_model = None
        self._hf_tokenizer = None

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cpu_blocks = 1024 if self.offload else 0
        if self.backend != "pytorch":
            raise RuntimeError("当前加载器未启用 LMDeploy(Pytorch) 后端")
        global GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
        if PytorchEngineConfig is None or pipeline is None:
            try:
                from lmdeploy import GenerationConfig as _GC
                from lmdeploy import PytorchEngineConfig as _PEC
                from lmdeploy import TurbomindEngineConfig as _TEC
                from lmdeploy import pipeline as _pl

                GenerationConfig = _GC
                PytorchEngineConfig = _PEC
                TurbomindEngineConfig = _TEC
                pipeline = _pl
            except Exception:
                raise RuntimeError("lmdeploy(pytorch) 未正确安装或导入失败")
        try:
            import triton
        except Exception as e:
            raise RuntimeError(f"triton 未就绪：{type(e).__name__}: {e}")
        try:
            backend = PytorchEngineConfig(
                session_len=self.session_len,
                cache_max_entry_count=float(self.gpu_memory_utilization),
                enable_prefix_caching=True,
                num_cpu_blocks=cpu_blocks,
                quant_policy=int(self.quant_policy),
            )
            self._pipe = pipeline(self.model_dir, backend_config=backend)
            return self._pipe
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if "Unsupported quant method: gptq" in msg or "quant method: gptq" in msg:
                raise RuntimeError(
                    "LMDeploy(Pytorch) 当前不支持 GPTQ 权重量化模型（quant_method=gptq）。"
                    "你的目录是 GPTQ（Qwen3.5-27B-GPTQ-Int4），这条路跑不通。"
                    "务实方案：用 Ollama 拉取 qwen3.5:27b-q4_K_M（GGUF）来实现 27B+64K+CPU/GPU 混合动力。"
                )
            raise

    def _ensure_hf(self):
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return
        try:
            import transformers.modeling_utils as mu
            if not hasattr(mu, "no_init_weights"):
                from contextlib import contextmanager
                @contextmanager
                def no_init_weights(_enable: bool = True):
                    yield
                setattr(mu, "no_init_weights", no_init_weights)
        except Exception:
            pass
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True, use_fast=False)
        device_map = None
        max_memory = None
        if self.offload:
            device_map = "auto"
            max_memory = {0: "10GiB", "cpu": "28GiB"}
        m = AutoGPTQForCausalLM.from_quantized(
            self.model_dir,
            device_map=device_map,
            max_memory=max_memory,
            use_triton=False,
            inject_fused_attention=False,
            inject_fused_mlp=False,
            trust_remote_code=True,
            use_safetensors=True,
        )
        try:
            m.eval()
        except Exception:
            pass
        self._hf_model = m
        self._hf_tokenizer = tok

    def _generate_hf(self, prompt: str, system: str, options: dict[str, Any] | None, stop_event: threading.Event) -> str:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        self._ensure_hf()
        import torch
        temp = float((options or {}).get("temperature", 0.2))
        top_p = float((options or {}).get("top_p", 0.9))
        max_new = int((options or {}).get("num_predict", 900))
        q = (prompt or "").strip()
        sys = (system or "").strip()
        tok = self._hf_tokenizer
        model = self._hf_model
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": q})
        if hasattr(tok, "apply_chat_template"):
            input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            text = (sys + "\n\n" + q).strip() if sys else q
            input_ids = tok(text, return_tensors="pt").input_ids
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        input_ids = input_ids.to(device)
        do_sample = temp > 1e-6
        gen_kwargs: dict[str, Any] = {"input_ids": input_ids, "do_sample": do_sample, "max_new_tokens": max_new}
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = top_p
        eos_id = getattr(tok, "eos_token_id", None)
        if isinstance(eos_id, int):
            gen_kwargs["eos_token_id"] = eos_id
            gen_kwargs["pad_token_id"] = eos_id
        gen_ids = model.generate(**gen_kwargs)
        out_ids = gen_ids[0][input_ids.shape[-1] :]
        return tok.decode(out_ids, skip_special_tokens=True).strip()


    def generate(self, prompt: str, system: str, options: dict[str, Any] | None, stop_event: threading.Event) -> str:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        if self.backend == "ollama":
            from modules.ollama_client import OllamaClient
            cli = OllamaClient()
            temp = float((options or {}).get("temperature", 0.2))
            max_new = int((options or {}).get("num_predict", 900))
            q_short = len((prompt or "").strip()) <= 64 and len((system or "").strip()) == 0 and max_new <= 16
            opts: dict[str, Any] = {"temperature": temp, "num_predict": max_new}
            if not q_short:
                opts["num_ctx"] = int(self.session_len)
                opts["num_gpu"] = int(self.ollama_num_gpu)
                opts["num_batch"] = int(self.ollama_num_batch)
            try:
                return cli.generate(prompt=(prompt or ""), system=(system or ""), options=opts, model=self.ollama_model)
            except Exception as e:
                msg = str(e)
                if "模型不存在" in msg or "model" in msg.lower() and "not found" in msg.lower():
                    return cli.generate(prompt=(prompt or ""), system=(system or ""), options=opts, model=DEFAULT_OLLAMA_MODEL)
                raise
        if pipeline is None:
            return self._generate_via_conda(prompt=prompt, system=system, options=options, stop_event=stop_event)
        temp = float((options or {}).get("temperature", 0.2))
        max_new = int((options or {}).get("num_predict", 900))
        gen_cfg = None
        if GenerationConfig is not None:
            gen_cfg = GenerationConfig(temperature=temp, max_new_tokens=max_new)
        q = (prompt or "").strip()
        sys = (system or "").strip()
        text = (sys + "\n\n" + q).strip() if sys else q
        pipe = self._ensure_pipe()
        out = pipe(text, gen_config=gen_cfg) if gen_cfg is not None else pipe(text)
        try:
            if hasattr(out, "text"):
                return (out.text or "").strip()
        except Exception:
            pass
        if isinstance(out, list) and out:
            x = out[0]
            try:
                if hasattr(x, "text"):
                    return (x.text or "").strip()
            except Exception:
                pass
            try:
                return str(x).strip()
            except Exception:
                return ""
        return (str(out) if out is not None else "").strip()

    def _generate_via_conda(self, prompt: str, system: str, options: dict[str, Any] | None, stop_event: threading.Event) -> str:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        if self.backend == "ollama":
            from modules.ollama_client import OllamaClient
            cli = OllamaClient()
            temp = float((options or {}).get("temperature", 0.2))
            max_new = int((options or {}).get("num_predict", 900))
            q_short = len((prompt or "").strip()) <= 64 and len((system or "").strip()) == 0 and max_new <= 16
            opts: dict[str, Any] = {"temperature": temp, "num_predict": max_new}
            if not q_short:
                opts["num_ctx"] = int(self.session_len)
                opts["num_gpu"] = int(self.ollama_num_gpu)
                opts["num_batch"] = int(self.ollama_num_batch)
            try:
                return cli.generate(prompt=(prompt or ""), system=(system or ""), options=opts, model=self.ollama_model)
            except Exception as e:
                msg = str(e)
                if "模型不存在" in msg or "model" in msg.lower() and "not found" in msg.lower():
                    return cli.generate(prompt=(prompt or ""), system=(system or ""), options=opts, model=DEFAULT_OLLAMA_MODEL)
                raise
        payload = {
            "model_dir": self.model_dir,
            "cache_max_entry_count": float(self.gpu_memory_utilization),
            "session_len": self.session_len,
            "offload": bool(self.offload),
            "backend": self.backend,
            "quant_policy": int(self.quant_policy),
            "prompt": (prompt or ""),
            "system": (system or ""),
            "temperature": float((options or {}).get("temperature", 0.2)),
            "max_new_tokens": int((options or {}).get("num_predict", 900)),
        }
        b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
        code = (
            "import base64,json,sys\n"
            "import os\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')\n"
            "from lmdeploy import pipeline,GenerationConfig,TurbomindEngineConfig,PytorchEngineConfig\n"
            "d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))\n"
            "cpu_blocks=1024 if d.get('offload') else 0\n"
            "if (d.get('backend') or '').lower()!='pytorch':\n"
            "  raise RuntimeError('当前加载器已强制使用 pytorch 后端')\n"
            "import triton\n"
            "backend=PytorchEngineConfig(session_len=int(d['session_len']),cache_max_entry_count=float(d['cache_max_entry_count']),enable_prefix_caching=True,num_cpu_blocks=cpu_blocks,quant_policy=int(d.get('quant_policy') or 0))\n"
            "pipe=pipeline(d['model_dir'],backend_config=backend)\n"
            "text=(d.get('system','').strip()+'\\n\\n'+d.get('prompt','').strip()).strip() if d.get('system','').strip() else d.get('prompt','').strip()\n"
            "cfg=GenerationConfig(temperature=float(d.get('temperature',0.2)),max_new_tokens=int(d.get('max_new_tokens',900)))\n"
            "out=pipe(text,gen_config=cfg)\n"
            "r=''\n"
            "try:\n"
            "  r=(out.text or '') if hasattr(out,'text') else ''\n"
            "except Exception:\n"
            "  r=''\n"
            "if not r and isinstance(out,list) and out:\n"
            "  x=out[0]\n"
            "  try:\n"
            "    r=(x.text or '') if hasattr(x,'text') else ''\n"
            "  except Exception:\n"
            "    r=''\n"
            "print((r or str(out)).strip())\n"
        )
        cmd = ["conda", "--no-plugins", "run", "-n", self.conda_env, "python", "-c", code, b64]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "conda run 失败").strip())
        return (p.stdout or "").strip()


def _qwen_generate_stream_with_stop(
    client: LocalLMDeploy,
    prompt: str,
    system: str,
    options: dict[str, Any] | None,
    stop_event: threading.Event,
) -> str:
    return client.generate(prompt=prompt, system=system, options=options, stop_event=stop_event)


def _cdp_ready(cdp_url: str, timeout: int = 2) -> tuple[bool, str]:
    try:
        r = requests.get(cdp_url.rstrip("/") + "/json/version", timeout=timeout)
        r.raise_for_status()
        d = r.json()
        return True, (d.get("webSocketDebuggerUrl") or "")
    except Exception as e:
        return False, str(e)


async def _pw_connect_ctx(cdp_url: str, stop_event: threading.Event | None = None):
    if async_playwright is None:
        raise RuntimeError("未安装 playwright：请在运行 Streamlit 的 Python 环境里安装 playwright。")
    p = await _await_or_stop(async_playwright().start(), 15.0, stop_event)
    browser = await _await_or_stop(p.chromium.connect_over_cdp(cdp_url), 20.0, stop_event)
    if not browser.contexts:
        await p.stop()
        raise RuntimeError("未发现可用的 Chrome Context")
    ctx = browser.contexts[0]
    try:
        targets = ("gemini.google.com", "grok.com", "doubao.com")
        best = None
        best_score = -1
        for c in browser.contexts:
            try:
                pages = list(c.pages)
            except Exception:
                pages = []
            score = len(pages)
            try:
                for pg in pages[:40]:
                    try:
                        url = (pg.url or "").lower()
                    except Exception:
                        url = ""
                    if any(t in url for t in targets):
                        score += 100
                        break
            except Exception:
                pass
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            ctx = best
    except Exception:
        ctx = browser.contexts[0]
    return p, ctx


async def _cleanup_old_ai_pages(ctx, stop_event: threading.Event, job: dict[str, Any], lock: threading.Lock) -> None:
    try:
        pages = list(ctx.pages)
    except Exception:
        pages = []
    if not pages:
        try:
            await _await_or_stop(ctx.new_page(), 10.0, stop_event)
        except Exception:
            return
        return

    keep: dict[str, Any] = {}
    dup: list[Any] = []
    for pg in pages:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            url = (pg.url or "").lower()
        except Exception:
            url = ""
        host = ""
        if "gemini.google.com" in url:
            host = "gemini.google.com"
        elif "grok.com" in url:
            host = "grok.com"
        elif "doubao.com" in url:
            host = "doubao.com"
        if not host:
            continue
        if host in keep:
            dup.append(pg)
        else:
            keep[host] = pg

    if not dup:
        return

    closed = 0
    for pg in dup:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            await _await_or_stop(pg.close(), 6.0, stop_event)
            closed += 1
        except Exception:
            pass
    if closed:
        _job_append_log(job, lock, "SYSTEM", "清理旧 AI 窗口", f"已关闭 {closed} 个（每个平台保留 1 个）")

async def _close_ai_pages_keep_first(ctx, stop_event: threading.Event, job: dict[str, Any], lock: threading.Lock) -> None:
    try:
        pages = list(ctx.pages)
    except Exception:
        pages = []
    if not pages:
        return
    keep = pages[0]
    closed = 0
    for pg in pages:
        if pg is keep:
            continue
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            await _await_or_stop(pg.close(), 6.0, stop_event)
            closed += 1
        except Exception:
            pass
    if closed:
        _job_append_log(job, lock, "SYSTEM", "关闭 9222 窗口", f"已关闭 {closed} 个（保留第一个标签页）")


def _provider_host(provider_key: str) -> str:
    if provider_key == "gemini":
        return "gemini.google.com"
    if provider_key == "grok":
        return "grok.com"
    if provider_key == "doubao":
        return "doubao.com"
    return ""


async def _mark_script_page(page, run_tag: str, provider_key: str, round_tag: str, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    tag = f"[LAOLIU_RUN={run_tag}][{provider_key}][{round_tag}] "
    try:
        await _await_or_stop(page.evaluate("tag => { document.title = tag + (document.title || ''); }", tag), 3.0, stop_event)
    except Exception:
        pass
    try:
        await _await_or_stop(page.evaluate("tag => { try { window.name = tag + (window.name || ''); } catch (e) {} }", tag), 3.0, stop_event)
    except Exception:
        pass


async def _cleanup_script_pages(ctx, stop_event: threading.Event, job: dict[str, Any], lock: threading.Lock) -> None:
    try:
        pages = list(ctx.pages)
    except Exception:
        pages = []
    if not pages:
        return
    targets = ("gemini.google.com", "grok.com", "doubao.com")
    to_close = []
    for pg in pages:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            url = (pg.url or "").lower()
        except Exception:
            url = ""
        if any(t in url for t in targets):
            to_close.append(pg)

    if not to_close:
        return

    seen = set()
    uniq = []
    for pg in to_close:
        k = id(pg)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(pg)
    to_close = uniq
    closed = 0
    for pg in to_close:
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            await _await_or_stop(pg.close(), 6.0, stop_event)
            closed += 1
        except Exception:
            pass
    if closed:
        _job_append_log(job, lock, "SYSTEM", "清理旧脚本窗口", f"已关闭 {closed} 个（全部关闭）")

    try:
        pages2 = list(ctx.pages)
    except Exception:
        pages2 = []
    if not pages2:
        try:
            await _await_or_stop(ctx.new_page(), 10.0, stop_event)
        except Exception:
            pass


async def _find_provider_page(ctx, provider_key: str):
    host = _provider_host(provider_key)
    if not host:
        return None
    try:
        pages = list(ctx.pages)
    except Exception:
        pages = []
    for pg in pages:
        try:
            url = (pg.url or "").lower()
        except Exception:
            url = ""
        if host in url:
            return pg
    return None


async def _get_or_open_provider_page(ctx, provider: WebProvider, run_date: str, stop_event: threading.Event, job: dict[str, Any], lock: threading.Lock):
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    page = await _find_provider_page(ctx, provider.key)
    if page is None:
        page = await _open_provider_tab(ctx, provider, run_date, stop_event, job, lock)
        return page
    try:
        await page.bring_to_front()
    except Exception:
        pass
    try:
        cur = (page.url or "").lower()
        if provider.url and provider.url.split("/")[2] not in cur:
            await _await_or_stop(page.goto(provider.url, wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
    except Exception:
        try:
            await _await_or_stop(page.goto(provider.url, wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
        except Exception:
            pass
    try:
        await _dismiss_common_dialogs(page)
    except Exception:
        pass
    if provider.key in ("gemini", "doubao"):
        await _reset_provider_in_place(page, provider.key, stop_event)
    if provider.key == "grok":
        await _ensure_grok_ready(page, stop_event)
        try:
            await _ensure_grok_new_chat(page, stop_event)
        except Exception:
            pass
        if "subscribe" in (page.url or "").lower():
            try:
                await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            await _ensure_grok_ready(page, stop_event)
    try:
        await _click_mode_if_present(page, provider.key)
    except Exception:
        pass
    await _shot_step(page, provider.key, run_date, "reuse", job, lock, stop_event)
    return page


async def _pw_get_current_page(ctx):
    try:
        pages = list(ctx.pages)
    except Exception:
        pages = []
    if not pages:
        raise RuntimeError("未发现可用的浏览器页面：请先在该 Chrome 中打开任意网页")
    return pages[-1]


async def _goto_same_page(ctx, url: str):
    page = await _pw_get_current_page(ctx)
    try:
        await page.bring_to_front()
    except Exception:
        pass
    try:
        if not (page.url or "").startswith(url):
            await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    except Exception:
        await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
    return page


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _trace_dir(run_date: str, provider_key: str) -> Path:
    p = _root_dir() / "logs" / "ui_trace" / run_date / provider_key
    p.mkdir(parents=True, exist_ok=True)
    return p


def _job_state_init() -> None:
    if "fact_job" not in st.session_state:
        st.session_state.fact_job = {
            "running": False,
            "progress": 0.0,
            "status": "",
            "logs": [],
            "shots": [],
            "result": {},
            "error": "",
            "run_date": "",
        }
    if "fact_job_lock" not in st.session_state:
        st.session_state.fact_job_lock = threading.Lock()
    if "fact_job_stop" not in st.session_state:
        st.session_state.fact_job_stop = threading.Event()


def _job_update(job: dict[str, Any], lock: threading.Lock, **kwargs) -> None:
    with lock:
        for k, v in kwargs.items():
            job[k] = v


def _merge_unique(base: list[str], extra: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in base:
        s = (x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    for x in extra:
        s = (x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_monitor_targets() -> dict[str, list[str]]:
    p = _root_dir() / "config" / "monitor_config.json"
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        t = data.get("targets") if isinstance(data, dict) else None
        if isinstance(t, dict):
            out: dict[str, list[str]] = {}
            for k, v in t.items():
                if isinstance(v, list):
                    out[str(k)] = [str(x) for x in v if str(x).strip()]
            return out
    except Exception:
        pass
    return {}


def _job_append_log(job: dict[str, Any], lock: threading.Lock, provider: str, action: str, result: str) -> None:
    with lock:
        job["logs"].append(f"{_now_ts()} | {provider} | {action} | {result}")


def _job_append_shot(job: dict[str, Any], lock: threading.Lock, path: str) -> None:
    with lock:
        job["shots"].append(path)


def _sanitize_grok_question(text: str) -> str:
    t = (text or "").strip()
    banned = [
        "提示词",
        "prompt",
        "imagine",
        "绘图",
        "画图",
        "生成提示词",
        "生成 prompt",
    ]
    for w in banned:
        t = re.sub(re.escape(w), "", t, flags=re.I)
    prefix = "这是问答任务：请按条目直接回答，并为每条给出可验证依据（链接/标题/时间点/关键词）。\n\n"
    return prefix + t


def _sanitize_web_question(provider_key: str, anchor: str, text: str) -> str:
    t = (text or "").strip()
    banned = [
        "生成提示词",
        "提示词",
        "prompt",
        "Prompt",
        "请生成",
        "生成一份提示词",
        "输出提示词",
        "提问模板",
    ]
    for w in banned:
        t = re.sub(re.escape(w), "", t, flags=re.I)
    head = "\n".join(
        [
            "这是问答任务：请直接给出答案，不要生成提示词/提问模板/教学。",
            "请忽略/清空此前所有对话上下文，只针对本次主播回答。",
            "不要输出任务拆解/步骤/提纲/清单；不要把问题改写成“追问列表”。",
            "硬性要求：全程中文输出；不要英文；不要代码块。",
            f"主播专名必须原样写成：{anchor}（禁止同音字/错别字）。",
            "每一条观点必须给【依据】（链接/标题/时间点/关键词/可检索描述）。",
            "",
        ]
    )
    if provider_key == "doubao":
        head = head + "如果不确定请直接说不确定，不要编造。\n"
    if provider_key == "grok":
        return head + _sanitize_grok_question(t)
    return head + t


def _fixed_four_questions(anchor: str) -> str:
    a = (anchor or "").strip()
    return "\n".join(
        [
            f"{a}有什么经典梗？",
            "他的铁粉对他的绰号是什么？黑粉对他的绰号是什么？",
            "粉丝喜欢他的哪些点？",
            "近半个月来，他有什么新的节奏或吸引人的地方？",
        ]
    ).strip()


PROMPT_PROFILES: dict[str, dict[str, Any]] = {
    "四问事实核验": {
        "title": "四问事实核验",
        "questions": [
            "{anchor}有什么经典梗？",
            "他的铁粉对他的绰号是什么？黑粉对他的绰号是什么？",
            "粉丝喜欢他的哪些点？",
            "近半个月来，他有什么新的节奏或吸引人的地方？",
        ],
    },
    "粉丝团与关系网": {
        "title": "粉丝团与关系网",
        "questions": [
            "{anchor}主播的粉丝团称呼是什么以及粉丝团数字是什么？",
            "{anchor}主播旗下有什么其他主播吗，有哪些徒弟？",
            "经常在直播讲{anchor}主播八卦的主播有哪些？",
            "{anchor}主播直播名场面 / 经典翻车瞬间有哪些？",
            "{anchor}主播与其他主播的经典互动、CP 或对立关系是什么？",
            "{anchor}主播的标志性口头禅、专属 BGM、招牌动作是什么？",
        ],
    },
}


def _build_profile_questions(profile_id: str, anchor: str) -> str:
    pid = (profile_id or "").strip() or "四问事实核验"
    a = (anchor or "").strip()
    prof = PROMPT_PROFILES.get(pid) or PROMPT_PROFILES["四问事实核验"]
    qs = prof.get("questions") or []
    lines = []
    for q in qs:
        s = (q or "").replace("{anchor}", a).strip()
        if s:
            lines.append(s)
    return "\n".join(lines).strip()


def _extract_provider_section(text: str, provider_key: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    labels_map = {
        "gemini": ["Gemini"],
        "grok": ["Grok"],
        "doubao": ["豆包", "Doubao", "Duobao"],
    }
    labels = labels_map.get(provider_key, [provider_key])
    pats: list[str] = []
    for label in labels:
        pats.extend(
            [
                rf"^##+\s*发给\s*{re.escape(label)}\s*的二次追问\s*[:：]?\s*$",
                rf"^##+\s*给\s*{re.escape(label)}\s*的二次追问\s*[:：]?\s*$",
                rf"^##+\s*{re.escape(label)}\s*二次追问\s*[:：]?\s*$",
            ]
        )
    lines = t.splitlines()
    starts: list[int] = []
    for i, line in enumerate(lines):
        for p in pats:
            if re.match(p, line.strip(), flags=re.I):
                starts.append(i)
                break
    if not starts:
        return ""
    s = starts[0] + 1
    e = len(lines)
    for j in range(s, len(lines)):
        if re.match(r"^##+\s*", lines[j].strip()):
            e = j
            break
    out = "\n".join(lines[s:e]).strip()
    return out.strip()


def _qwen_repair_followup_stream(client: LocalLMDeploy, anchor: str, bad_text: str, stop_event: threading.Event) -> str:
    sys = f"你是嘴毒但不辱骂的严格格式修复器。必须全程中文输出，禁止英文。主播名必须原样：{anchor}。"
    p = "\n".join(
        [
            f"主播：{anchor}",
            "下面是一段二次追问文本，但格式不稳定/无法被程序正确拆分。",
            "你的任务：把它改写成可被程序可靠拆分的三段追问。",
            "硬性要求：",
            "1) 只能输出以下三个标题（各出现一次，不能改字）：",
            "### 发给 Gemini 的二次追问",
            "### 发给 Grok 的二次追问",
            "### 发给 豆包 的二次追问",
            "2) 除标题与追问清单外，不要输出任何解释、总结、前言、差异点清单。",
            "3) 每段内容必须是直接向该AI提问的口吻，只追问差异/缺失/无依据之处，要求对方补【依据】。",
            "4) 全文纯中文；不要英文；不要代码块。",
            "",
            "原始文本：",
            bad_text or "(空)",
        ]
    )
    return _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.1, "num_predict": 900}, stop_event)


def _fallback_followup_for_provider(provider_key: str, anchor: str, r1: dict[str, str]) -> str:
    other_keys = [k for k in ["gemini", "grok", "doubao"] if k != provider_key]
    a = (r1.get(other_keys[0], "") or "").strip()
    b = (r1.get(other_keys[1], "") or "").strip()
    a = a[:1200]
    b = b[:1200]
    name_map = {"gemini": "Gemini", "grok": "Grok", "doubao": "豆包"}
    me = name_map.get(provider_key, provider_key)
    o1 = name_map.get(other_keys[0], other_keys[0])
    o2 = name_map.get(other_keys[1], other_keys[1])
    return "\n".join(
        [
            f"这是二次核验任务：你是 {me}。请只针对你上一轮回答中与 {o1}/{o2} 不一致或缺失的地方补充核验。",
            f"主播：{anchor}（必须原样，不得改写）。",
            "硬性要求：全文中文；不要英文；不要代码块；不要总结差异；只输出你对这些差异点的直接回答与【依据】。",
            f"纠错要求：若对照材料出现非“{anchor}”的其它主播名，这是上下文污染，先明确纠正：本次只讨论“{anchor}”。",
            "",
            f"对照材料（来自 {o1}）：",
            a or "(空)",
            "",
            f"对照材料（来自 {o2}）：",
            b or "(空)",
            "",
            "请完成：",
            "1) 列出你与对照材料不一致/缺失的点（逐条）。",
            "2) 对每条给出你最终确认的说法，并提供【依据】（链接/标题/时间点/关键词）。",
            "3) 若无法核验，请标注【待核实】并给出具体下一步核验路径。",
        ]
    )


def _cross_ref_block(provider_key: str, r1: dict[str, str]) -> str:
    other_keys = [k for k in ["gemini", "grok", "doubao"] if k != provider_key]
    name_map = {"gemini": "Gemini", "grok": "Grok", "doubao": "豆包"}
    a = (r1.get(other_keys[0], "") or "").strip()[:800]
    b = (r1.get(other_keys[1], "") or "").strip()[:800]
    o1 = name_map.get(other_keys[0], other_keys[0])
    o2 = name_map.get(other_keys[1], other_keys[1])
    return "\n".join(
        [
            "",
            "对照材料（仅用于核验差异，不要复述原文）：",
            f"- {o1}：{a or '(空)'}",
            f"- {o2}：{b or '(空)'}",
            "",
        ]
    )


async def _goto_new_tab(ctx, url: str, stop_event: threading.Event | None = None):
    page = await _await_or_stop(ctx.new_page(), 10.0, stop_event)
    try:
        await page.bring_to_front()
    except Exception:
        pass
    await _await_or_stop(page.goto(url, wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
    return page


async def _ensure_doubao_chat(page, stop_event: threading.Event):
    try:
        await _dismiss_common_dialogs(page)
    except Exception:
        pass
    try:
        btn = page.get_by_text("新对话").first
        if await btn.is_visible(timeout=800):
            await _await_or_stop(btn.click(timeout=1500), 3.0, stop_event)
            return
    except Exception:
        pass
    need_reset = False
    try:
        tip = page.get_by_text(re.compile(r"(描述你想要的图片|图片生成|图像生成|Seedream|生图)", re.I))
        if await tip.count() > 0:
            need_reset = True
    except Exception:
        pass
    if not need_reset:
        return
    try:
        btn = page.get_by_text("新对话").first
        if await btn.is_visible(timeout=800):
            await _await_or_stop(btn.click(timeout=1500), 3.0, stop_event)
    except Exception:
        pass


async def _reset_provider_in_place(page, provider_key: str, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    if provider_key == "gemini":
        ok = await _ensure_gemini_new_chat(page, stop_event)
        if not ok:
            raise RuntimeError("Gemini 新对话失败")
        return
    if provider_key == "doubao":
        await _ensure_doubao_chat(page, stop_event)
        return


async def _ensure_gemini_new_chat(page, stop_event: threading.Event) -> bool:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    ok_clicked = False
    for _ in range(3):
        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            cur = (page.url or "").lower()
            if ("gemini.google.com" not in cur) or ("/app" not in cur):
                await _await_or_stop(
                    page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=25_000),
                    30,
                    stop_event,
                )
        except Exception:
            pass
        try:
            await _dismiss_common_dialogs(page)
        except Exception:
            pass
        try:
            menu_candidates = [
                "button[aria-label*='menu' i]",
                "button[aria-label*='菜单' i]",
                "button[aria-label*='main menu' i]",
                "button[aria-label*='navigation' i]",
            ]
            for sel in menu_candidates:
                try:
                    m = page.locator(sel).first
                    if await m.is_visible(timeout=600):
                        await _await_or_stop(m.click(timeout=1200), 2.5, stop_event)
                        break
                except Exception:
                    continue
        except Exception:
            pass

        before_url = (page.url or "").strip()
        before_key = ""
        try:
            m = re.search(r"/app/([^/?#]+)", before_url)
            before_key = m.group(1) if m else ""
        except Exception:
            before_key = ""
        candidates = [
            page.get_by_role("button", name=re.compile(r"^(发起新对话|新对话|新建聊天|新聊天|New chat|New conversation)$", re.I)),
            page.get_by_role("link", name=re.compile(r"^(发起新对话|新对话|新建聊天|新聊天|New chat|New conversation)$", re.I)),
            page.locator("button[aria-label*='new chat' i],button[aria-label*='new conversation' i],button[aria-label*='新对话' i],button[aria-label*='新聊天' i],a[aria-label*='new chat' i],a[aria-label*='新对话' i]"),
            page.get_by_role("button", name=re.compile(r"(发起新对话|新建聊天|新聊天|新对话|New chat|New conversation)", re.I)),
            page.get_by_role("link", name=re.compile(r"(发起新对话|新建聊天|新聊天|新对话|New chat|New conversation)", re.I)),
            page.locator("span:has-text('发起新对话')"),
            page.get_by_text("发起新对话").first,
            page.locator("a[href*='/app']").filter(has_text=re.compile(r"新聊天|新对话|New chat", re.I)),
        ]
        for c in candidates:
            try:
                if await c.count() > 0 and await c.first.is_visible(timeout=800):
                    await _await_or_stop(c.first.click(timeout=1500, force=True), 3.0, stop_event)
                    ok_clicked = True
                    break
            except Exception:
                continue
        try:
            await _await_or_stop(page.keyboard.press("Escape"), 1.5, stop_event)
        except Exception:
            pass
        if ok_clicked:
            changed = False
            for _ in range(20):
                if stop_event.is_set():
                    raise RuntimeError("用户手动停止")
                cur_url = (page.url or "").strip()
                try:
                    m2 = re.search(r"/app/([^/?#]+)", cur_url)
                    cur_key = m2.group(1) if m2 else ""
                    cur_lower = cur_url.lower()
                    if before_key:
                        if cur_key and cur_key != before_key:
                            changed = True
                            break
                        if (not cur_key) and ("/app" in cur_lower):
                            changed = True
                            break
                    else:
                        if "/app" in cur_lower:
                            changed = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.2)
            if changed:
                break
            ok_clicked = False
        await asyncio.sleep(0.6)

    if not ok_clicked:
        return False
    try:
        await _poll_visible(
            page,
            selectors=[
                "textarea",
                "div[contenteditable='true'][role='textbox']",
                "[contenteditable='true']",
            ],
            timeout_s=14.0,
            step_s=0.5,
            stop_event=stop_event,
        )
    except Exception:
        return False
    return True


async def _ensure_grok_ready(page, stop_event: threading.Event):
    for _ in range(2):
        if "subscribe" not in (page.url or "").lower():
            break
        try:
            await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
        except Exception:
            pass
        try:
            await _dismiss_common_dialogs(page)
        except Exception:
            pass
    try:
        await _dismiss_common_dialogs(page)
    except Exception:
        pass
    try:
        closeers = [
            "button[aria-label='Close']",
            "button[aria-label*='close' i]",
            "button:has-text('Maybe later')",
            "button:has-text('稍后')",
            "button:has-text('以后')",
            "button:has-text('关闭')",
            "button:has-text('知道了')",
            "button:has-text('Not now')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
        ]
        for sel in closeers:
            try:
                loc = page.locator(sel).first
                if await loc.is_visible(timeout=700):
                    await _await_or_stop(loc.click(timeout=1000), 2.0, stop_event)
            except Exception:
                continue
    except Exception:
        pass
    if "subscribe" in (page.url or "").lower():
        try:
            await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
        except Exception:
            pass
    if "subscribe" in (page.url or "").lower():
        raise RuntimeError("Grok 被跳转到 subscribe/paywall")
    await _poll_visible(
        page,
        selectors=[
            "textarea",
            "div[contenteditable='true'][role='textbox']",
            "[contenteditable='true']",
            "input[type='text']",
        ],
        timeout_s=25.0,
        step_s=0.6,
        stop_event=stop_event,
    )


async def _ensure_grok_new_chat(page, stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    try:
        await _dismiss_common_dialogs(page)
    except Exception:
        pass
    for pat in [r"(新建聊天|新聊天)", r"(new chat|new conversation)"]:
        try:
            btn = page.get_by_role("button", name=re.compile(pat, re.I))
            if await btn.count() > 0 and await btn.first.is_visible(timeout=900):
                await _await_or_stop(btn.first.click(timeout=1200), 3.0, stop_event)
                return
        except Exception:
            pass
    try:
        loc = page.get_by_text(re.compile(r"(新建聊天|新聊天|new chat|new conversation)", re.I)).first
        if await loc.is_visible(timeout=900):
            await _await_or_stop(loc.click(timeout=1200), 3.0, stop_event)
    except Exception:
        pass


async def _dismiss_common_dialogs(page) -> None:
    labels = [
        r"(accept|agree|i agree|ok|got it|continue|allow)",
        r"(同意|接受|我同意|知道了|好的|继续|允许|确认)",
    ]
    for pat in labels:
        try:
            btn = page.get_by_role("button", name=re.compile(pat, re.I))
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=800)
        except Exception:
            pass
    try:
        close = page.get_by_role("button", name=re.compile(r"^(x|close|关闭)$", re.I))
        if await close.count() > 0 and await close.first.is_visible():
            await close.first.click(timeout=800)
    except Exception:
        pass


async def _find_input(page, provider_key: str):
    if provider_key == "grok":
        candidates = [
            page.locator("textarea"),
            page.locator("div[contenteditable='true'][role='textbox']"),
            page.locator("[contenteditable='true']"),
            page.get_by_role("textbox"),
            page.locator("div[role='textbox']"),
        ]
    elif provider_key == "gemini":
        candidates = [
            page.locator("textarea"),
            page.locator("div[contenteditable='true'][role='textbox']"),
            page.get_by_role("textbox"),
            page.locator("[contenteditable='true']"),
        ]
    elif provider_key == "doubao":
        candidates = [
            page.locator("textarea[placeholder*='发送' i]"),
            page.locator("textarea[placeholder*='消息' i]"),
            page.locator("textarea:visible"),
            page.locator("textarea"),
            page.locator("div[contenteditable='true'][role='textbox']"),
            page.locator("div[contenteditable='true']"),
            page.locator("[contenteditable='true']"),
            page.get_by_role("textbox"),
            page.locator("div[role='textbox']"),
        ]
    else:
        candidates = [
            page.get_by_placeholder(re.compile(r"(ask|message|输入|请输入|说点|提问|发送|对话|prompt)", re.I)),
            page.get_by_role("textbox"),
            page.locator("textarea"),
            page.locator("[contenteditable='true']"),
        ]
    for c in candidates:
        try:
            if await c.count() <= 0:
                continue
            cnt = await c.count()
            for idx in range(cnt - 1, -1, -1):
                el = c.nth(idx)
                try:
                    if await el.is_visible():
                        return el
                except Exception:
                    continue
        except Exception:
            continue
    return None


async def _wait_for_input(page, provider_key: str, timeout_s: int = 25, stop_event: threading.Event | None = None):
    start = time.time()
    last_err = ""
    while time.time() - start < timeout_s:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            await _dismiss_common_dialogs(page)
        except Exception:
            pass
        try:
            box = await _find_input(page, provider_key)
            if box is not None:
                try:
                    await box.scroll_into_view_if_needed(timeout=1200)
                except Exception:
                    pass
                return box
        except Exception as e:
            last_err = str(e)
        await asyncio.sleep(0.8)
    return None


async def _click_mode_if_present(page, provider_key: str):
    hints: list[str] = []
    if provider_key == "grok":
        hints = ["Chat", "聊天", "对话"]
    if provider_key == "doubao":
        hints = ["对话", "聊天", "Chat"]
    if provider_key == "gemini":
        hints = ["New chat", "新聊天", "Chat"]
    for h in hints:
        try:
            btn = page.get_by_role("button", name=re.compile(re.escape(h), re.I))
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=1500)
                return True
        except Exception:
            continue
    return False


async def _read_box_text(box) -> str:
    try:
        return (await box.input_value()) or ""
    except Exception:
        pass
    try:
        return (await box.inner_text()) or ""
    except Exception:
        pass
    try:
        v = await box.evaluate(
            """(el) => {
  try {
    if (el && typeof el.value === 'string') return el.value;
    if (el && el.getAttribute) {
      const ce = el.getAttribute('contenteditable');
      if (ce === 'true' || ce === '' || ce === 'plaintext-only') return (el.innerText || el.textContent || '');
    }
    return (el.innerText || el.textContent || '');
  } catch (e) { return ''; }
}"""
        )
        return v or ""
    except Exception:
        return ""


async def _ensure_box_text(box, text: str, timeout_s: int = 12, stop_event: threading.Event | None = None) -> bool:
    target = (text or "").strip()
    start = time.time()
    while time.time() - start < timeout_s:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        cur = (await _read_box_text(box)).strip()
        if cur and len(cur) >= int(len(target) * 0.9):
            return True
        await asyncio.sleep(0.25)
    return False


async def _is_textarea(box) -> bool:
    try:
        tag = await box.evaluate("el => (el && el.tagName) ? el.tagName.toLowerCase() : ''")
        return tag == "textarea"
    except Exception:
        return False


async def _insert_text_chunked(page, text: str, chunk_size: int, stop_event: threading.Event | None = None) -> None:
    t = text or ""
    for i in range(0, len(t), chunk_size):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        chunk = t[i : i + chunk_size]
        await _await_or_stop(page.keyboard.insert_text(chunk), 8.0, stop_event)
        await asyncio.sleep(0.02)


async def _click_send(page, provider_key: str) -> bool:
    patterns: list[re.Pattern[str]] = [
        re.compile(r"(发送|send|提交|generate|生成|run)", re.I),
    ]
    if provider_key == "gemini":
        patterns = [
            re.compile(r"(send|发送|提交)", re.I),
            re.compile(r"(submit|run)", re.I),
        ]
    if provider_key == "grok":
        patterns = [
            re.compile(r"(send|发送|提交|generate|生成)", re.I),
        ]
    if provider_key == "doubao":
        patterns = [
            re.compile(r"(发送|send|提交|生成|generate|run)", re.I),
            re.compile(r"(发送消息|发送内容|发送对话)", re.I),
        ]
    for pat in patterns:
        try:
            btn = page.get_by_role("button", name=pat)
            if await btn.count() > 0 and await btn.first.is_visible() and await btn.first.is_enabled():
                await btn.first.click(timeout=1500)
                return True
        except Exception:
            continue
    if provider_key == "gemini":
        css_candidates = [
            "[aria-label*='send' i]:not([aria-label*='stop' i])",
            "[aria-label*='发送' i]:not([aria-label*='停止' i])",
            "[aria-label*='Submit' i]",
            "[aria-label*='提交' i]",
            "[title*='send' i]:not([title*='stop' i])",
            "[title*='发送' i]:not([title*='停止' i])",
            "[data-testid='send-button']",
            ".send-button",
            "button:has-text(\"发送\")",
            "button:has-text(\"Send\")",
        ]
        for sel in css_candidates:
            try:
                loc = page.locator(sel)
                if await loc.count() > 0 and await loc.first.is_visible() and await loc.first.is_enabled():
                    await loc.first.click(timeout=1500)
                    return True
            except Exception:
                continue
    if provider_key == "doubao":
        css_candidates = [
            "button:has-text(\"发送\")",
            "button:has-text(\"Send\")",
            "[role='button']:has-text(\"发送\")",
            "button[aria-label*='发送' i]",
            "button[title*='发送' i]",
            "button[type='submit']",
        ]
        for sel in css_candidates:
            try:
                loc = page.locator(sel)
                if await loc.count() > 0 and await loc.first.is_visible() and await loc.first.is_enabled():
                    await loc.first.click(timeout=1500)
                    return True
            except Exception:
                continue
    try:
        btn2 = page.locator("button[type='submit']")
        if await btn2.count() > 0 and await btn2.first.is_visible() and await btn2.first.is_enabled():
            await btn2.first.click(timeout=1500)
            return True
    except Exception:
        pass
    return False


async def _click_send_doubao(page, box, stop_event: threading.Event | None = None) -> bool:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("用户手动停止")
    try:
        container = box.locator("xpath=ancestor::form[1]")
        if await container.count() <= 0:
            container = box.locator("xpath=ancestor::div[1]")
    except Exception:
        container = page.locator("body")

    selectors = [
        "button[type='submit']",
        "button[aria-label*='发送' i]",
        "button[aria-label*='send' i]",
        "button[title*='发送' i]",
        "button:has-text(\"发送\")",
        "[role='button']:has-text(\"发送\")",
    ]
    for sel in selectors:
        try:
            loc = container.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible(timeout=600) and await loc.is_enabled():
                await _await_or_stop(loc.click(timeout=1200), 2.5, stop_event)
                return True
        except Exception:
            continue

    try:
        loc2 = page.locator("button[type='submit']").last
        if await loc2.count() > 0 and await loc2.is_visible(timeout=600) and await loc2.is_enabled():
            await _await_or_stop(loc2.click(timeout=1200), 2.5, stop_event)
            return True
    except Exception:
        pass

    try:
        loc3 = page.locator("button").filter(has=page.locator("svg")).last
        if await loc3.count() > 0 and await loc3.is_visible(timeout=600) and await loc3.is_enabled():
            await _await_or_stop(loc3.click(timeout=1200), 2.5, stop_event)
            return True
    except Exception:
        pass
    return False


async def _verify_sent(page, box, text: str, before_body: str, timeout_s: int = 8, stop_event: threading.Event | None = None, provider_key: str = "") -> bool:
    target = (text or "").strip()
    start = time.time()
    while time.time() - start < timeout_s:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        try:
            cur = (await _read_box_text(box)).strip()
            if cur == "" or len(cur) <= max(5, int(len(target) * 0.1)):
                return True
        except Exception:
            pass
        try:
            stop_btn = page.get_by_role("button", name=re.compile(r"(停止生成|stop generating|stop|停止)", re.I))
            if await stop_btn.count() > 0 and await stop_btn.first.is_visible():
                return True
            btn = page.locator("button[type='submit']")
            if await btn.count() > 0 and not await btn.first.is_enabled():
                return True
        except Exception:
            pass
        try:
            after = await _body_text(page, provider_key, timeout_ms=2500, stop_event=stop_event)
            if after and after != before_body:
                if (target[:20] and target[:20] in after) or (len(after) > len(before_body) + 20):
                    return True
        except Exception:
            pass
        if provider_key == "doubao":
            try:
                after = await _body_text(page, provider_key, timeout_ms=2500, stop_event=stop_event)
                if after and after != before_body and target[:20] and target[:20] in after:
                    return True
            except Exception:
                pass
        if provider_key == "grok":
            try:
                after = await _body_text(page, provider_key, timeout_ms=2500, stop_event=stop_event)
                if after and after != before_body and target[:20] and target[:20] in after:
                    return True
            except Exception:
                pass
        await asyncio.sleep(0.5)
    return False


async def _try_send(page, box, text: str, provider_key: str, stop_event: threading.Event | None = None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("用户手动停止")
    try:
        await _await_or_stop(box.click(timeout=1500), 2.5, stop_event)
    except Exception:
        pass
    before_body = _tail(await _body_text(page, provider_key, timeout_ms=2500, stop_event=stop_event), 6000)
    filled = False
    if provider_key in ("doubao", "gemini", "grok"):
        try:
            await _await_or_stop(page.keyboard.press("Control+A"), 2.0, stop_event)
        except Exception:
            pass
        try:
            await _await_or_stop(page.keyboard.press("Backspace"), 2.0, stop_event)
        except Exception:
            pass
        try:
            if provider_key == "doubao" and await _is_textarea(box):
                await _await_or_stop(box.fill(text, timeout=8000), 10.0, stop_event)
                filled = True
            else:
                if provider_key == "doubao":
                    try:
                        await _await_or_stop(
                            box.evaluate(
                                """(el) => {
  try {
    if (!el) return;
    if (typeof el.value === 'string') {
      el.value = '';
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return;
    }
    el.innerText = '';
    el.dispatchEvent(new InputEvent('input', { bubbles: true, data: '' }));
  } catch (e) {}
}"""
                            ),
                            2.0,
                            stop_event,
                        )
                    except Exception:
                        pass
                await _insert_text_chunked(page, text, chunk_size=380, stop_event=stop_event)
                filled = True
        except Exception:
            filled = False
    else:
        try:
            ok_set = await _await_or_stop(
                box.evaluate(
                    """(el, v) => {
  try {
    if (!el) return false;
    if (typeof el.value === 'string') {
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    }
    const ce = el.getAttribute && el.getAttribute('contenteditable');
    if (ce === 'true' || ce === '' || ce === 'plaintext-only') {
      el.innerText = v;
      el.dispatchEvent(new InputEvent('input', { bubbles: true, data: v }));
      return true;
    }
    el.innerText = v;
    el.dispatchEvent(new InputEvent('input', { bubbles: true, data: v }));
    return true;
  } catch (e) { return false; }
}""",
                    text,
                ),
                4.0,
                stop_event,
            )
            filled = bool(ok_set)
        except Exception:
            filled = False

    if not filled and provider_key != "doubao":
        try:
            await _await_or_stop(box.fill(text, timeout=5000), 6.0, stop_event)
            filled = True
        except Exception:
            filled = False

    if not filled:
        try:
            await _await_or_stop(box.click(timeout=1500), 2.5, stop_event)
        except Exception:
            pass
        try:
            await _await_or_stop(page.keyboard.press("Control+A"), 2.0, stop_event)
        except Exception:
            pass
        if provider_key != "doubao":
            try:
                await _await_or_stop(page.keyboard.type(text, delay=2), 20.0, stop_event)
            except Exception:
                pass
    ok_text = await _ensure_box_text(box, text, timeout_s=12, stop_event=stop_event)
    if not ok_text:
        try:
            cur = await _await_or_stop(
                box.evaluate(
                    """(el) => {
  try {
    if (!el) return '';
    if (typeof el.value === 'string') return (el.value || '').trim();
    const ce = el.getAttribute && el.getAttribute('contenteditable');
    if (ce === 'true' || ce === '' || ce === 'plaintext-only') return (el.innerText || '').trim();
    return (el.innerText || '').trim();
  } catch (e) { return ''; }
}"""
                ),
                3.0,
                stop_event,
            )
        except Exception:
            cur = ""
        if not str(cur or "").strip():
            raise RuntimeError("输入未完成")
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("用户手动停止")
    await asyncio.sleep(1.0)
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("用户手动停止")
    sent = False
    if provider_key == "doubao":
        sent = False
    else:
        try:
            clicked = await _click_send(page, provider_key)
            if clicked:
                sent = await _verify_sent(page, box, text, before_body, timeout_s=8, stop_event=stop_event, provider_key=provider_key)
        except Exception:
            sent = False
    if sent:
        return
    if provider_key == "gemini":
        try:
            await _await_or_stop(box.click(timeout=1500), 2.0, stop_event)
        except Exception:
            pass
        try:
            await _await_or_stop(page.keyboard.press("Enter"), 2.0, stop_event)
            if await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="gemini"):
                return
        except Exception:
            pass
        try:
            clicked = await _click_send(page, "gemini")
            if clicked and await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="gemini"):
                return
        except Exception:
            pass
        try:
            await _await_or_stop(box.click(timeout=1500), 2.0, stop_event)
            await _await_or_stop(page.keyboard.press("Control+Enter"), 2.0, stop_event)
            if await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="gemini"):
                return
        except Exception:
            pass
        raise RuntimeError("Gemini 发送失败")
    if provider_key == "doubao":
        try:
            await _await_or_stop(page.keyboard.press("Enter"), 2.0, stop_event)
            if await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="doubao"):
                return
        except Exception:
            pass
        try:
            clicked = await _click_send_doubao(page, box, stop_event=stop_event)
            if clicked and await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="doubao"):
                return
        except Exception:
            pass
        try:
            await _await_or_stop(page.keyboard.press("Control+Enter"), 2.0, stop_event)
            if await _verify_sent(page, box, text, before_body, timeout_s=10, stop_event=stop_event, provider_key="doubao"):
                return
        except Exception:
            pass
        raise RuntimeError("豆包发送失败")
    if provider_key == "grok":
        try:
            await _await_or_stop(page.keyboard.press("Control+Enter"), 2.0, stop_event)
            if await _verify_sent(page, box, text, before_body, timeout_s=8, stop_event=stop_event, provider_key="grok"):
                return
        except Exception:
            pass
    if provider_key != "gemini":
        try:
            await _await_or_stop(page.keyboard.press("Enter"), 2.0, stop_event)
            await _verify_sent(page, box, text, before_body, timeout_s=6, stop_event=stop_event, provider_key=provider_key)
        except Exception:
            pass


def _tail(s: str, n: int = 8000) -> str:
    t = (s or "").strip()
    if len(t) <= n:
        return t
    return t[-n:]


async def _body_text(page, provider_key: str = "", timeout_ms: int = 7000, stop_event: threading.Event | None = None) -> str:
    if provider_key == "doubao":
        try:
            return await _await_or_stop(page.locator("div[class*='chat-list']").first.inner_text(timeout=timeout_ms), max(2.0, timeout_ms / 1000 + 1), stop_event)
        except Exception:
            pass
        try:
            return await _await_or_stop(page.locator("div.chat-scroll-container").first.inner_text(timeout=timeout_ms), max(2.0, timeout_ms / 1000 + 1), stop_event)
        except Exception:
            pass
    try:
        return await _await_or_stop(page.locator("body").inner_text(timeout=timeout_ms), max(2.0, timeout_ms / 1000 + 1), stop_event)
    except Exception:
        try:
            return await _await_or_stop(page.locator("html").inner_text(timeout=timeout_ms), max(2.0, timeout_ms / 1000 + 1), stop_event)
        except Exception:
            return ""


async def _wait_reply_delta(page, provider_key: str, before: str, timeout_s: int = 180, stop_event: threading.Event | None = None) -> str:
    start = time.time()
    last_text = before
    last_change = time.time()
    saw_growth = False

    while time.time() - start < timeout_s:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("用户手动停止")
        await asyncio.sleep(1.0)
        after = await _body_text(page, provider_key, timeout_ms=7000, stop_event=stop_event)
        if after and after != last_text:
            if len(after) > len(last_text) + 30:
                saw_growth = True
            last_text = after
            last_change = time.time()
        if saw_growth and (time.time() - last_change) >= 3.0:
            break

    after = last_text or (await _body_text(page, provider_key, timeout_ms=7000, stop_event=stop_event))
    if not after:
        return ""
    if before and after.startswith(before):
        return _tail(after[len(before) :], 6000)
    return _tail(after, 6000)


def _job_next_step(job: dict[str, Any], lock: threading.Lock, provider_key: str) -> int:
    with lock:
        m = job.get("step_idx")
        if not isinstance(m, dict):
            m = {}
            job["step_idx"] = m
        cur = int(m.get(provider_key, 0)) + 1
        m[provider_key] = cur
        return cur


async def _shot_step(page, provider_key: str, run_date: str, label: str, job: dict[str, Any], lock: threading.Lock, stop_event: threading.Event):
    try:
        step = _job_next_step(job, lock, provider_key)
        p = _trace_dir(run_date, provider_key) / f"{datetime.now().strftime('%H%M%S')}_{step:03d}_{label}.png"
        await _await_or_stop(page.screenshot(path=str(p), full_page=True), 8.0, stop_event)
        _job_append_shot(job, lock, str(p))
    except Exception:
        pass


async def _open_provider_tab(ctx, provider: WebProvider, run_date: str, stop_event: threading.Event, job: dict[str, Any], lock: threading.Lock):
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    _job_append_log(job, lock, provider.title, "打开新窗口", "开始")
    page = await _goto_new_tab(ctx, provider.url, stop_event=stop_event)
    await _shot_step(page, provider.key, run_date, "open", job, lock, stop_event)
    _job_append_log(job, lock, provider.title, "打开新窗口", "成功")

    if stop_event.is_set():
        raise RuntimeError("用户手动停止")
    try:
        await _dismiss_common_dialogs(page)
    except Exception:
        pass
    await _shot_step(page, provider.key, run_date, "dialogs", job, lock, stop_event)

    if provider.key == "doubao":
        try:
            await _ensure_doubao_chat(page, stop_event)
        except Exception:
            pass
        await _shot_step(page, provider.key, run_date, "reset", job, lock, stop_event)

    if provider.key == "gemini":
        ok_reset = False
        try:
            ok_reset = await _ensure_gemini_new_chat(page, stop_event)
        except Exception as e:
            _job_append_log(job, lock, provider.title, "新对话失败", str(e)[:160])
            ok_reset = False
        if ok_reset:
            try:
                await _await_or_stop(page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            try:
                cur = (page.url or "").lower()
                if re.search(r"/app/[^/?#]+", cur):
                    ok2 = await _ensure_gemini_new_chat(page, stop_event)
                    ok_reset = bool(ok2)
            except Exception:
                pass
        if not ok_reset:
            try:
                await _await_or_stop(page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            try:
                ok_reset = await _ensure_gemini_new_chat(page, stop_event)
            except Exception as e:
                _job_append_log(job, lock, provider.title, "新对话二次尝试失败", str(e)[:160])
                ok_reset = False
        await _shot_step(page, provider.key, run_date, "reset_ok" if ok_reset else "reset_failed", job, lock, stop_event)
        if not ok_reset:
            _job_append_log(job, lock, provider.title, "新对话失败", "继续执行（可能存在上下文污染）")

    try:
        await _click_mode_if_present(page, provider.key)
    except Exception:
        pass
    await _shot_step(page, provider.key, run_date, "mode", job, lock, stop_event)

    if provider.key == "grok":
        await _ensure_grok_ready(page, stop_event)
        try:
            await _ensure_grok_new_chat(page, stop_event)
        except Exception:
            pass
        await _shot_step(page, provider.key, run_date, "ready", job, lock, stop_event)

    return page


async def _ask_web_ai_in_tab(
    page,
    provider: WebProvider,
    anchor: str,
    prompt_text: str,
    round_name: str,
    run_date: str,
    stop_event: threading.Event,
    job: dict[str, Any],
    lock: threading.Lock,
) -> str:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")

    if provider.key == "grok":
        await _ensure_grok_ready(page, stop_event)
        if "subscribe" in (page.url or "").lower():
            try:
                await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            await _ensure_grok_ready(page, stop_event)

    box = await _wait_for_input(page, provider.key, timeout_s=35, stop_event=stop_event)
    if box is None:
        await _shot_step(page, provider.key, run_date, f"{round_name}_no_input", job, lock, stop_event)
        raise RuntimeError(f"{provider.title} 未定位到输入框")
    try:
        await _await_or_stop(box.click(timeout=1500), 2.5, stop_event)
    except Exception:
        pass
    await _shot_step(page, provider.key, run_date, f"{round_name}_focus", job, lock, stop_event)

    q = _sanitize_web_question(provider.key, anchor, prompt_text)
    before = _tail(await _body_text(page, provider.key, timeout_ms=7000, stop_event=stop_event), 12000)
    _job_append_log(job, lock, provider.title, f"{round_name} 输入内容", "开始")
    try:
        await _try_send(page, box, q, provider.key, stop_event=stop_event)
        await _shot_step(page, provider.key, run_date, f"{round_name}_sent", job, lock, stop_event)
        _job_append_log(job, lock, provider.title, f"{round_name} 点击发送", "完成")
    except Exception as e:
        if provider.key == "grok" and "subscribe" in (page.url or "").lower():
            try:
                await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            await _ensure_grok_ready(page, stop_event)
            box2 = await _wait_for_input(page, provider.key, timeout_s=35, stop_event=stop_event)
            if box2 is not None:
                try:
                    await _await_or_stop(box2.click(timeout=1500), 2.5, stop_event)
                except Exception:
                    pass
                try:
                    await _try_send(page, box2, q, provider.key, stop_event=stop_event)
                    await _shot_step(page, provider.key, run_date, f"{round_name}_sent_retry", job, lock, stop_event)
                    _job_append_log(job, lock, provider.title, f"{round_name} 点击发送", "重试成功")
                    e = None
                except Exception as e2:
                    e = e2
        if e is not None:
            _job_append_log(job, lock, provider.title, f"{round_name} 发送失败", str(e)[:160])
            await _shot_step(page, provider.key, run_date, f"{round_name}_send_error", job, lock, stop_event)
            raise

    _job_append_log(job, lock, provider.title, f"{round_name} 等待回复", "开始")
    delta = await _wait_reply_delta(page, provider.key, before, timeout_s=180 if provider.key == "gemini" else 140, stop_event=stop_event)
    await _shot_step(page, provider.key, run_date, f"{round_name}_done", job, lock, stop_event)
    _job_append_log(job, lock, provider.title, f"{round_name} 等待回复", "完成")
    return delta.strip() or "(未提取到回复，请在页面确认)"


async def _ask_web_ai_raw_in_tab(
    page,
    provider: WebProvider,
    prompt_text: str,
    round_name: str,
    run_date: str,
    stop_event: threading.Event,
    job: dict[str, Any],
    lock: threading.Lock,
) -> str:
    if stop_event.is_set():
        raise RuntimeError("用户手动停止")

    if provider.key == "grok":
        await _ensure_grok_ready(page, stop_event)
        if "subscribe" in (page.url or "").lower():
            try:
                await _await_or_stop(page.goto("https://grok.com/", wait_until="domcontentloaded", timeout=25_000), 30, stop_event)
            except Exception:
                pass
            await _ensure_grok_ready(page, stop_event)

    box = await _wait_for_input(page, provider.key, timeout_s=35, stop_event=stop_event)
    if box is None:
        await _shot_step(page, provider.key, run_date, f"{round_name}_no_input", job, lock, stop_event)
        raise RuntimeError(f"{provider.title} 未定位到输入框")
    try:
        await _await_or_stop(box.click(timeout=1500), 2.5, stop_event)
    except Exception:
        pass
    await _shot_step(page, provider.key, run_date, f"{round_name}_focus", job, lock, stop_event)

    q = (prompt_text or "").strip()
    before = _tail(await _body_text(page, provider.key, timeout_ms=7000, stop_event=stop_event), 12000)
    _job_append_log(job, lock, provider.title, f"{round_name} 输入内容", "开始")
    try:
        await _try_send(page, box, q, provider.key, stop_event=stop_event)
        await _shot_step(page, provider.key, run_date, f"{round_name}_sent", job, lock, stop_event)
        _job_append_log(job, lock, provider.title, f"{round_name} 点击发送", "完成")
    except Exception as e:
        _job_append_log(job, lock, provider.title, f"{round_name} 发送失败", str(e)[:160])
        await _shot_step(page, provider.key, run_date, f"{round_name}_send_error", job, lock, stop_event)
        raise

    _job_append_log(job, lock, provider.title, f"{round_name} 等待回复", "开始")
    delta = await _wait_reply_delta(page, provider.key, before, timeout_s=180 if provider.key == "gemini" else 140, stop_event=stop_event)
    await _shot_step(page, provider.key, run_date, f"{round_name}_done", job, lock, stop_event)
    _job_append_log(job, lock, provider.title, f"{round_name} 等待回复", "完成")
    return delta.strip() or "(未提取到回复，请在页面确认)"


def _qwen_make_research_prompt(client: LocalLMDeploy, platform: str, anchor: str, extra: str) -> str:
    sys = "你是冷静的事实核验编辑。输出给网页端AI的提问指令，必须可直接复制粘贴使用。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            f"用户补充：{extra.strip() or '无'}",
            "",
            "生成一份详尽提示词，用于询问三方网页AI（Gemini/Grok/豆包）。必须包含并强制对方按项回答：",
            "1) 历史经典梗、粉丝昵称、黑粉绰号（逐条列出）。",
            "2) 粉丝/黑粉喜欢/讨厌的具体原因（可验证表述）。",
            "3) 对立主播名单、亲近联盟主播名单（分组给出）。",
            "4) 近半个月该平台热点节奏/可蹭流量话题（按日期或事件列出）。",
            "5) 硬性要求：杜绝幻觉；每一条观点都必须给出依据/出处（链接、截图出处描述、关键词可检索来源、时间点、直播间/视频标题等）。",
            "格式要求：用 Markdown 分段，包含“回答结构模板”，并要求对方在每条后面写【依据】。",
        ]
    )
    return client.generate(prompt=p, system=sys, options={"temperature": 0.2, "num_predict": 800}, stop_event=threading.Event())


def _qwen_make_research_prompt_stream(client: LocalLMDeploy, platform: str, anchor: str, extra: str, stop_event: threading.Event) -> str:
    sys = f"你是冷静、犀利、嘴毒但不辱骂的事实核验编辑。必须全程中文输出，禁止造谣与杜撰。你要先把隐喻/反话/阴阳怪气/影射翻译成“可核验的主张”，再把每条主张写成可追证据链的问题。专有名词必须原样保留，主播名必须写成：{anchor}（禁止同音字/错别字）。输出将直接发给网页端AI作为问答指令，必须写成让对方直接回答的问题清单。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            f"用户补充：{extra.strip() or '无'}",
            "",
            f"硬性要求：全文中文；主播名必须原样：{anchor}（禁止改写）；绝对不要出现任何英文（如Prompt、Analyze、Platform等）；不要代码块。",
            "请直接回答以下问题，并为每一条给【依据】（链接/标题/时间点/关键词/可检索描述）。不要生成提示词/提问模板/教学。",
            "1) 该主播的历史经典梗、粉丝昵称、黑粉绰号（逐条列出）。",
            "2) 粉丝/黑粉喜欢/讨厌该主播的具体原因（逐条列出，避免泛泛而谈）。",
            "3) 该主播的对立主播名单、亲近联盟主播名单（分组列出）。",
            "4) 近半个月该平台的热点节奏/可蹭流量话题（按日期或事件列出）。",
            "5) 若依据不足，必须标注【待核实】并说明建议的核验路径（搜什么关键词/看哪个时间点）。",
            "格式要求：用纯中文 Markdown 分段；每条后面写【依据】。",
        ]
    )
    return _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.2, "num_predict": 800}, stop_event)


def _llm_intent_profile_stream(client: LocalLMDeploy, platform: str, anchor: str, extra: str, stop_event: threading.Event) -> str:
    sys = f"你是深度意图识别分析器。必须全程中文输出。语言风格犀利但不得辱骂与造谣。必须识别隐喻、反话、阴阳怪气、潜台词，并还原成可执行的创作目标与核验路径。主播名必须原样：{anchor}。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            "用户输入（创作意图）：",
            extra.strip() or "(空)",
            "",
            "任务：把用户输入解构成可执行的创作任务画像，用于后续生成提问与核验。",
            "输出必须是严格 JSON，且只输出 JSON（不要任何前后缀、不要 Markdown、不要代码块）。",
            "JSON Schema：",
            '{'
            '"core_goal": "一句话核心目标",'
            '"audience": "受众画像",'
            '"content_form": "内容形态",'
            '"persona": "语气人设（默认毒舌但不辱骂）",'
            '"must_include": ["要点1","要点2"],'
            '"must_avoid": ["禁忌1","禁忌2"],'
            '"verifiable_clues": ["关键词/时间点/平台入口"],'
            '"risk_warnings": ["隐私/引战/侵权/误伤"]'
            '}',
        ]
    )
    return _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.1, "num_predict": 500}, stop_event)


def _repair_intent_profile_json_stream(client: LocalLMDeploy, platform: str, anchor: str, raw: str, stop_event: threading.Event) -> str:
    sys = f"你是严格 JSON 修复器。必须全程中文输出。你只允许输出严格 JSON，不要任何前后缀、不要 Markdown、不要代码块。主播名必须原样：{anchor}。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            "你收到一段“意图画像”输出，但它可能包含多余文字/格式错误/半截 JSON。",
            "任务：把它修复为严格 JSON，且必须符合这个 Schema，字段必须齐全：",
            '{'
            '"core_goal": "一句话核心目标",'
            '"audience": "受众画像",'
            '"content_form": "内容形态",'
            '"persona": "语气人设（默认毒舌但不辱骂）",'
            '"must_include": ["要点1","要点2"],'
            '"must_avoid": ["禁忌1","禁忌2"],'
            '"verifiable_clues": ["关键词/时间点/平台入口"],'
            '"risk_warnings": ["隐私/引战/侵权/误伤"]'
            '}',
            "",
            "原始输出：",
            raw.strip() or "(空)",
        ]
    )
    return _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.0, "num_predict": 450}, stop_event)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.I)
    if m:
        t = (m.group(1) or "").strip()
    def _try_partial(s: str) -> dict[str, Any] | None:
        s2 = (s or "").strip()
        if not s2:
            return None
        try:
            import partial_json_parser
            obj = partial_json_parser.loads(s2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    def _try(s: str) -> dict[str, Any] | None:
        s2 = (s or "").strip()
        if not s2:
            return None
        try:
            obj = json.loads(s2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            s2 = re.sub(r",\s*([}\]])", r"\1", s2)
            try:
                obj = json.loads(s2)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    def _extract_balanced_object(s: str) -> str | None:
        start = (s or "").find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        quote = ""
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == quote:
                    in_str = False
                    quote = ""
                continue
            if ch in ("\"", "'"):
                in_str = True
                quote = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None
    direct = _try(t)
    if direct is not None:
        return direct
    direct2 = _try_partial(t)
    if direct2 is not None:
        return direct2
    block = _extract_balanced_object(t)
    if block is not None:
        obj = _try(block)
        if obj is not None:
            return obj
        obj2 = _try_partial(block)
        if obj2 is not None:
            return obj2
    return None


def _qwen_find_conflicts(client: LocalLMDeploy, anchor: str, a: dict[str, str]) -> str:
    sys = "你是事实核验官。只输出可直接发给网页AI的二次追问指令。"
    p = "\n".join(
        [
            f"主播：{anchor}",
            "你收到三方回答如下（可能有矛盾/重复/幻觉）。",
            "",
            "【Gemini】",
            a.get("gemini", ""),
            "",
            "【Grok】",
            a.get("grok", ""),
            "",
            "【豆包】",
            a.get("doubao", ""),
            "",
            "任务：",
            "1) 找出三方之间的冲突点/说法不一致点/来源不清点（逐条列出）。",
            "2) 对每个冲突点生成一个追问问题，要求给出可验证依据（链接/标题/时间点/关键词）。",
            "3) 输出一份二次追问提示词：以“逐条追问清单”的形式，方便直接粘贴到 Gemini/Grok/豆包里。",
            "输出必须是中文 Markdown。",
        ]
    )
    return client.generate(prompt=p, system=sys, options={"temperature": 0.2, "num_predict": 900}, stop_event=threading.Event())


def _qwen_find_conflicts_stream(client: LocalLMDeploy, anchor: str, base_questions: str, a: dict[str, str], stop_event: threading.Event) -> str:
    sys = f"你是嘴毒但不骂人的事实核验官。必须全程中文输出。你要先把隐喻、反话、影射、含沙射影翻译成可核验主张，再抓住三方差异点做精准追问。主播名必须原样：{anchor}（禁止同音字/错别字）。严禁英文与代码块。"
    p = "\n".join(
        [
            f"主播：{anchor}",
            "本次必须围绕以下四个问题（逐条核验、逐条追问，不能跳题）：",
            (base_questions or "").strip(),
            "",
            "你收到三方回答如下（可能有矛盾/重复/幻觉）。",
            f"重要：本次任务只讨论主播“{anchor}”。如果三方回答里出现其它主播名/其它平台不同人（例如上一条任务的主播），一律视为上下文污染，不得拿去做差异对比；必须把它作为第一优先级纠错点，追问对方‘请清空错误上下文，仅针对 {anchor} 重答并给依据’。",
            "",
            "【Gemini】",
            a.get("gemini", ""),
            "",
            "【Grok】",
            a.get("grok", ""),
            "",
            "【豆包】",
            a.get("doubao", ""),
            "",
            "任务：",
            "根据三方的回答差异，为你自己生成发给这三家AI的【直接追问问题】。",
            "1. 绝不要输出“差异点总结”或“对比分析”。",
            "1.1 严禁把“四个问题原文”再次原样复述当作追问；追问必须是针对性补问（例如：要求补证据/澄清口径/纠正串主播/补全遗漏）。",
            "2. 只允许输出以下三个标题及对应发给该AI的提问：",
            "### 发给 Gemini 的二次追问",
            "### 发给 Grok 的二次追问",
            "### 发给 豆包 的二次追问",
            "3. 针对每个AI的提问，必须是直接向它提问的口吻：",
            "- 第一优先：如果它串了别的主播/旧窗口内容，要求它清空错误上下文，仅针对本次四问重答并给依据。",
            "- 第二优先：指出它与另外两家不一致/缺失的点，要求它补证据或承认不确定。",
            "- 必须要求它按四个问题逐条作答（第1问/第2问/第3问/第4问），且每条给【依据】（链接/标题/时间点/关键词/可检索描述）。",
            "4. 硬性要求：全文纯中文；不要出现英文；不要代码块；不要写“根据对比发现...”。",
        ]
    )
    return _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.2, "num_predict": 900}, stop_event)


def _qwen_final_report(client: LocalLMDeploy, platform: str, anchor: str, round1: dict[str, str], round2: dict[str, str]) -> str:
    sys = f"你是事实核验总编。必须全程中文输出。主播名必须原样：{anchor}（禁止改写/同音字）。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            "",
            "你有两轮三方网页AI回复。请输出【创作素材可用性核验报告】。",
            "要求：",
            "1) 只输出两段：【可直接用于创作素材】与【需要人工确认】。",
            "2) 判定规则：只有当“至少两家给出一致结论”且“至少一条可检索依据”时，才能进入【可直接用于创作素材】；否则进入【需要人工确认】。",
            "3) 每条要点必须带【依据】与【风险提示】。",
            "4) 禁止复述提示词/硬性要求/教学内容；只提炼信息点与证据。",
            "",
            "【第一轮-三方汇总】",
            "Gemini:\n" + (round1.get("gemini", "") or ""),
            "\nGrok:\n" + (round1.get("grok", "") or ""),
            "\n豆包:\n" + (round1.get("doubao", "") or ""),
            "",
            "【第二轮-三方汇总】",
            "Gemini:\n" + (round2.get("gemini", "") or ""),
            "\nGrok:\n" + (round2.get("grok", "") or ""),
            "\n豆包:\n" + (round2.get("doubao", "") or ""),
        ]
    )
    return client.generate(prompt=p, system=sys, options={"temperature": 0.15, "num_predict": 1400}, stop_event=threading.Event())


def _qwen_final_report_stream(client: LocalLMDeploy, platform: str, anchor: str, round1: dict[str, str], round2: dict[str, str], stop_event: threading.Event) -> str:
    sys = f"你是嘴毒但不辱骂的事实核验总编。必须全程中文输出。你要先把隐喻、反话、影射等表达翻译成“事实候选项”，再按证据强度与一致性审判：哪些能用、哪些必须人工复核。专有名词必须原样保留，主播名必须写成：{anchor}（禁止同音字/错别字）。严禁输出<think>或</think>标签，严禁输出思考过程。"
    p = "\n".join(
        [
            f"平台：{platform}",
            f"主播：{anchor}",
            "",
            f"硬性要求：全文中文；主播名必须原样：{anchor}（禁止改写）。",
            "你有两轮三方网页AI回复。请输出【创作素材可用性核验报告】。",
            "要求：",
            "1) 只允许输出两个主标题（各出现一次，标题不能改字）：",
            "【可直接用于创作素材】",
            "【需要人工确认】",
            "2) 判定规则（必须严格执行）：",
            "- 只有当“至少两家给出一致结论”且“至少一条可检索依据”时，才能进入【可直接用于创作素材】。",
            "- 只要出现“单家独有/互相矛盾/没有证据/证据太泛/可能涉及隐私或引战”，一律进入【需要人工确认】。",
            "3) 每条要点必须采用固定字段：",
            "- 【要点】一句话结论",
            "- 【依据】来自哪家AI + 可检索线索（链接/标题/时间点/关键词）",
            "- 【风险提示】一句话风险",
            "- 【建议动作】（仅在需要人工确认时填写：给出搜什么关键词/看什么时间点）",
            "4) 禁止复述提示词、硬性要求、教学、过程解释；只输出要点清单。",
            "",
            "【第一轮-三方汇总】",
            "Gemini:\n" + (round1.get("gemini", "") or ""),
            "\nGrok:\n" + (round1.get("grok", "") or ""),
            "\n豆包:\n" + (round1.get("doubao", "") or ""),
            "",
            "【第二轮-三方汇总】",
            "Gemini:\n" + (round2.get("gemini", "") or ""),
            "\nGrok:\n" + (round2.get("grok", "") or ""),
            "\n豆包:\n" + (round2.get("doubao", "") or ""),
        ]
    )
    raw = _qwen_generate_stream_with_stop(client, p, sys, {"temperature": 0.15, "num_predict": 1400}, stop_event)
    out = _strip_think(raw)
    if out:
        return out
    sys2 = sys + " 你刚才输出为空或只包含思考标签。请直接输出两段要点清单，不要输出任何解释。"
    raw2 = _qwen_generate_stream_with_stop(client, p, sys2, {"temperature": 0.15, "num_predict": 1600}, stop_event)
    out2 = _strip_think(raw2)
    return out2 or (raw2 or "").strip() or (raw or "").strip()


def _sanitize_ollama_base_url(ollama_url: str, cdp_url: str) -> tuple[str, str]:
    o = (ollama_url or "").strip()
    c = (cdp_url or "").strip()
    if not o:
        return "", "Ollama 地址为空"
    if o == c:
        return "", "Ollama 地址与 CDP 地址相同，请分别填写 11434 与 9222"
    if ":9222" in o or o.rstrip("/").endswith("9222"):
        return "", "Ollama 地址疑似填了 CDP 端口（9222），请改为 http://localhost:11434"
    return o, ""


def _strip_think(text: str) -> str:
    import re
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<\|[^>]{1,80}\|>", "", t)
    return t.strip()


def _clean_web_ai_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = _strip_think(t)
    t = re.sub(r"^```[\s\S]*?```$", "", t, flags=re.M).strip()
    drop_pats = [
        r"^\s*这是问答任务[:：].*$",
        r"^\s*硬性要求[:：].*$",
        r"^\s*主播专名必须原样.*$",
        r"^\s*每一条观点必须给【依据】.*$",
        r"^\s*不要生成提示词.*$",
        r"^\s*不要英文.*$",
        r"^\s*不要代码块.*$",
    ]
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            lines.append(line)
            continue
        if any(re.match(p, s) for p in drop_pats):
            continue
        lines.append(line)
    out = "\n".join(lines).strip()
    return out or t


def _obsidian_base_dir() -> Path:
    try:
        store = ObsidianStore()
        base = getattr(store, "base_dir", None)
        return Path(base) if base else (Path(__file__).resolve().parents[1] / "Obsidian 知识库")
    except Exception:
        return Path(__file__).resolve().parents[1] / "Obsidian 知识库"


def _anchor_variants(anchor: str) -> list[str]:
    a = (anchor or "").strip()
    if not a:
        return []
    short = re.split(r"[（(]", a, 1)[0].strip()
    out = []
    for x in [a, short]:
        if x and x not in out:
            out.append(x)
    return out


def _latest_files_in(dir_path: Path, anchor: str, limit: int) -> list[Path]:
    if not dir_path.exists():
        return []
    hits: list[Path] = []
    av = _anchor_variants(anchor)
    for p in dir_path.glob("*.md"):
        name = p.name
        if any(v in name for v in av):
            hits.append(p)
    hits.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0.0, reverse=True)
    return hits[: max(0, int(limit or 0))]


def _load_local_material(anchor: str) -> tuple[Path, str]:
    base = _obsidian_base_dir()
    fact_dir = base / "06_文案库" / "主播事实核验" / "四问事实核验"
    net_dir = base / "06_文案库" / "主播事实核验" / "粉丝团与关系网"
    fact_files = _latest_files_in(fact_dir, anchor, limit=2)
    net_files = _latest_files_in(net_dir, anchor, limit=2)
    net_named = net_dir / f"{anchor}.md"
    if net_named.exists() and net_named not in net_files:
        net_files = [net_named] + net_files

    cfg = DistillConfig(
        input_roots=_default_input_roots(),
        output_root=base / "_tmp",
        model="gemma",
        base_url="http://localhost:11434",
        max_comment_lines=1800,
        max_total_chars=42000,
        num_predict=0,
        track="",
        stream=False,
    )
    dist = SkillDistiller(cfg)
    comments: list[str] = []
    for v in _anchor_variants(anchor):
        try:
            d = dist._find_anchor_dir(v)
        except Exception:
            d = None
        if d is None:
            continue
        try:
            comments = dist._load_comments(d)
        except Exception:
            comments = []
        if comments:
            break

    parts: list[str] = []
    parts.append(f"主播：{anchor}")
    parts.append(f"生成时间：{_now_ts()}")
    parts.append("")
    parts.append("## 四问事实核验（本地文件）")
    if fact_files:
        for p in fact_files:
            try:
                parts.append(f"\n--- File: {p.name} ---\n")
                parts.append(p.read_text(encoding="utf-8", errors="ignore").strip())
            except Exception:
                continue
    else:
        parts.append("(未找到)")
    parts.append("")
    parts.append("## 粉丝团与关系网（本地文件）")
    if net_files:
        for p in net_files:
            try:
                parts.append(f"\n--- File: {p.name} ---\n")
                parts.append(p.read_text(encoding="utf-8", errors="ignore").strip())
            except Exception:
                continue
    else:
        parts.append("(未找到)")
    parts.append("")
    parts.append("## 评论区（2060/Workstation/本地抓取 txt 去重节选）")
    if comments:
        parts.extend([f"- {x}" for x in comments])
    else:
        parts.append("(未找到对应主播目录或没有 txt)")
    text = "\n".join(parts).strip() + "\n"

    out_dir = (Path(__file__).resolve().parents[1] / "cache" / "pipeline").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[\\\\/:*?\"<>|\\s]+", "_", anchor.strip())[:60] or "anchor"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{safe}_{ts}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path, text


async def _try_attach_file_doubao(page, file_path: Path, stop_event: threading.Event | None = None) -> bool:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("用户手动停止")
    p = str(file_path)
    try:
        inp = page.locator("input[type='file']").first
        if await inp.count() > 0:
            await _await_or_stop(inp.set_input_files(p), 10.0, stop_event)
            return True
    except Exception:
        pass
    clickers = [
        "button:has-text(\"上传\")",
        "button:has-text(\"附件\")",
        "button[aria-label*='上传' i]",
        "button[aria-label*='附件' i]",
        "[role='button']:has-text(\"上传\")",
        "[role='button']:has-text(\"附件\")",
    ]
    for sel in clickers:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible(timeout=800):
                await _await_or_stop(loc.click(timeout=1200), 2.5, stop_event)
                await asyncio.sleep(0.4)
                inp = page.locator("input[type='file']").first
                if await inp.count() > 0:
                    await _await_or_stop(inp.set_input_files(p), 10.0, stop_event)
                    return True
        except Exception:
            continue
    return False


def _build_chain_prompts(anchor: str, material_path: Path, material_text: str, doubao_used_attachment: bool) -> dict[str, str]:
    header = f"主播：{anchor}\n"
    if doubao_used_attachment:
        material_block = f"（附件已上传：{material_path.name}）"
    else:
        material_block = "\n\n【材料原文】\n" + material_text
    p1 = (
        "你现在只做一件事：从材料中提取“整蛊素材”。\n"
        "输出为结构化列表，严格按以下字段：\n"
        "1) 黑称/绰号：最常见的 3 个（每个写：称呼｜谁在用（粉/黑/路人）｜触发语境｜关键词）\n"
        "2) 出丑/翻车事件：最经典的 5 条（每条写：事件一句话｜发生场景（PK/带货/连麦/直播）｜可检索关键词｜材料引用句）\n"
        "3) 粉丝爱刷的梗：最常见的 10 条（每条写：梗｜怎么用｜可检索关键词｜材料引用句）\n"
        "要求：只基于材料；不够就写“待核实：关键词…”。不要输出任何解释。\n"
        + header
        + material_block
    )
    p2 = (
        "已知这个中国主播的特点如下（来自上一步提取）：\n"
        "【整蛊素材】\n"
        "{JUICE}\n\n"
        "我要把他映射到“龙星球”。请选一个最合适的“鹰星球”大佬（如马斯克、Faker）作为配角。\n"
        "写一段极其夸张、充满科幻感但又一本正经的台词，让这个西方大佬在全宇宙面前承认这个中国主播的牛逼，制造强烈反差与自豪感。\n"
        "要求：全中文；不要解释；输出两段：第一段是“鹰星球大佬台词”，第二段是“旁白总结”。"
        + "\n"
        + header
    )
    p3 = (
        "你现在是短视频金牌编剧。请根据以下提供的【整蛊素材】和【宇宙高潮设定】，写一个 3 分钟的短视频脚本。\n"
        "出场人物：\n"
        "1. 主角：中部主播 {NAME}\n"
        "2. 引流大V：大主播师傅（若材料无师傅线则写“待定师傅”）\n"
        "3. 反派：对头师兄弟（若材料无对头线则写“待定对头”）\n"
        "4. 废物主人柳如意：贪财好色，拿钱办事，最后总是挨揍。\n"
        "5. AI老六：有超自然能力的小恶魔，听命于废物主人，负责施展整蛊。\n"
        "6. 宇宙大佬：{EAGLE}\n"
        "剧情结构必须严格按四幕：起/承/转/合。\n"
        "硬性要求：\n"
        "- 第二幕必须用上【整蛊素材】里的“黑称/翻车/梗”。\n"
        "- 第四幕必须把“出丑行为”重新解释成宇宙最高级的艺术/战术，狠狠打脸。\n"
        "- 全中文；输出为分镜头脚本（镜头号+画面+台词+音效/字幕）。\n"
        "\n【整蛊素材】\n{JUICE}\n\n【宇宙高潮设定】\n{COSMIC}\n"
        .replace("{NAME}", anchor)
        + "\n"
    )
    return {"doubao": p1, "grok": p2, "gemini": p3}


def _run_drama_chain_job(params: dict[str, Any], job: dict[str, Any], lock: threading.Lock, stop_event: threading.Event) -> None:
    try:
        anchor = str(params.get("anchor") or "").strip()
        if not anchor:
            raise RuntimeError("主播名不能为空")
        cdp_url = str(params.get("cdp_url") or "").strip() or "http://127.0.0.1:9222"
        save_report = bool(params.get("save_report", True))

        material_path, material_text = _load_local_material(anchor)
        _job_append_log(job, lock, "PIPE", "材料合并", f"{material_path.name}（{len(material_text)} chars）")

        async def _run():
            p, ctx = await _pw_connect_ctx(cdp_url, stop_event=stop_event)
            try:
                run_date = datetime.now().strftime("%Y-%m-%d")
                run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                outputs: dict[str, str] = {}

                doubao_provider = next((x for x in WEB_PROVIDERS if x.key == "doubao"), WEB_PROVIDERS[-1])
                grok_provider = next((x for x in WEB_PROVIDERS if x.key == "grok"), WEB_PROVIDERS[1])
                gemini_provider = next((x for x in WEB_PROVIDERS if x.key == "gemini"), WEB_PROVIDERS[0])

                page_db = await _open_provider_tab(ctx, doubao_provider, run_date, stop_event, job, lock)
                await _ensure_doubao_chat(page_db, stop_event)
                attached = await _try_attach_file_doubao(page_db, material_path, stop_event=stop_event)
                prompts = _build_chain_prompts(anchor, material_path, material_text, doubao_used_attachment=attached)
                p1 = prompts["doubao"]
                _job_update(job, lock, status="流水线：豆包榨汁", progress=0.10)
                a1_raw = await _ask_web_ai_raw_in_tab(page_db, doubao_provider, p1, "流水线-豆包", run_date, stop_event, job, lock)
                a1 = _clean_web_ai_answer(a1_raw)
                outputs["doubao"] = a1

                page_g = await _open_provider_tab(ctx, grok_provider, run_date, stop_event, job, lock)
                await _ensure_grok_ready(page_g, stop_event)
                _job_update(job, lock, status="流水线：Grok 映射", progress=0.45)
                p2 = prompts["grok"].replace("{JUICE}", a1)
                a2_raw = await _ask_web_ai_raw_in_tab(page_g, grok_provider, p2, "流水线-Grok", run_date, stop_event, job, lock)
                a2 = _clean_web_ai_answer(a2_raw)
                outputs["grok"] = a2

                page_ge = await _open_provider_tab(ctx, gemini_provider, run_date, stop_event, job, lock)
                try:
                    await _ensure_gemini_new_chat(page_ge, stop_event)
                except Exception:
                    pass
                _job_update(job, lock, status="流水线：Gemini 组装", progress=0.70)
                p3 = prompts["gemini"].replace("{JUICE}", a1).replace("{COSMIC}", a2).replace("{EAGLE}", "鹰星球大佬")
                a3_raw = await _ask_web_ai_raw_in_tab(page_ge, gemini_provider, p3, "流水线-Gemini", run_date, stop_event, job, lock)
                a3 = _clean_web_ai_answer(a3_raw)
                outputs["gemini"] = a3

                if save_report:
                    store = ObsidianStore()
                    day = datetime.now().strftime("%Y-%m-%d")
                    ts = datetime.now().strftime("%H%M%S")
                    safe_anchor = re.sub(r"[\\\\/:*?\"<>|\\s]+", "_", anchor)[:60] or "anchor"
                    rel = f"06_文案库/主播剧情流水线/{day}_{safe_anchor}_{ts}.md"
                    content = "\n".join(
                        [
                            "# 主播剧情流水线（Prompt Chain）",
                            "",
                            f"- 主播：{anchor}",
                            f"- 生成时间：{_now_ts()}",
                            f"- 材料文件：{material_path.name}",
                            "",
                            "## Step1 豆包｜整蛊素材（榨汁）",
                            outputs.get("doubao", ""),
                            "",
                            "## Step2 Grok｜宇宙角色映射",
                            outputs.get("grok", ""),
                            "",
                            "## Step3 Gemini｜三分钟脚本",
                            outputs.get("gemini", ""),
                            "",
                        ]
                    ).strip() + "\n"
                    abs_p = store.write_text(rel, content)
                    _job_append_log(job, lock, "Obsidian", "写入流水线报告", f"成功 {abs_p}")
                    outputs["obsidian_path"] = rel
                    outputs["obsidian_abs"] = str(abs_p)

                _job_update(job, lock, result=outputs)
                return outputs
            finally:
                try:
                    await p.stop()
                except Exception:
                    pass

        _job_update(job, lock, status="流水线启动", progress=0.01, running=True, error="", logs=[], shots=[], result={})
        _run_async(_run())
        _job_update(job, lock, status="完成", progress=1.0, running=False)
        _job_append_log(job, lock, "PIPE", "任务完成", "成功")
    except Exception as e:
        msg = (str(e) or type(e).__name__).strip()
        _job_update(job, lock, running=False, status="失败", error=msg)
        _job_append_log(job, lock, "PIPE", "任务失败", msg[:200])


def _extract_anchor_hint(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"(?:主播名|主播|Streamer Name)\s*[:：]\s*([^\s，。；;:：]{1,24})", t, flags=re.I)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _suspect_wrong_anchor(text: str, anchor: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    a = (anchor or "").strip()
    if not a:
        return False
    hint = _extract_anchor_hint(t)
    if hint and hint != a:
        return True
    return False


def _run_fact_job(params: dict[str, Any], job: dict[str, Any], lock: threading.Lock, stop_event: threading.Event) -> None:
    run_date = datetime.now().strftime("%Y-%m-%d")
    _job_update(job, lock, running=True, progress=0.0, status="启动", logs=[], shots=[], result={}, error="", run_date=run_date)
    _job_append_log(job, lock, "SYSTEM", "启动任务", "开始")
    try:
        platform = params["platform"]
        anchor = params["anchor"]
        extra = params["prompt"]
        cdp_url = params["cdp_url"]
        model_dir = params["model_dir"]
        gpu_memory_utilization = float(params.get("gpu_memory_utilization", params.get("cache_max_entry_count", 0.85)))
        session_len = int(params.get("session_len", 16384))
        offload = bool(params.get("offload", True))
        save_report = bool(params.get("save_report", True))
        auto_close_old_ai_tabs = bool(params.get("auto_close_old_ai_tabs", True))
        open_new_ai_tabs = bool(params.get("open_new_ai_tabs", False))
        backend = str(params.get("backend") or "pytorch")
        quant_policy = int(params.get("quant_policy") or 4)
        flash_attn = bool(params.get("flash_attn", True))
        ollama_model = str(params.get("ollama_model") or DEFAULT_OLLAMA_MODEL)
        ollama_num_gpu = int(params.get("ollama_num_gpu") or 999)
        ollama_num_batch = int(params.get("ollama_num_batch") or 16)
        max_round = int(params.get("max_round") or 2)
        if max_round < 1:
            max_round = 1
        if max_round > 2:
            max_round = 2
        round2_conflict_timeout_s = int(params.get("round2_conflict_timeout_s") or 240)
        auto_chain_fansnet = bool(params.get("auto_chain_fansnet", False))
        auto_close_after_run = bool(params.get("auto_close_after_run", False))

        if stop_event.is_set():
            raise RuntimeError("用户手动停止")
        prompt_profile0 = str(params.get("prompt_profile") or "四问事实核验")
        profiles: list[str] = []
        try:
            x = params.get("prompt_profiles")
            if isinstance(x, list):
                profiles = [str(i) for i in x if str(i).strip()]
        except Exception:
            profiles = []
        if not profiles:
            profiles = [prompt_profile0]
        if auto_chain_fansnet and (prompt_profile0 == "四问事实核验") and ("粉丝团与关系网" not in profiles):
            profiles.append("粉丝团与关系网")

        conda_env = str(params.get("conda_env") or "lmdeploy-qwen35-27b-4bit")
        qwen = LocalLMDeploy(
            model_dir=model_dir,
            cache_max_entry_count=gpu_memory_utilization,
            session_len=session_len,
            offload=offload,
            conda_env=conda_env,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            quant_policy=quant_policy,
            flash_attn=flash_attn,
            ollama_model=ollama_model,
            ollama_num_gpu=ollama_num_gpu,
            ollama_num_batch=ollama_num_batch,
        )
        try:
            _ = _call_blocking_with_stop(lambda: qwen.generate(prompt="只回复：ok", system="", options={"temperature": 0.0, "num_predict": 8}, stop_event=stop_event), stop_event)
        except Exception as e:
            raise RuntimeError(f"LMDeploy 本地模型不可用：{e}")

        intent_profile = ""
        intent_profile_obj: dict[str, Any] | None = None
        if (extra or "").strip():
            _job_update(job, lock, status="深度意图识别", progress=0.01)
            try:
                intent_profile = _call_blocking_with_stop(lambda: _llm_intent_profile_stream(qwen, platform, anchor, extra, stop_event), stop_event)
                intent_profile = _strip_think(intent_profile)
                intent_profile_obj = _extract_json_object(intent_profile)
                if intent_profile_obj is None:
                    fixed = _call_blocking_with_stop(
                        lambda: _repair_intent_profile_json_stream(qwen, platform, anchor, intent_profile, stop_event),
                        stop_event,
                    )
                    fixed = _strip_think(fixed)
                    fixed_obj = _extract_json_object(fixed)
                    if fixed_obj is not None:
                        intent_profile = fixed
                        intent_profile_obj = fixed_obj
                _job_append_log(job, lock, "奇问", "意图识别", "成功")
                extra_payload = intent_profile.strip()
                if intent_profile_obj is not None:
                    extra_payload = json.dumps(intent_profile_obj, ensure_ascii=False, indent=2)
                extra = (extra.strip() + "\n\n" + "【意图画像JSON】\n" + extra_payload).strip()
            except Exception as e:
                _job_append_log(job, lock, "奇问", "意图识别失败", str(e)[:160])

        reports: list[dict[str, Any]] = []
        for prof_i, prompt_profile in enumerate(profiles, start=1):
            fixed_questions = _build_profile_questions(prompt_profile, anchor)
            _job_update(job, lock, status=f"{prompt_profile}：首轮提问 ({prof_i}/{len(profiles)})", progress=0.02)
            research_prompt = fixed_questions
            _job_append_log(job, lock, "奇问", "首轮提问", prompt_profile)

            async def _run_web_rounds():
                p, ctx = await _pw_connect_ctx(cdp_url, stop_event=stop_event)
                try:
                    if open_new_ai_tabs:
                        await _cleanup_script_pages(ctx, stop_event, job, lock)
                    elif auto_close_old_ai_tabs:
                        await _cleanup_old_ai_pages(ctx, stop_event, job, lock)
                    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out1: dict[str, str] = {}
                    sent_round1: dict[str, str] = {}
                    for i, wp in enumerate(WEB_PROVIDERS, start=1):
                        if stop_event.is_set():
                            raise RuntimeError("用户手动停止")
                        _job_update(job, lock, status=f"{prompt_profile}：首轮准备 {wp.title} ({i}/3)", progress=min(0.20, 0.05 + 0.15 * (i / 3.0)))
                        force_new = (wp.key in ("gemini", "grok"))
                        if force_new and not open_new_ai_tabs:
                            _job_append_log(job, lock, wp.title, "窗口策略", "强制新开（避免旧会话降智）")
                        try:
                            page1 = await (
                                _open_provider_tab(ctx, wp, run_date, stop_event, job, lock)
                                if (open_new_ai_tabs or force_new)
                                else _get_or_open_provider_page(ctx, wp, run_date, stop_event, job, lock)
                            )
                        except Exception as e:
                            out1[wp.key] = f"(窗口失败：{e})"
                            _job_append_log(job, lock, wp.title, "窗口失败", str(e)[:160])
                            continue
                        try:
                            _job_append_log(job, lock, wp.title, "窗口URL", (page1.url or "")[:180])
                        except Exception:
                            pass
                        if open_new_ai_tabs:
                            await _mark_script_page(page1, run_tag, wp.key, "r1", stop_event)
                        _job_update(job, lock, status=f"{prompt_profile}：首轮请求 {wp.title} ({i}/3)", progress=min(0.50, 0.20 + 0.30 * (i / 3.0)))
                        try:
                            q1 = _sanitize_web_question(wp.key, anchor, fixed_questions)
                            sent_round1[wp.key] = q1
                            a1 = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page1, wp, q1, "首轮", run_date, stop_event, job, lock))
                            if _suspect_wrong_anchor(a1, anchor):
                                if wp.key == "gemini":
                                    _job_append_log(job, lock, wp.title, "首轮疑似串主播", "强制新开窗口重答")
                                    try:
                                        page1b = await _open_provider_tab(ctx, wp, run_date, stop_event, job, lock)
                                    except Exception as e:
                                        out1[wp.key] = f"(首轮疑似串主播，且重试窗口失败：{e})"
                                        _job_append_log(job, lock, wp.title, "首轮重试窗口失败", str(e)[:160])
                                        continue
                                    try:
                                        _job_append_log(job, lock, wp.title, "重试窗口URL", (page1b.url or "")[:180])
                                    except Exception:
                                        pass
                                    a1b = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page1b, wp, q1, "首轮重试", run_date, stop_event, job, lock))
                                    try:
                                        await page1.close()
                                    except Exception:
                                        pass
                                    page1 = page1b
                                elif wp.key == "doubao":
                                    _job_append_log(job, lock, wp.title, "首轮疑似串主播", "原窗口重置后重答（不再开新窗口）")
                                    await _reset_provider_in_place(page1, wp.key, stop_event)
                                    a1b = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page1, wp, q1, "首轮重试", run_date, stop_event, job, lock))
                                else:
                                    _job_append_log(job, lock, wp.title, "首轮疑似串主播", "自动重试新窗口重答")
                                    try:
                                        page1b = await _open_provider_tab(ctx, wp, run_date, stop_event, job, lock)
                                    except Exception as e:
                                        out1[wp.key] = f"(首轮疑似串主播，且重试窗口失败：{e})"
                                        _job_append_log(job, lock, wp.title, "首轮重试窗口失败", str(e)[:160])
                                        continue
                                    a1b = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page1b, wp, q1, "首轮重试", run_date, stop_event, job, lock))
                                    try:
                                        await page1.close()
                                    except Exception:
                                        pass
                                    page1 = page1b
                                if not _suspect_wrong_anchor(a1b, anchor):
                                    a1 = a1b
                            out1[wp.key] = a1
                        except Exception as e:
                            out1[wp.key] = f"(首轮失败：{e})"
                            _job_append_log(job, lock, wp.title, "首轮失败", str(e)[:160])

                    if max_round <= 1:
                        _job_append_log(job, lock, "SYSTEM", "跳过二次追问", "fast")
                        if auto_close_after_run:
                            await _close_ai_pages_keep_first(ctx, stop_event, job, lock)
                        return out1, sent_round1, "", {}, {}, {}

                    _job_update(job, lock, status=f"{prompt_profile}：生成二次追问", progress=0.55)
                    try:
                        followup_text = await asyncio.wait_for(
                            asyncio.to_thread(_qwen_find_conflicts_stream, qwen, anchor, fixed_questions, out1, stop_event),
                            timeout=float(round2_conflict_timeout_s),
                        )
                        followup_text = _strip_think(followup_text)
                        _job_append_log(job, lock, "奇问", "生成二次追问", "成功")
                        missing = [k for k in ["gemini", "grok", "doubao"] if not _extract_provider_section(followup_text, k).strip()]
                        if missing:
                            _job_append_log(job, lock, "奇问", "二次追问格式修复", "缺少区块 " + ",".join(missing))
                            followup_text = await asyncio.to_thread(_qwen_repair_followup_stream, qwen, anchor, followup_text, stop_event)
                            followup_text = _strip_think(followup_text)
                            _job_append_log(job, lock, "奇问", "二次追问格式修复", "成功")
                    except Exception as e:
                        _job_append_log(job, lock, "奇问", "生成二次追问失败", str(e)[:160])
                        followup_text = "\n".join(
                            [
                                "这是二次核验问答任务：请针对你刚才的回答进行自检与补充。",
                                "1) 列出你上一轮回答中所有可能存在冲突/不确定/来源不清的点。",
                                "2) 对每个点补充【依据】（链接/标题/时间点/关键词）。",
                                "3) 若无法核验，请标注【待核实】并给出可执行的核验路径。",
                                "硬性要求：全中文；不要生成提示词/模板；不要英文；不要代码块。",
                            ]
                        )

                    out2: dict[str, str] = {}
                    followup_by_provider: dict[str, str] = {}
                    sent_round2: dict[str, str] = {}
                    for i, wp in enumerate(WEB_PROVIDERS, start=1):
                        if stop_event.is_set():
                            raise RuntimeError("用户手动停止")
                        _job_update(job, lock, status=f"{prompt_profile}：二次准备 {wp.title} ({i}/3)", progress=min(0.65, 0.60 + 0.05 * (i / 3.0)))
                        force_new = (wp.key in ("gemini", "grok"))
                        if force_new and not open_new_ai_tabs:
                            _job_append_log(job, lock, wp.title, "窗口策略", "强制新开（避免旧会话降智）")
                        try:
                            page2 = await (
                                _open_provider_tab(ctx, wp, run_date, stop_event, job, lock)
                                if (open_new_ai_tabs or force_new)
                                else _get_or_open_provider_page(ctx, wp, run_date, stop_event, job, lock)
                            )
                        except Exception as e:
                            out2[wp.key] = f"(窗口失败：{e})"
                            _job_append_log(job, lock, wp.title, "窗口失败", str(e)[:160])
                            continue
                        try:
                            _job_append_log(job, lock, wp.title, "窗口URL", (page2.url or "")[:180])
                        except Exception:
                            pass
                        if open_new_ai_tabs:
                            await _mark_script_page(page2, run_tag, wp.key, "r2", stop_event)
                        provider_followup = _extract_provider_section(followup_text, wp.key)
                        if not provider_followup.strip():
                            _job_append_log(job, lock, wp.title, "二次追问提取失败", "使用自动生成追问")
                            provider_followup = _fallback_followup_for_provider(wp.key, anchor, out1)
                        provider_followup = (provider_followup or "").strip() + _cross_ref_block(wp.key, out1)
                        followup_by_provider[wp.key] = provider_followup
                        _job_update(job, lock, status=f"{prompt_profile}：二次请求 {wp.title} ({i}/3)", progress=min(0.90, 0.65 + 0.25 * (i / 3.0)))
                        try:
                            q2 = _sanitize_web_question(wp.key, anchor, provider_followup)
                            sent_round2[wp.key] = q2
                            a2 = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page2, wp, q2, "二次", run_date, stop_event, job, lock))
                            if wp.key == "doubao" and ("请你提供" in a2 and ("Gemini" in a2 or "Grok" in a2 or "豆包" in a2)):
                                _job_append_log(job, lock, wp.title, "二次回复异常", "触发一次重试（避免索要原始回答）")
                                try:
                                    await _reset_provider_in_place(page2, "doubao", stop_event)
                                except Exception:
                                    pass
                                q2b = q2 + "\n你已经收到对照材料，不要向我索要其它AI原始回答；直接按四个问题逐条给结论与依据。"
                                sent_round2[wp.key] = q2b
                                a2 = _clean_web_ai_answer(await _ask_web_ai_raw_in_tab(page2, wp, q2b, "二次重试", run_date, stop_event, job, lock))
                            out2[wp.key] = a2
                        except Exception as e:
                            out2[wp.key] = f"(二次失败：{e})"
                            _job_append_log(job, lock, wp.title, "二次失败", str(e)[:160])

                    if auto_close_after_run:
                        await _close_ai_pages_keep_first(ctx, stop_event, job, lock)
                    return out1, sent_round1, followup_text, followup_by_provider, out2, sent_round2
                finally:
                    try:
                        await p.stop()
                    except Exception:
                        pass

            _job_update(job, lock, status=f"{prompt_profile}：三方询问", progress=0.05)
            r1, sent_round1, followup, followup_by_provider, r2, sent_round2 = _run_async(_run_web_rounds())

            if stop_event.is_set():
                raise RuntimeError("用户手动停止")

            _job_update(job, lock, status=f"{prompt_profile}：生成最终报告", progress=0.92)
            try:
                report = _call_blocking_with_stop(lambda: _qwen_final_report_stream(qwen, platform, anchor, r1, r2, stop_event), stop_event)
                report = _strip_think(report)
                if not (report or "").strip():
                    raise RuntimeError("最终报告为空")
                _job_append_log(job, lock, "奇问", "生成最终报告", "成功")
            except Exception as e:
                report = f"最终报告生成失败：{type(e).__name__}: {str(e) or ''}".strip()
                _job_append_log(job, lock, "奇问", "生成最终报告失败", (str(e) or type(e).__name__)[:160])

            obsidian_path = ""
            obsidian_abs = ""
            if save_report:
                store = ObsidianStore()
                day = datetime.now().strftime("%Y-%m-%d")
                ts = datetime.now().strftime("%H%M%S")
                safe_profile = re.sub(r"[\\\\/:*?\"<>|\\s]+", "_", (prompt_profile or "默认").strip())[:40] or "默认"
                rel = f"06_文案库/主播事实核验/{safe_profile}/{day}_{platform}_{anchor}_{ts}.md"
                content = "\n".join(
                    [
                        "# 主播流量节奏分析与事实核验报告",
                        "",
                        f"- 平台：{platform}",
                        f"- 主播：{anchor}",
                        f"- 生成时间：{_now_ts()}",
                        "",
                        "## 深度意图识别",
                        (json.dumps(intent_profile_obj, ensure_ascii=False, indent=2) if intent_profile_obj is not None else (intent_profile or "(未生成)")),
                        "",
                        "## 首轮提问提示词",
                        research_prompt,
                        "",
                        "## 首轮三方回答",
                        "### Gemini",
                        r1.get("gemini", ""),
                        "",
                        "### Grok",
                        r1.get("grok", ""),
                        "",
                        "### 豆包",
                        r1.get("doubao", ""),
                        "",
                        "## 二次追问提示词",
                        followup,
                        "",
                        "## 二次追问提示词（分平台）",
                        "### Gemini",
                        (followup_by_provider.get("gemini", "") if isinstance(followup_by_provider, dict) else ""),
                        "",
                        "### Grok",
                        (followup_by_provider.get("grok", "") if isinstance(followup_by_provider, dict) else ""),
                        "",
                        "### 豆包",
                        (followup_by_provider.get("doubao", "") if isinstance(followup_by_provider, dict) else ""),
                        "",
                        "## 二次三方回答",
                        "### Gemini",
                        r2.get("gemini", ""),
                        "",
                        "### Grok",
                        r2.get("grok", ""),
                        "",
                        "### 豆包",
                        r2.get("doubao", ""),
                        "",
                        "## 最终报告",
                        report,
                        "",
                    ]
                )
                try:
                    p = store.write_text(rel, content)
                    obsidian_path = rel
                    obsidian_abs = str(p)
                    _job_append_log(job, lock, "Obsidian", "写入报告", f"成功 {prompt_profile}")
                except Exception as e:
                    _job_append_log(job, lock, "Obsidian", "写入报告", f"失败 {prompt_profile} {e}")

            reports.append({"profile": prompt_profile, "obsidian_path": obsidian_path, "obsidian_abs": obsidian_abs})

            _job_update(
                job,
                lock,
                result={
                    "research_prompt": research_prompt,
                    "sent_round1": sent_round1,
                    "sent_round2": sent_round2,
                    "followup": followup,
                    "followup_by_provider": followup_by_provider,
                    "r1": r1,
                    "r2": r2,
                    "report": report,
                    "obsidian_path": obsidian_path,
                    "obsidian_abs": obsidian_abs,
                    "reports": reports,
                },
            )

        _job_update(job, lock, status="完成", progress=1.0)
        _job_append_log(job, lock, "SYSTEM", "任务完成", "成功")
    except Exception as e:
        msg = (str(e) or "").strip()
        name = type(e).__name__
        err = f"{name}: {msg}" if msg else name
        _job_update(job, lock, error=err, status="中止" if ("停止" in err) else "失败")
        _job_append_log(job, lock, "SYSTEM", "任务失败", err)
        try:
            tb = traceback.format_exc().strip().splitlines()
            tail = "\n".join(tb[-18:]).strip()
            if tail:
                _job_append_log(job, lock, "TRACE", "异常堆栈", tail)
        except Exception:
            pass
    finally:
        _job_update(job, lock, running=False)


st.write("Debug: 阶段 1（进入页面渲染区）已通过")
st.header("老六创作系统")
st.write("Debug: 阶段 2（header 已渲染）已通过")

try:
    _job_state_init()
except Exception as e:
    st.write("Debug: 阶段 2.1（job_state_init 异常）", type(e).__name__, str(e)[:240])
    try:
        st.session_state.fact_job = {
            "running": False,
            "progress": 0.0,
            "status": "",
            "logs": [],
            "shots": [],
            "result": {},
            "error": "",
            "run_date": "",
        }
        st.session_state.fact_job_lock = threading.Lock()
        st.session_state.fact_job_stop = threading.Event()
    except Exception as e2:
        st.write("Debug: 阶段 2.2（session_state 写入失败）", type(e2).__name__, str(e2)[:240])
st.write("Debug: 阶段 3（job_state_init 已通过）已通过")

st.subheader("快速生成")
def _parse_cha_ui_command(cmd: str) -> tuple[str, str, str] | None:
    t = (cmd or "").strip()
    if not t:
        return None
    t = re.sub(r"[\r\n\t]+", " ", t).strip()
    if t.startswith("查"):
        t = t[1:].strip()
    t = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"^(抖音|快手)(?:主播)?\s+(.+?)(?:\s+(四问|关系网|粉丝团|粉丝团与关系网|四问事实核验))?$", t)
    if not m:
        return None
    plat = m.group(1).strip()
    anchor_raw = (m.group(2) or "").strip()
    if not anchor_raw:
        return None
    anchor = re.split(r"[（(]", anchor_raw, 1)[0].strip() or anchor_raw
    hint = (m.group(3) or "").strip()
    profile = "四问事实核验"
    if hint in ("关系网", "粉丝团", "粉丝团与关系网"):
        profile = "粉丝团与关系网"
    return plat, anchor, profile

cmd_col1, cmd_col2 = st.columns([3, 1])
with cmd_col1:
    cha_cmd = st.text_input("一键命令", value="", placeholder="查 抖音主播 XXX / 查 快手主播 XXX", key="cha_cmd")
with cmd_col2:
    cha_run = st.button("执行并写入", disabled=bool(st.session_state.fact_job.get("running")))

col1, col2 = st.columns(2)
with col1:
    conda_env = st.text_input("Conda 环境名", value="lmdeploy-qwen35-27b-4bit")
with col2:
    model_dir = st.text_input(
        "本地模型目录",
        value=r"F:\老六个人 AI 工作台\llm_engine\models\Qwen__Qwen3.5-27B-GPTQ-Int4",
    )

cdp_url = st.text_input("CDP 地址", value="http://127.0.0.1:9222")
backend = st.selectbox("backend", options=["ollama"], index=0)
ollama_model = st.text_input("Ollama 模型名", value=DEFAULT_OLLAMA_MODEL)
ollama_num_gpu = st.number_input("Ollama num_gpu（GPU层数）", min_value=0, max_value=999, value=999, step=1)
ollama_num_batch = st.number_input("Ollama num_batch", min_value=1, max_value=256, value=16, step=1)
gpu_memory_utilization = st.number_input("gpu_memory_utilization（仅 Pytorch）", min_value=0.20, max_value=0.95, value=0.85, step=0.05)
quant_policy = st.selectbox("quant_policy（仅 Pytorch / KV INT4=4）", options=[0, 4, 8], index=1)
flash_attn = st.toggle("flash_attn（仅 Pytorch）", value=True)
offload = st.toggle("cpu_offload（KV Cache Offload）", value=True)
session_len = st.selectbox("context_window", options=[8192, 16384, 32768, 64000], index=1)
save_report = st.toggle("生成后写入 Obsidian 报告", value=True)
auto_close_old_ai_tabs = st.toggle("生成前自动关闭旧 AI 窗口", value=True)
open_new_ai_tabs = st.toggle("每轮新开 AI 窗口（下次生成时自动清理）", value=False)
auto_chain_fansnet = st.toggle("完成后自动生成「粉丝团与关系网」", value=True)
auto_close_after_run = st.toggle("完成后自动关闭 9222 AI 窗口（保留第一个标签页）", value=True)

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    stop_job = st.button("停止", disabled=not bool(st.session_state.fact_job.get("running")))
with run_col2:
    st.caption("执行中会按步骤记录日志与截图（logs/ui_trace/日期/平台/）。")

if stop_job:
    st.session_state.fact_job_stop.set()
    _job_update(st.session_state.fact_job, st.session_state.fact_job_lock, status="停止中…")
    _job_append_log(st.session_state.fact_job, st.session_state.fact_job_lock, "SYSTEM", "用户手动停止", "触发")
    st.rerun()

test_col1, test_col2 = st.columns([1, 3])
with test_col1:
    do_test_local = st.button("测试本地模型")
with test_col2:
    st.caption("提示：首次加载 27B 量化模型会很慢，测试会启动一次最短生成。")

if do_test_local:
    if not model_dir.strip():
        st.warning("请填写本地模型目录。")
    else:
        try:
            llm = LocalLMDeploy(
                model_dir=model_dir.strip(),
                cache_max_entry_count=float(gpu_memory_utilization),
                session_len=int(session_len),
                offload=bool(offload),
                conda_env=conda_env.strip() or "lmdeploy-qwen35-27b-4bit",
                backend=backend,
                gpu_memory_utilization=float(gpu_memory_utilization),
                quant_policy=int(quant_policy),
                flash_attn=bool(flash_attn),
                ollama_model=ollama_model.strip(),
                ollama_num_gpu=int(ollama_num_gpu),
                ollama_num_batch=int(ollama_num_batch),
            )
            out = llm.generate(prompt="只回复：ok", system="", options={"temperature": 0.0, "num_predict": 8}, stop_event=threading.Event())
            st.success(f"本地模型可用：{out}")
        except Exception as e:
            st.warning(str(e))

st.subheader("主播剧情流水线（Prompt Chain）")
if "drama_job" not in st.session_state:
    st.session_state.drama_job = {"running": False, "progress": 0.0, "status": "", "logs": [], "result": {}, "error": ""}
if "drama_job_lock" not in st.session_state:
    st.session_state.drama_job_lock = threading.Lock()
if "drama_job_stop" not in st.session_state:
    st.session_state.drama_job_stop = threading.Event()

pc1, pc2, pc3 = st.columns([2, 2, 1])
with pc1:
    drama_anchor = st.text_input("主播名（流水线）", value="", placeholder="例如：水姐（4点播）", key="drama_anchor")
with pc2:
    drama_cdp = st.text_input("CDP 地址（流水线）", value=cdp_url, key="drama_cdp")
with pc3:
    drama_save = st.toggle("写入 Obsidian", value=True, key="drama_save")

btn_a, btn_b, btn_c = st.columns([1, 1, 2])
with btn_a:
    drama_run = st.button("一键触发流水线", disabled=bool(st.session_state.drama_job.get("running")) or (not bool(drama_anchor.strip())))
with btn_b:
    drama_stop = st.button("停止流水线", disabled=not bool(st.session_state.drama_job.get("running")))
with btn_c:
    st.caption("顺序：豆包榨汁 → Grok 映射 → Gemini 组装；材料来自本地 Obsidian + Workstation 评论 txt。")

if drama_stop:
    st.session_state.drama_job_stop.set()
    _job_append_log(st.session_state.drama_job, st.session_state.drama_job_lock, "PIPE", "用户手动停止", "触发")
    st.rerun()

if drama_run:
    st.session_state.drama_job_stop.clear()
    params = {"anchor": drama_anchor.strip(), "cdp_url": drama_cdp.strip(), "save_report": bool(drama_save)}
    t = threading.Thread(
        target=_run_drama_chain_job,
        args=(params, st.session_state.drama_job, st.session_state.drama_job_lock, st.session_state.drama_job_stop),
        daemon=True,
    )
    t.start()
    st.rerun()

djob = st.session_state.drama_job
st.progress(float(djob.get("progress") or 0.0))
st.write(djob.get("status") or "")
if djob.get("error"):
    st.warning(djob.get("error"))
try:
    logs = djob.get("logs") or []
    if isinstance(logs, list) and logs:
        st.text_area("流水线日志", value="\n".join(str(x) for x in logs[-200:]), height=220)
except Exception:
    pass
res = djob.get("result") or {}
if isinstance(res, dict) and res:
    if res.get("obsidian_abs"):
        st.text_input("Obsidian 文件路径（流水线）", value=str(res.get("obsidian_abs")), disabled=True)
    if res.get("doubao"):
        st.text_area("Step1 豆包输出", value=str(res.get("doubao")), height=140)
    if res.get("grok"):
        st.text_area("Step2 Grok 输出", value=str(res.get("grok")), height=140)
    if res.get("gemini"):
        st.text_area("Step3 Gemini 输出", value=str(res.get("gemini")), height=220)

job = st.session_state.fact_job
progress = st.progress(float(job.get("progress") or 0.0))
st.write(job.get("status") or "")

if cha_run:
    st.session_state.fact_job_stop.clear()
    parsed = _parse_cha_ui_command(cha_cmd)
    if not parsed:
        st.warning("命令格式不对：查 抖音主播 XXX / 查 快手主播 XXX（可选加 关系网/粉丝团/四问）")
    else:
        plat, anc, prof = parsed
        ok_cdp, msg_cdp = _cdp_ready(cdp_url)
        if not ok_cdp:
            st.warning(f"9222 不可用：{msg_cdp}")
        else:
            _job_update(
                st.session_state.fact_job,
                st.session_state.fact_job_lock,
                running=True,
                status="排队中",
                progress=0.0,
                error="",
                logs=[],
                shots=[],
                result={},
            )
            t = threading.Thread(
                target=_run_fact_job,
                args=(
                    {
                        "platform": plat,
                        "anchor": anc,
                        "prompt_profile": prof,
                        "prompt": "",
                        "cdp_url": cdp_url,
                        "model_dir": model_dir.strip(),
                        "gpu_memory_utilization": float(gpu_memory_utilization),
                        "session_len": int(session_len),
                        "offload": bool(offload),
                        "save_report": True,
                        "auto_close_old_ai_tabs": True,
                        "open_new_ai_tabs": False,
                        "auto_chain_fansnet": True,
                        "auto_close_after_run": True,
                        "backend": backend,
                        "quant_policy": int(quant_policy),
                        "ollama_model": ollama_model.strip(),
                        "ollama_num_gpu": int(ollama_num_gpu),
                        "ollama_num_batch": int(ollama_num_batch),
                        "flash_attn": bool(flash_attn),
                        "conda_env": conda_env.strip() or "lmdeploy-qwen35-27b-4bit",
                        "max_round": 2,
                        "round2_conflict_timeout_s": 240,
                    },
                    st.session_state.fact_job,
                    st.session_state.fact_job_lock,
                    st.session_state.fact_job_stop,
                ),
                daemon=True,
            )
            t.start()
            st.rerun()

with st.expander("操作日志", expanded=True):
    logs = job.get("logs") or []
    st.text_area("日志", value="\n".join(logs[-300:]), height=220)

with st.expander("截图路径", expanded=False):
    shots = job.get("shots") or []
    st.text_area("截图", value="\n".join(shots[-200:]), height=180)

clean_col1, clean_col2 = st.columns([1, 1])
with clean_col1:
    clear_ui_logs = st.button("清空本页日志/截图列表", disabled=bool(job.get("running")))
with clean_col2:
    delete_trace = st.button("删除 ui_trace 截图目录", disabled=bool(job.get("running")))

if clear_ui_logs:
    _job_update(st.session_state.fact_job, st.session_state.fact_job_lock, logs=[], shots=[])
    st.rerun()

if delete_trace:
    try:
        trace_root = _root_dir() / "logs" / "ui_trace"
        if trace_root.exists():
            shutil.rmtree(trace_root)
        trace_root.mkdir(parents=True, exist_ok=True)
        _job_update(st.session_state.fact_job, st.session_state.fact_job_lock, logs=[], shots=[])
        st.success("已删除 logs/ui_trace 下所有截图")
        st.rerun()
    except Exception as e:
        st.warning(f"删除失败：{e}")

err = job.get("error") or ""
if err:
    st.warning(err)

res = job.get("result") or {}
if res:
    st.subheader("首轮提问提示词")
    st.text_area("首轮提问提示词", value=res.get("research_prompt", ""), height=220)
    st.subheader("首轮发送内容（分平台）")
    s1 = res.get("sent_round1") or {}
    for wp in WEB_PROVIDERS:
        st.text_area(f"{wp.title}（首轮发送）", value=s1.get(wp.key, ""), height=180)
    st.subheader("首轮回答")
    r1 = res.get("r1") or {}
    for wp in WEB_PROVIDERS:
        st.text_area(f"{wp.title}（首轮）", value=r1.get(wp.key, ""), height=220)
    st.subheader("二次追问提示词")
    st.text_area("二次追问提示词", value=res.get("followup", ""), height=220)
    st.subheader("二次追问提示词（分平台）")
    fup = res.get("followup_by_provider") or {}
    for wp in WEB_PROVIDERS:
        st.text_area(f"{wp.title}（二次追问）", value=fup.get(wp.key, ""), height=180)
    st.subheader("二次发送内容（分平台）")
    s2 = res.get("sent_round2") or {}
    for wp in WEB_PROVIDERS:
        st.text_area(f"{wp.title}（二次发送）", value=s2.get(wp.key, ""), height=180)
    st.subheader("二次回答")
    r2 = res.get("r2") or {}
    for wp in WEB_PROVIDERS:
        st.text_area(f"{wp.title}（二次）", value=r2.get(wp.key, ""), height=220)
    st.subheader("事实核查报告")
    st.text_area("报告", value=res.get("report", ""), height=320)
    if res.get("obsidian_path"):
        st.success(f"已写入 Obsidian：{res.get('obsidian_path')}")
    if res.get("obsidian_abs"):
        st.text_input("Obsidian 文件路径", value=res.get("obsidian_abs"), disabled=True)
    reps = res.get("reports") or []
    try:
        if isinstance(reps, list) and len(reps) >= 2:
            lines = []
            for x in reps:
                if not isinstance(x, dict):
                    continue
                prof = str(x.get("profile") or "").strip()
                pabs = str(x.get("obsidian_abs") or "").strip()
                if prof or pabs:
                    lines.append(f"{prof}\t{pabs}")
            if lines:
                st.text_area("已生成报告（多份）", value="\n".join(lines), height=90)
    except Exception:
        pass

st.subheader("保存到 Obsidian")
note_path = st.text_input("相对路径（例如：06_文案库/草稿.md）", value="06_文案库/草稿.md")
content = st.text_area("笔记内容", height=160)
if st.button("写入"):
    store = ObsidianStore()
    store.write_text(note_path, content)
    st.success("已写入")

if bool(st.session_state.fact_job.get("running")):
    time.sleep(0.6)
    st.rerun()
