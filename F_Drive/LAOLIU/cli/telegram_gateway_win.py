from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


_LOG_PATH = Path(r"F:\LAOLIU\logs\telegram_gateway_win.log")
_LOCK_PATH = Path(r"F:\LAOLIU\logs\telegram_gateway_win.lock")
_HTTP = requests.Session()
_HTTP.trust_env = False
_LOCK_FH = None


def _log(msg: str) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _LOG_PATH.open("a", encoding="utf-8").write(f"[{ts}] {msg}\n")
    except Exception:
        pass


@dataclass(frozen=True)
class ChatContext:
    chat_id: int
    user_id: int
    message_id: int
    text: str


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _get_token() -> str:
    token = _env("TELEGRAM_BOT_TOKEN") or _env("HERMES_TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("missing TELEGRAM_BOT_TOKEN")
    return token


def _configure_proxy() -> str:
    proxy = _env("TELEGRAM_PROXY") or _env("ALL_PROXY") or _env("HTTPS_PROXY") or _env("HTTP_PROXY")
    proxy = (proxy or "").strip()
    if not proxy:
        _HTTP.proxies = {}
        return ""
    if "host.docker.internal" in proxy:
        proxy = proxy.replace("host.docker.internal", "127.0.0.1")
    _HTTP.proxies = {"http": proxy, "https": proxy}
    return proxy


def _allowed_users() -> set[int]:
    raw = _env("TELEGRAM_ALLOWED_USERS") or _env("HERMES_TELEGRAM_ALLOWED_USERS")
    out: set[int] = set()
    for part in re.split(r"[,\s]+", raw):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _api_base(token: str) -> str:
    return f"https://api.telegram.org/bot{token}"


def _tg_post(token: str, method: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    url = _api_base(token) + "/" + method.lstrip("/")
    r = _HTTP.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _send_text(token: str, chat_id: int, text: str, reply_to_message_id: int | None = None) -> None:
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
        payload["allow_sending_without_reply"] = True
    _tg_post(token, "sendMessage", payload, timeout=60)


def _is_cmd(text: str, prefix: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return t == prefix or t.startswith(prefix + " ")


def _strip_alias(anchor_raw: str) -> str:
    a = (anchor_raw or "").strip()
    if not a:
        return ""
    x = re.split(r"[（(]", a, 1)[0].strip()
    return x or a


def _parse_cha(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not _is_cmd(t, "查"):
        return None
    raw = t[1:].strip()
    if not raw:
        return {"error": "用法：查 抖音主播 <名字> 或 查 快手主播 <名字>；可在末尾加 关系网/粉丝团/四问；可加 重跑/刷新；可加 一轮/单轮/不追问"}

    def has_kw(s: str, kw: str) -> bool:
        return f" {kw} " in f" {s} "

    force = has_kw(raw, "重跑") or has_kw(raw, "刷新") or has_kw(raw, "强制")
    single = has_kw(raw, "一轮") or has_kw(raw, "单轮") or has_kw(raw, "不追问")
    raw2 = raw
    for kw in ("重跑", "刷新", "强制", "一轮", "单轮", "不追问", "深度", "二轮"):
        raw2 = raw2.replace(kw, " ")
    raw2 = re.sub(r"\s+", " ", raw2).strip()

    m = re.match(r"^(抖音|快手)(?:主播)?\s+(.*?)(?:\s+(四问|关系网|粉丝团|粉丝团与关系网|四问事实核验|梗与粉丝))?$", raw2)
    if not m:
        return {"error": "用法：查 抖音主播 <名字> 或 查 快手主播 <名字>；可在末尾加 关系网/粉丝团/四问；可加 重跑/刷新；可加 一轮/单轮/不追问"}
    platform = m.group(1).strip()
    anchor_raw = re.sub(r"[\r\n\t]+", " ", (m.group(2) or "")).strip()
    if not anchor_raw:
        return {"error": "主播名不能为空"}
    anchor = _strip_alias(anchor_raw)
    hint = (m.group(3) or "").strip()
    profile = "四问事实核验"
    if hint in ("关系网", "粉丝团", "粉丝团与关系网"):
        profile = "粉丝团与关系网"
    elif hint in ("四问", "四问事实核验"):
        profile = "四问事实核验"

    return {
        "platform": platform,
        "anchor_raw": anchor_raw,
        "anchor": anchor,
        "profile": profile,
        "force": bool(force),
        "max_round": 1 if single else 2,
    }


def _run_anchor_factcheck(platform: str, anchor: str, profile: str, force: bool, max_round: int) -> dict[str, Any]:
    script = Path(r"F:\LAOLIU\cli\anchor_factcheck_cli.py")
    if not script.exists():
        script = Path(r"F:\老六个人 AI 工作台\cli\anchor_factcheck_cli.py")
    if not script.exists():
        raise RuntimeError("anchor_factcheck_cli.py 不存在")

    python_exe = sys.executable
    argv = [
        python_exe,
        str(script),
        "--platform",
        platform,
        "--anchor",
        anchor,
        "--profile",
        profile,
        "--max-round",
        str(int(max_round)),
    ]
    if force:
        argv.append("--force")
    env = os.environ.copy()
    env.setdefault("LAOLIU_OLLAMA_KEEP_ALIVE", "30m")
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")

    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    if p.returncode != 0:
        raise RuntimeError((err or out or "unknown error")[:600])
    last = out.splitlines()[-1] if out else ""
    return json.loads(last)


_CHAT_LOCKS: dict[int, threading.Lock] = {}


def _chat_lock(chat_id: int) -> threading.Lock:
    if chat_id not in _CHAT_LOCKS:
        _CHAT_LOCKS[chat_id] = threading.Lock()
    return _CHAT_LOCKS[chat_id]


def _handle_message(token: str, ctx: ChatContext, allowed: set[int]) -> None:
    if allowed and (ctx.user_id not in allowed):
        return

    cmd = _parse_cha(ctx.text)
    if cmd is None:
        return
    if "error" in cmd:
        _send_text(token, ctx.chat_id, str(cmd["error"]), reply_to_message_id=ctx.message_id)
        return

    platform = str(cmd["platform"])
    anchor = str(cmd["anchor"])
    anchor_raw = str(cmd["anchor_raw"])
    profile = str(cmd["profile"])
    force = bool(cmd["force"])
    max_round = int(cmd["max_round"])

    lock = _chat_lock(ctx.chat_id)
    if not lock.acquire(blocking=False):
        _send_text(token, ctx.chat_id, "当前有任务在跑，等它结束再发（或用 /stop 结束旧任务）。", reply_to_message_id=ctx.message_id)
        return

    def work():
        try:
            lines = [f"收到：查 {platform}主播 {anchor}"]
            if anchor_raw != anchor:
                lines.append(f"备注：{anchor_raw}")
            lines.append(f"模式：{max_round} 轮（含二次追问）")
            if force:
                lines.append("策略：强制重跑")
            lines.append("开始执行（三家对比：Gemini/Grok/豆包）…")
            _send_text(token, ctx.chat_id, "\n".join(lines), reply_to_message_id=ctx.message_id)

            res = _run_anchor_factcheck(platform=platform, anchor=anchor, profile=profile, force=force, max_round=max_round)
            if res.get("status") == "exists":
                _send_text(token, ctx.chat_id, f"已存在：{res.get('path','')}", reply_to_message_id=ctx.message_id)
                return
            p = res.get("obsidian_abs") or res.get("obsidian_path") or ""
            _send_text(token, ctx.chat_id, f"完成：{p}", reply_to_message_id=ctx.message_id)
        except Exception as e:
            _send_text(token, ctx.chat_id, f"执行失败：{type(e).__name__}: {str(e)[:900]}", reply_to_message_id=ctx.message_id)
        finally:
            try:
                lock.release()
            except Exception:
                pass

    t = threading.Thread(target=work, daemon=True)
    t.start()


def main() -> int:
    global _LOCK_FH
    token = _get_token()
    allowed = _allowed_users()
    offset = int(_env("TELEGRAM_UPDATE_OFFSET") or "0")
    poll_timeout = int(_env("TELEGRAM_POLL_TIMEOUT_S") or "45")
    poll_sleep = float(_env("TELEGRAM_POLL_SLEEP_S") or "0.6")

    try:
        import msvcrt

        _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOCK_FH = _LOCK_PATH.open("a+", encoding="utf-8")
        try:
            msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_NBLCK, 1)
        except Exception:
            return 0
    except Exception:
        pass

    proxy = _configure_proxy()
    _log("started")
    if proxy:
        _log("proxy_on")
    else:
        _log("proxy_off")
    print("telegram_gateway_win started", flush=True)

    while True:
        try:
            payload: dict[str, Any] = {"timeout": poll_timeout, "allowed_updates": ["message"]}
            if offset > 0:
                payload["offset"] = offset
            resp = _tg_post(token, "getUpdates", payload, timeout=poll_timeout + 120)
            ok = bool(resp.get("ok"))
            if not ok:
                time.sleep(2.0)
                continue
            updates = resp.get("result") or []
            for upd in updates:
                try:
                    uid = int((upd.get("update_id") or 0))
                    offset = max(offset, uid + 1)
                    msg = upd.get("message") or {}
                    text = msg.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    chat = msg.get("chat") or {}
                    frm = msg.get("from") or {}
                    ctx = ChatContext(
                        chat_id=int(chat.get("id") or 0),
                        user_id=int(frm.get("id") or 0),
                        message_id=int(msg.get("message_id") or 0),
                        text=text,
                    )
                    if ctx.chat_id and ctx.user_id and ctx.message_id:
                        _handle_message(token, ctx, allowed)
                except Exception:
                    continue
            time.sleep(poll_sleep)
        except Exception as e:
            msg = str(e)[:260]
            if token and token in msg:
                msg = msg.replace(token, "<token>")
            msg = re.sub(r"https://api\.telegram\.org/bot[^/\s]+", "https://api.telegram.org/bot<token>", msg)
            _log(f"poll_error:{type(e).__name__}:{msg}")
            try:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if int(code or 0) == 409:
                    time.sleep(15.0)
                    continue
            except Exception:
                pass
            time.sleep(2.5)
            continue


if __name__ == "__main__":
    raise SystemExit(main())
