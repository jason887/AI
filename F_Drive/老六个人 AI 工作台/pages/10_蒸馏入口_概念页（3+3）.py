from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

from modules.skill_distiller import _default_input_roots


st.set_page_config(page_title="蒸馏入口（概念页 3+3）", layout="wide")
st.header("蒸馏入口（评论 + 事实 → 概念页模板填充｜3+3）")

proj_root = Path(__file__).resolve().parents[1]
cli_script = proj_root / "cli" / "distiller_skill.py"
default_queue_path = (proj_root / "cache" / "distill_queue.txt").resolve()
default_queue_path.parent.mkdir(parents=True, exist_ok=True)

col1, col2 = st.columns([1, 2])
with col1:
    anchor = st.text_input("主播名", value="旭旭宝宝")
    track = st.text_input("分类/赛道（可选）", value="娱乐")
    model = st.text_input("Ollama 模型名", value="qwen35:9b-q4_0")
    num_predict = st.number_input("max_tokens", min_value=128, max_value=4096, value=1200, step=64)
    run_btn = st.button("开始蒸馏并写入 Obsidian", disabled=not bool(anchor.strip()))

with col2:
    out_dir1 = Path(r"F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验\粉丝团与关系网")
    out_dir2 = Path(r"F:\老六个人 AI 工作台\Obsidian知识库\06_文案库\主播事实核验\粉丝团与关系网")
    out_dir = out_dir1 if out_dir1.exists() else out_dir2
    out_path = out_dir / f"{anchor.strip() or '<主播名>'}.md"
    st.write("输出路径：")
    st.code(str(out_path), language="text")

    roots = _default_input_roots()
    st.write("评论素材根目录（自动探测，可用于确认可见性）：")
    st.code("\n".join(str(p) for p in roots) if roots else "(未探测到)", language="text")

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _clean_queue_lines(text: str) -> list[str]:
    out: list[str] = []
    seen = set()
    for raw in (text or "").splitlines():
        ln = (raw or "").strip()
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        if "；" in ln and ("主播" in ln or "抖音" in ln or "快手" in ln):
            ln = ln.split("；", 1)[0].strip()
        if ln.startswith("查"):
            ln = ln[1:].strip()
        ln = ln.replace("抖音主播", "").replace("快手主播", "").replace("主播", "").strip()
        ln = ln.strip(" -\t")
        if not ln:
            continue
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out


def _resolve_out_dir() -> Path:
    return out_dir1 if out_dir1.exists() else out_dir2


def _out_path_for(anchor_name: str) -> Path:
    return _resolve_out_dir() / f"{(anchor_name or '').strip()}.md"


def _append_batch_log(job: dict, lock: threading.Lock, msg: str) -> None:
    line = f"{_now()} {msg}"
    with lock:
        job["updated_at"] = time.time()
        logs = job.get("logs")
        if not isinstance(logs, list):
            logs = []
        logs.append(line)
        job["logs"] = logs[-500:]
        job["last"] = line


def _run_one(anchor_name: str, track_text: str, model_name: str, max_tokens: int, timeout_s: int) -> tuple[bool, str, str]:
    if not cli_script.exists():
        return False, "", f"脚本不存在：{cli_script}"
    a = (anchor_name or "").strip()
    if not a:
        return False, "", "主播名为空"
    t = (track_text or "").strip()
    env = dict(os.environ)
    if (model_name or "").strip():
        env["OLLAMA_MODEL"] = (model_name or "").strip()
    env["DISTILL_NUM_PREDICT"] = str(int(max_tokens))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    argv = [sys.executable, "-X", "utf8", str(cli_script), a]
    if t:
        argv.append(t)
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=int(timeout_s))
    except subprocess.TimeoutExpired:
        return False, "", f"超时（>{timeout_s}s）"
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    last = lines[-1] if lines else ""
    if p.returncode != 0:
        tail = (err or out or "(无输出)")[:800]
        return False, "", f"执行失败({p.returncode}) {tail}"
    try:
        j = json.loads(last)
    except Exception:
        return False, "", f"解析输出失败：{(out or '(无输出)')[:800]}"
    if str(j.get("status") or "").lower() != "ok":
        return False, "", str(j.get("error") or "蒸馏失败")
    return True, str(j.get("path") or "").strip(), str(j.get("message") or "").strip()


def _batch_worker(params: dict, job: dict, lock: threading.Lock, stop_event: threading.Event) -> None:
    try:
        anchors = params.get("anchors") or []
        track_text = str(params.get("track") or "").strip()
        model_name = str(params.get("model") or "").strip()
        max_tokens = int(params.get("max_tokens") or 1200)
        force = bool(params.get("force"))
        timeout_s = int(params.get("timeout_s") or 1200)
        sleep_s = float(params.get("sleep_s") or 0.2)

        total = len(anchors)
        _append_batch_log(job, lock, f"[Batch] 启动 total={total} force={int(force)} model={model_name or '(default)'} max_tokens={max_tokens}")
        for idx, a in enumerate(anchors, start=1):
            if stop_event.is_set():
                _append_batch_log(job, lock, "[Batch] 已停止")
                break
            out_path = _out_path_for(a)
            if (not force) and out_path.exists() and out_path.stat().st_size >= 200:
                with lock:
                    job["done"] = int(job.get("done") or 0) + 1
                    job["skipped"] = int(job.get("skipped") or 0) + 1
                    job["current"] = a
                    job["progress"] = float(job["done"]) / max(1, total)
                    job["updated_at"] = time.time()
                _append_batch_log(job, lock, f"[Skip] {a}（已存在：{out_path.name}） {idx}/{total}")
                time.sleep(sleep_s)
                continue

            with lock:
                job["current"] = a
                job["progress"] = float(int(job.get("done") or 0)) / max(1, total)
                job["updated_at"] = time.time()
            _append_batch_log(job, lock, f"[Run ] {a} {idx}/{total}")
            ok, path, msg = _run_one(a, track_text=track_text, model_name=model_name, max_tokens=max_tokens, timeout_s=timeout_s)
            if ok:
                with lock:
                    job["done"] = int(job.get("done") or 0) + 1
                    job["progress"] = float(job["done"]) / max(1, total)
                    job["last_path"] = path
                    job["updated_at"] = time.time()
                _append_batch_log(job, lock, f"[OK  ] {a} -> {path or out_path} {msg}".strip())
            else:
                with lock:
                    job["done"] = int(job.get("done") or 0) + 1
                    job["failed"] = int(job.get("failed") or 0) + 1
                    job["progress"] = float(job["done"]) / max(1, total)
                    job["updated_at"] = time.time()
                _append_batch_log(job, lock, f"[Fail] {a} {msg}".strip())
            time.sleep(sleep_s)

        with lock:
            job["running"] = False
            job["finished_at"] = _now()
            job["progress"] = float(job.get("done") or 0) / max(1, total) if total else 0.0
            job["updated_at"] = time.time()
        _append_batch_log(job, lock, "[Batch] 结束")
    except Exception as e:
        import traceback

        tb = traceback.format_exc(limit=8)
        _append_batch_log(job, lock, f"[Crash] {type(e).__name__}: {str(e)[:220]}")
        _append_batch_log(job, lock, tb.replace('\n', ' | ')[:900])
        with lock:
            job["running"] = False
            job["finished_at"] = _now()
            job["updated_at"] = time.time()


if "distill_batch_job" not in st.session_state:
    st.session_state.distill_batch_job = {"running": False, "progress": 0.0, "done": 0, "skipped": 0, "failed": 0, "current": "", "logs": []}
if "distill_batch_lock" not in st.session_state:
    st.session_state.distill_batch_lock = threading.Lock()
if "distill_batch_stop" not in st.session_state:
    st.session_state.distill_batch_stop = threading.Event()

st.divider()
st.subheader("批量蒸馏（名单文件：一行一个主播）")

qcol1, qcol2, qcol3 = st.columns([3, 1, 1])
with qcol1:
    queue_path = st.text_input("名单文件路径", value=str(default_queue_path), help="一行一个主播名；支持以 # 开头的注释行。")
with qcol2:
    force_rerun = st.toggle("强制重跑", value=False, help="开启后会忽略已存在的输出文件，全部重新蒸馏。")
with qcol3:
    timeout_s = st.number_input("单个超时(s)", min_value=60, max_value=7200, value=1200, step=60)

batch_track = st.text_input("批量分类/赛道（可选）", value=track, help="留空则不传 track。")

btn1, btn2, btn3 = st.columns([1, 1, 2])
with btn1:
    start_batch = st.button("开始批量蒸馏", disabled=bool(st.session_state.distill_batch_job.get("running")))
with btn2:
    stop_batch = st.button("停止批量", disabled=not bool(st.session_state.distill_batch_job.get("running")))
with btn3:
    st.caption("会自动跳过已蒸馏（输出文件已存在）的主播；适合挂机跑。")

job = st.session_state.distill_batch_job
lock = st.session_state.distill_batch_lock
stop_event = st.session_state.distill_batch_stop

if stop_batch:
    stop_event.set()
    _append_batch_log(job, lock, "[UI] 收到停止指令")
    st.rerun()

if start_batch:
    p = Path(queue_path.strip())
    if not p.exists():
        st.error(f"名单文件不存在：{p}")
        st.stop()
    raw = p.read_text(encoding="utf-8", errors="ignore")
    anchors = _clean_queue_lines(raw)
    if not anchors:
        st.warning("名单为空（请一行一个主播名）")
        st.stop()
    stop_event.clear()
    with lock:
        st.session_state.distill_batch_job = {"running": True, "progress": 0.0, "done": 0, "skipped": 0, "failed": 0, "current": "", "logs": [], "started_at": _now(), "updated_at": time.time()}
        job = st.session_state.distill_batch_job
    params = {
        "anchors": anchors,
        "track": batch_track.strip(),
        "model": model.strip(),
        "max_tokens": int(num_predict),
        "force": bool(force_rerun),
        "timeout_s": int(timeout_s),
        "sleep_s": 0.25,
    }
    th = threading.Thread(target=_batch_worker, args=(params, job, lock, stop_event), daemon=True)
    th.start()
    st.rerun()

prog = float(job.get("progress") or 0.0)
st.progress(min(1.0, max(0.0, prog)))
st.write(f"状态：{'运行中' if job.get('running') else '空闲'}｜当前：{job.get('current') or '-'}｜完成：{job.get('done') or 0}｜跳过：{job.get('skipped') or 0}｜失败：{job.get('failed') or 0}")
try:
    upd = float(job.get("updated_at") or 0.0)
    if upd > 0:
        st.caption(f"最后更新：{time.strftime('%H:%M:%S', time.localtime(upd))}")
except Exception:
    pass
logs = job.get("logs") or []
if isinstance(logs, list) and logs:
    st.text_area("批量日志", value="\n".join(str(x) for x in logs[-200:]), height=260)

auto_refresh = st.toggle("自动刷新（运行中每 1 秒）", value=bool(job.get("running")), key="distill_batch_autorefresh")
if bool(job.get("running")) and bool(auto_refresh):
    time.sleep(1.0)
    st.rerun()

if run_btn:
    with st.spinner("蒸馏中（本地 Ollama 推理中）..."):
        ok, path, msg = _run_one(
            anchor_name=anchor.strip(),
            track_text=track.strip(),
            model_name=model.strip(),
            max_tokens=int(num_predict),
            timeout_s=1200,
        )
    if not ok:
        st.error(msg or "蒸馏失败")
        st.stop()
    if msg:
        st.success(msg)
    if path:
        st.write("已写入：")
        st.code(path, language="text")
        try:
            p2 = Path(path)
            if p2.exists():
                st.text_area("写入内容预览", value=p2.read_text(encoding="utf-8", errors="ignore"), height=520)
        except Exception:
            pass
