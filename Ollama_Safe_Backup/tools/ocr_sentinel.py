import argparse
import json
import time
import subprocess
from pathlib import Path


ADB = r"D:\scrcpy\scrcpy-win64-v2.4\adb.exe"
LOG_DIR = Path(r"F:\Ollama_Safe_Backup\logs")
SHOT_DIR = LOG_DIR / "ocr_sentinel"
STATS_PATH = LOG_DIR / "ocr_sentinel_stats.json"

KEYWORDS = ["仅限充电", "仅充电", "传输文件", "取消", "始终允许", "允许", "USB", "USB 用于"]


def run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout)


def adb(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return run([ADB, *args], timeout=timeout)


def adb_s(serial: str, args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return run([ADB, "-s", serial, *args], timeout=timeout)


def list_devices() -> dict[str, str]:
    p = adb(["devices"], timeout=10)
    out = (p.stdout or "").splitlines()
    res = {}
    for line in out[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            res[parts[0].strip()] = parts[1].strip()
    return res


def is_online(state: str) -> bool:
    return (state or "").strip() == "device"


def hard_reconnect_if_offline(serial: str, state: str):
    if (state or "").strip() != "offline":
        return
    adb(["reconnect", "offline"], timeout=20)
    time.sleep(1.0)
    adb(["start-server"], timeout=20)


def screencap(serial: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([ADB, "-s", serial, "exec-out", "screencap", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout or len(p.stdout) < 10_000:
        return False
    out_path.write_bytes(p.stdout)
    return out_path.exists() and out_path.stat().st_size > 10_000


def tap(serial: str, x: int, y: int):
    adb_s(serial, ["shell", "input", "tap", str(int(x)), str(int(y))], timeout=8)


def keyevent(serial: str, code: int):
    adb_s(serial, ["shell", "input", "keyevent", str(int(code))], timeout=8)


def load_stats() -> dict:
    try:
        if STATS_PATH.exists():
            return json.loads(STATS_PATH.read_text(encoding="utf-8", errors="ignore") or "{}")
    except Exception:
        pass
    return {"total_taps": 0, "by_device": {}, "last_seen": {}}


def save_stats(stats: dict):
    try:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def try_init_ocr():
    try:
        import ddddocr
        return ("ddddocr", ddddocr.DdddOcr(show_ad=False))
    except Exception:
        pass
    try:
        import easyocr
        return ("easyocr", easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False))
    except Exception:
        return ("none", None)


def scan_keywords_dddd(ocr, image_path: str) -> list[tuple[str, float]]:
    try:
        b = Path(image_path).read_bytes()
        txt = ocr.classification(b) or ""
        hits = []
        for k in KEYWORDS:
            if k in txt:
                hits.append((k, 0.5))
        return hits
    except Exception:
        return []


def scan_keywords_easy(reader, image_path: str) -> list[tuple[int, int, str, float]]:
    try:
        results = reader.readtext(image_path)
    except Exception:
        results = []
    hits = []
    for bbox, text, conf in results:
        t = (text or "").strip()
        if not t:
            continue
        if any(k in t for k in KEYWORDS):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x = int((min(xs) + max(xs)) / 2)
            y = int((min(ys) + max(ys)) / 2)
            hits.append((x, y, t, float(conf)))
    return hits


def pick_mask_fallback() -> tuple[int, int]:
    return 540, 260


def handle_device(serial: str, state: str, engine: str, ocr, stats: dict, wake: bool) -> None:
    hard_reconnect_if_offline(serial, state)
    if not is_online(state):
        return
    if wake:
        keyevent(serial, 224)
        time.sleep(0.2)
        keyevent(serial, 82)
        time.sleep(0.2)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    shot = SHOT_DIR / f"{serial.replace(':','_')}.png"
    if not screencap(serial, shot):
        return

    did_tap = False
    if engine == "easyocr" and ocr is not None:
        hits = scan_keywords_easy(ocr, str(shot))
        if hits:
            x, y, text, conf = hits[0]
            tap(serial, x, y)
            did_tap = True
            stats["total_taps"] = int(stats.get("total_taps") or 0) + 1
            stats.setdefault("by_device", {}).setdefault(serial, 0)
            stats["by_device"][serial] = int(stats["by_device"][serial]) + 1
            stats.setdefault("last_seen", {})[serial] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text, "conf": conf, "shot": str(shot)}
    elif engine == "ddddocr" and ocr is not None:
        hits = scan_keywords_dddd(ocr, str(shot))
        if hits:
            x, y = pick_mask_fallback()
            tap(serial, x, y)
            did_tap = True
            stats["total_taps"] = int(stats.get("total_taps") or 0) + 1
            stats.setdefault("by_device", {}).setdefault(serial, 0)
            stats["by_device"][serial] = int(stats["by_device"][serial]) + 1
            stats.setdefault("last_seen", {})[serial] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "text": hits[0][0], "conf": hits[0][1], "shot": str(shot)}

    if did_tap:
        save_stats(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--wake-new", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats = load_stats()
    engine, ocr = try_init_ocr()
    if engine == "none":
        return 2

    known = set(["192.168.10.179:34543", "c43451370a20", "KJHI6LGI4PCMGM6D", "13dede9d"])

    def tick():
        nonlocal stats
        devs = list_devices()
        for serial, state in devs.items():
            wake = args.wake_new and (serial not in known)
            handle_device(serial, state, engine, ocr, stats, wake=wake)
            known.add(serial)

    if args.once:
        tick()
        return 0

    while True:
        tick()
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())

