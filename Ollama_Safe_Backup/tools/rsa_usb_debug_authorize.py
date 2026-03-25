import argparse
import json
import subprocess
import time
from pathlib import Path

from engine_adb import VisualHealer, find_popup_keyword


ADB = r"D:\scrcpy\scrcpy-win64-v2.4\adb.exe"
LOG_DIR = Path(r"F:\Ollama_Safe_Backup\logs")


def run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout)


def list_devices() -> dict[str, str]:
    p = run([ADB, "devices", "-l"], timeout=10)
    res = {}
    for line in (p.stdout or "").splitlines()[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            res[parts[0]] = parts[1]
    return res


def screencap_png(serial: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([ADB, "-s", serial, "exec-out", "screencap", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        return False
    out_path.write_bytes(p.stdout)
    return out_path.exists() and out_path.stat().st_size > 10_000


def tap(serial: str, x: int, y: int):
    run([ADB, "-s", serial, "shell", "input", "tap", str(int(x)), str(int(y))], timeout=8)


def set_adb_enabled(serial: str) -> bool:
    p = run([ADB, "-s", serial, "shell", "settings", "put", "global", "adb_enabled", "1"], timeout=10)
    return p.returncode == 0


def log_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    row = {"ts": ts, **obj}
    path.write_text(
        (path.read_text(encoding="utf-8", errors="ignore") if path.exists() else "")
        + json.dumps(row, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def wait_for_device(serial: str, timeout_seconds: int, poll_seconds: int) -> str:
    start = time.time()
    last = ""
    while time.time() - start < timeout_seconds:
        st = list_devices().get(serial, "")
        if st:
            last = st
        if st == "device":
            return "device"
        time.sleep(max(1, int(poll_seconds)))
    return last or ""


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="13dede9d")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--unauth-threshold", type=int, default=30)
    ap.add_argument("--poll", type=int, default=5)
    ap.add_argument("--set-adb-enabled", action="store_true")
    args = ap.parse_args()

    serial = str(args.serial).strip()
    devices = list_devices()
    st = devices.get(serial, "")
    if not st:
        print(f"{serial} not_found", flush=True)
        return 2

    watch_log = LOG_DIR / "rsa_authorize_watch.log"
    if st == "unauthorized":
        t0 = time.time()
        if args.watch:
            while True:
                st2 = list_devices().get(serial, "")
                if st2 == "device":
                    break
                if st2 == "unauthorized" and (time.time() - t0) >= int(args.unauth_threshold):
                    msg = f"{serial} unauthorized > {int(args.unauth_threshold)}s. 请在手机上勾选“一律允许使用这台计算机进行调试”并点击“确定”。"
                    print(msg, flush=True)
                    log_line(watch_log, {"serial": serial, "state": "unauthorized", "note": "manual_confirm_required"})
                    t0 = time.time()
                time.sleep(max(1, int(args.poll)))
        else:
            print(f"{serial} unauthorized (manual confirm required)", flush=True)
            log_line(watch_log, {"serial": serial, "state": "unauthorized", "note": "manual_confirm_required"})
            return 3
        st = "device"

    if st != "device":
        run([ADB, "reconnect", "offline"], timeout=20)
        time.sleep(1.0)
        st = wait_for_device(serial, timeout_seconds=20, poll_seconds=2)
        if st != "device":
            print(f"{serial} state={st or 'unknown'}", flush=True)
            return 4

    before = LOG_DIR / "rsa_usb_debug_before.png"
    if not screencap_png(serial, before):
        print(f"{serial} screencap_failed", flush=True)
        return 4

    hit = find_popup_keyword(str(before), ["确定", "允许", "始终允许", "一律允许", "USB", "调试", "RSA"])
    healer = VisualHealer(device_id=serial)
    rsa = healer.detect_usb_debug_rsa_dialog(str(before)) or {}
    print(json.dumps({"serial": serial, "state": st, "ocr": (hit.keyword if hit else ""), "rsa": rsa}, ensure_ascii=False), flush=True)

    if not rsa.get("has_rsa_dialog"):
        if args.set_adb_enabled:
            ok = set_adb_enabled(serial)
            log_line(watch_log, {"serial": serial, "state": "device", "set_adb_enabled": ok})
        return 0

    cb = rsa.get("checkbox") or {}
    cf = rsa.get("confirm") or {}
    if int(cb.get("x") or 0) > 0 and int(cb.get("y") or 0) > 0:
        tap(serial, int(cb["x"]), int(cb["y"]))
        time.sleep(0.2)
    if int(cf.get("x") or 0) > 0 and int(cf.get("y") or 0) > 0:
        tap(serial, int(cf["x"]), int(cf["y"]))
        time.sleep(0.8)

    after = LOG_DIR / "rsa_usb_debug_after.png"
    if screencap_png(serial, after):
        rsa2 = healer.detect_usb_debug_rsa_dialog(str(after)) or {}
        print(json.dumps({"after": rsa2}, ensure_ascii=False), flush=True)
    if args.set_adb_enabled:
        ok = set_adb_enabled(serial)
        log_line(watch_log, {"serial": serial, "state": "device", "set_adb_enabled": ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

