from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


logger = logging.getLogger("engine_adb")


@dataclass(frozen=True)
class SwipeAction:
    x1: int
    y1: int
    x2: int
    y2: int
    duration_ms: int = 350


def _profile_for_device(device_id: str | None) -> str:
    d = (device_id or "").strip()
    if "c43451370a20" in d:
        return "redmi"
    if "192.168.10.179" in d or "KJHI6LGI4PCMGM6D" in d:
        return "oppo_new"
    if "13dede9d" in d:
        return "oppo_old"
    return "generic"


_GRID_2COL_6 = [
    (270, 560),
    (810, 560),
    (270, 1080),
    (810, 1080),
    (270, 1600),
    (810, 1600),
]


_TEMPLATES: dict[str, dict[str, dict[str, object]]] = {
    "generic": {
        "douyin": {"force_enter": (100, 300), "refresh_swipe": SwipeAction(540, 600, 540, 1800, 420), "grid_items": _GRID_2COL_6},
        "kuaishou": {"force_enter": (315, 633), "refresh_swipe": SwipeAction(540, 600, 540, 1800, 420), "grid_items": _GRID_2COL_6},
        "xiaohongshu": {"force_enter": (125, 315), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
    },
    "redmi": {
        "douyin": {"force_enter": (100, 300), "refresh_swipe": SwipeAction(540, 600, 540, 1800, 420), "grid_items": _GRID_2COL_6},
        "kuaishou": {"force_enter": (315, 633), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
        "xiaohongshu": {"force_enter": (125, 315), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
    },
    "oppo_new": {
        "douyin": {"force_enter": (100, 300), "refresh_swipe": SwipeAction(540, 600, 540, 1800, 420), "grid_items": _GRID_2COL_6},
        "kuaishou": {"force_enter": (315, 633), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
        "xiaohongshu": {"force_enter": (125, 315), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
    },
    "oppo_old": {
        "douyin": {"force_enter": (100, 300), "refresh_swipe": SwipeAction(540, 600, 540, 1800, 420), "grid_items": _GRID_2COL_6},
        "kuaishou": {"force_enter": (315, 633), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
        "xiaohongshu": {"force_enter": (125, 315), "refresh_swipe": SwipeAction(540, 650, 540, 1850, 420), "grid_items": _GRID_2COL_6},
    },
}


def get_ui_template(device_id: str | None) -> dict[str, dict[str, object]]:
    prof = _profile_for_device(device_id)
    return _TEMPLATES.get(prof) or _TEMPLATES["generic"]


@dataclass
class OCRHit:
    keyword: str
    x: int
    y: int
    text: str
    conf: float


class OCRGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._reader = None

    def _get_reader(self):
        with self._lock:
            if self._reader is not None:
                return self._reader
            import easyocr

            self._reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
            return self._reader

    def find_first(self, image_path: str, keywords: list[str]) -> OCRHit | None:
        reader = self._get_reader()
        try:
            results = reader.readtext(image_path)
        except Exception:
            return None

        for bbox, text, conf in results:
            t = (text or "").strip()
            if not t:
                continue
            matched = None
            for k in keywords:
                if k in t:
                    matched = k
                    break
            if not matched:
                continue
            try:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x = int((min(xs) + max(xs)) / 2)
                y = int((min(ys) + max(ys)) / 2)
                return OCRHit(keyword=matched, x=x, y=y, text=t, conf=float(conf))
            except Exception:
                continue
        return None

    def read_all(self, image_path: str):
        reader = self._get_reader()
        try:
            return reader.readtext(image_path)
        except Exception:
            return []


_DEFAULT_GUARD = OCRGuard()


def find_popup_keyword(image_path: str, keywords: list[str]) -> OCRHit | None:
    return _DEFAULT_GUARD.find_first(image_path, keywords)


def read_ocr(image_path: str):
    return _DEFAULT_GUARD.read_all(image_path)


class VisualHealer:
    def __init__(self, api_url: str = "", model: str = "", device_id: str | None = None):
        self.device_id = device_id

    @dataclass(frozen=True)
    class _Box:
        x1: int
        y1: int
        x2: int
        y2: int
        text: str
        conf: float

        @property
        def cx(self) -> int:
            return int((self.x1 + self.x2) / 2)

        @property
        def cy(self) -> int:
            return int((self.y1 + self.y2) / 2)

    def _boxes(self, image_path: str) -> list[_Box]:
        out: list[VisualHealer._Box] = []
        for bbox, text, conf in (read_ocr(image_path) or []):
            try:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                t = (text or "").strip()
                if not t:
                    continue
                out.append(self._Box(x1=x1, y1=y1, x2=x2, y2=y2, text=t, conf=float(conf)))
            except Exception:
                continue
        out.sort(key=lambda b: (b.y1, b.x1))
        return out

    def _find_first(self, boxes: list[_Box], keywords: list[str]) -> _Box | None:
        if not boxes:
            return None
        ks = [k for k in (keywords or []) if (k or "").strip()]
        if not ks:
            return None
        for b in boxes:
            for k in ks:
                if k in b.text:
                    return b
        return None

    def _template(self, platform: str) -> dict[str, object]:
        t = get_ui_template(self.device_id) or {}
        return (t.get(platform) or {}) if isinstance(t, dict) else {}

    def get_coordinates(self, image_path: str, target_desc: str) -> list:
        boxes = self._boxes(image_path)
        desc = (target_desc or "").strip()
        direct_keys = []
        for k in ["置顶", "Pinned", "允许", "确定", "取消", "继续安装", "始终允许", "我知道了", "仅充电"]:
            if k in desc:
                direct_keys.append(k)
        if direct_keys:
            hit = self._find_first(boxes, direct_keys)
            if hit is not None:
                return [hit.cx, hit.cy]

        for plat in ["douyin", "kuaishou", "xiaohongshu"]:
            if plat in desc.lower():
                t = self._template(plat)
                p = t.get("force_enter")
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    return [int(p[0]), int(p[1])]

        if "搜索" in desc and ("用户" in desc or "结果" in desc):
            t = self._template("douyin")
            p = t.get("force_enter")
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return [int(p[0]), int(p[1])]

        return []

    def analyze_video_grid(self, image_path: str, platform: str) -> dict:
        plat = (platform or "").strip() or "douyin"
        t = self._template(plat)
        pts = t.get("grid_items") or []
        if not isinstance(pts, list):
            pts = []
        boxes = self._boxes(image_path)
        pinned_marks = [b for b in boxes if ("置顶" in b.text or "Pinned" in b.text)]

        items = []
        for i, p in enumerate(list(pts)[:6]):
            try:
                x, y = int(p[0]), int(p[1])
            except Exception:
                continue
            pinned = False
            pinned_text = ""
            for b in pinned_marks:
                dx = b.cx - x
                dy = b.cy - y
                if dx * dx + dy * dy <= 180 * 180:
                    pinned = True
                    pinned_text = b.text
                    break
            items.append({"index": i + 1, "center": {"x": x, "y": y}, "pinned": pinned, "pinned_text": pinned_text, "hint": ""})
        return {"items": items}

    def read_publish_time(self, image_path: str, platform: str) -> dict:
        boxes = self._boxes(image_path)
        patterns = [
            r"\d+\s*小时前",
            r"\d+\s*分钟前",
            r"\d+\s*天前",
            r"\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}(?:\s*\d{1,2}[:：]\d{2})?",
            r"\d{1,2}[-/\.]\d{1,2}(?:\s*\d{1,2}[:：]\d{2})?",
            r"(凌晨|上午|中午|下午|晚上)\s*\d{1,2}[:：]\d{2}",
            r"\d{1,2}:\d{2}",
        ]
        for b in boxes:
            t = b.text
            if "昨天" in t or "前天" in t:
                return {"publish_text": t, "keywords_seen": ["昨天" if "昨天" in t else "", "前天" if "前天" in t else ""], "is_detail": True}
            for p in patterns:
                if re.search(p, t):
                    return {"publish_text": t, "keywords_seen": [p], "is_detail": True}
        return {"publish_text": "", "keywords_seen": [], "is_detail": False}

    def detect_system_dialog(self, image_path: str) -> dict:
        boxes = self._boxes(image_path)
        prio = ["仅充电", "取消", "USB", "传输文件"]
        hit = self._find_first(boxes, prio)
        if hit is None:
            return {"has_usb_dialog": False, "action": "none", "target": "", "x": 0, "y": 0, "evidence": ""}
        return {"has_usb_dialog": True, "action": "tap", "target": hit.text, "x": hit.cx, "y": hit.cy, "evidence": hit.text}

    def detect_confirm_dialog(self, image_path: str) -> dict:
        boxes = self._boxes(image_path)
        prio = ["仅充电", "始终允许", "一律允许", "允许", "确定", "我知道了", "继续安装", "USB调试确认"]
        hit = self._find_first(boxes, prio)
        if hit is None:
            return {"has_popup": False, "keyword": "", "x": 0, "y": 0, "evidence": ""}
        kw = ""
        for k in prio:
            if k in hit.text:
                kw = k
                break
        return {"has_popup": True, "keyword": kw, "x": hit.cx, "y": hit.cy, "evidence": hit.text}

    def detect_usb_debug_rsa_dialog(self, image_path: str) -> dict:
        boxes = self._boxes(image_path)
        has = False
        for b in boxes:
            if "RSA" in b.text or "USB 调试" in b.text or "USB调试" in b.text or "这台计算机" in b.text:
                has = True
                break
        if not has:
            return {"has_rsa_dialog": False, "checkbox": {"x": 0, "y": 0}, "confirm": {"x": 0, "y": 0}, "evidence": ""}

        confirm = self._find_first(boxes, ["一律允许", "始终允许", "允许", "确定"])
        cbx, cby = 0, 0
        if confirm is not None:
            cbx = max(1, confirm.cx - 260)
            cby = max(1, confirm.cy - 140)
        return {
            "has_rsa_dialog": True,
            "checkbox": {"x": int(cbx), "y": int(cby)},
            "confirm": {"x": int(confirm.cx) if confirm is not None else 0, "y": int(confirm.cy) if confirm is not None else 0},
            "evidence": "RSA",
        }


class ADBBase:
    def __init__(self, config: dict):
        self.adb_path = config.get("adb_path", "adb")
        self.device_id = config.get("device_ip", "")
        invalid_chars = '<>:"/\\\\|?*'
        device_folder = "".join("_" if ch in invalid_chars else ch for ch in (self.device_id or "default"))
        self.temp_root = Path(r"F:\Ollama_Safe_Backup") / "runtime" / device_folder
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.screen_width = 1080
        self.screen_height = 2400
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
            sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    def run_adb(self, cmd: str, timeout: int = 60) -> str:
        if self.device_id:
            full_cmd = f"\"{self.adb_path}\" -s {self.device_id} {cmd}"
        else:
            full_cmd = f"\"{self.adb_path}\" {cmd}"
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=timeout,
            )
            try:
                sys.stdout.flush()
            except Exception:
                pass
            return (result.stdout or "").strip()
        except Exception as e:
            logger.error(f"[ADB] 执行失败: {e}")
            return ""

    def get_ui_dump(self, max_retries: int = 3) -> str:
        temp_file = self.temp_root / "ui_dump.xml"
        for attempt in range(max_retries):
            try:
                self.run_adb("shell uiautomator dump /sdcard/ui_dump.xml")
                time.sleep(0.5)
                self.run_adb(f"pull /sdcard/ui_dump.xml \"{temp_file}\"")
                if temp_file.exists():
                    return temp_file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"[UI] 第{attempt+1}次获取失败: {e}")
                time.sleep(1)
        return ""

    def find_element_by_text(self, text: str, xml_data: str = None) -> Optional[Tuple[int, int]]:
        if xml_data is None:
            xml_data = self.get_ui_dump()
        try:
            root = ET.fromstring(xml_data)
            for node in root.iter("node"):
                node_text = node.attrib.get("text", "")
                content_desc = node.attrib.get("content-desc", "")
                if text in node_text or text in content_desc:
                    bounds = node.attrib.get("bounds", "")
                    coords = re.findall(r"\d+", bounds)
                    if len(coords) == 4:
                        x1, y1, x2, y2 = map(int, coords)
                        return (x1 + x2) // 2, (y1 + y2) // 2
        except ET.ParseError:
            pass
        return None

    def find_element_by_class(self, class_name: str, xml_data: str = None) -> Optional[Tuple[int, int]]:
        if xml_data is None:
            xml_data = self.get_ui_dump()
        try:
            root = ET.fromstring(xml_data)
            for node in root.iter("node"):
                if node.attrib.get("class") == class_name:
                    bounds = node.attrib.get("bounds", "")
                    coords = re.findall(r"\d+", bounds)
                    if len(coords) == 4:
                        x1, y1, x2, y2 = map(int, coords)
                        return (x1 + x2) // 2, (y1 + y2) // 2
        except ET.ParseError:
            pass
        return None

    def tap(self, x: int, y: int):
        self.run_adb(f"shell input tap {int(x)} {int(y)}")

    def press_key(self, keycode: int):
        self.run_adb(f"shell input keyevent {int(keycode)}")

    def press_back(self, times: int = 1):
        for _ in range(int(times)):
            self.press_key(4)
            time.sleep(0.3)

    def screenshot(self, save_path: str) -> bool:
        try:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            base = [self.adb_path]
            if self.device_id:
                base += ["-s", self.device_id]
            for attempt in range(2):
                p = subprocess.run([*base, "exec-out", "screencap", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if p.returncode == 0 and p.stdout and len(p.stdout) > 10_000:
                    out.write_bytes(p.stdout)
                    return out.exists() and out.stat().st_size > 10_000
                if attempt == 0:
                    time.sleep(1.5)
        except Exception:
            return False
        return False


class ADBMonitorBase(ADBBase):
    def __init__(self, config: dict):
        super().__init__(config)
        self.healer = VisualHealer(device_id=getattr(self, "device_id", None))
        self.screenshot_dir = self.temp_root / "temp_screenshots_auto"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _ui_template(self, platform: str) -> dict:
        t = get_ui_template(getattr(self, "device_id", None)) or {}
        if not isinstance(t, dict):
            return {}
        p = (platform or "").strip() or "douyin"
        return (t.get(p) or {}) if isinstance(t.get(p), dict) else {}

    def _fallback_force_enter(self) -> tuple[int, int] | None:
        for p in ["douyin", "kuaishou", "xiaohongshu"]:
            t = self._ui_template(p)
            xy = t.get("force_enter")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                try:
                    return int(xy[0]), int(xy[1])
                except Exception:
                    continue
        return None

    def _fallback_refresh_swipe(self, platform: str) -> SwipeAction | None:
        t = self._ui_template(platform)
        s = t.get("refresh_swipe")
        if isinstance(s, SwipeAction):
            return s
        return None

    def smart_tap(self, target_text: str = None, target_class: str = None, target_desc: str = "未知目标") -> bool:
        self.visual_clean_popup()
        xml_data = None
        try:
            xml_data = self.get_ui_dump(max_retries=1)
        except Exception:
            xml_data = None

        coords = None
        if xml_data:
            if target_text:
                coords = self.find_element_by_text(target_text, xml_data=xml_data)
            elif target_class:
                coords = self.find_element_by_class(target_class, xml_data=xml_data)

        if coords:
            logger.info(f"🎯 [XML 定位成功] {target_desc} -> {coords}")
            self.tap(coords[0], coords[1])
            return True

        logger.warning(f"⚠️ [XML 定位失败] {target_desc}，正在启动视觉自愈...")

        screenshot_path = str(self.screenshot_dir / "temp.png")
        if self.screenshot(screenshot_path):
            coords = self.healer.get_coordinates(screenshot_path, target_desc)
            if len(coords) == 2:
                logger.info(f"✨ [视觉纠偏成功] {target_desc} -> ({coords[0]}, {coords[1]})")
                self.tap(coords[0], coords[1])
                return True
            fb = self._fallback_force_enter()
            if fb is not None:
                fx, fy = fb
                logger.info(f"✨ [坐标兜底] {target_desc} -> ({fx}, {fy})")
                self.tap(int(fx), int(fy))
                return True
        return False

    def tap(self, x: int, y: int):
        self.visual_clean_popup()
        super().tap(x, y)

    def preflight_clear_popups(self) -> bool:
        return self.visual_clean_popup()

    def visual_clean_popup(self) -> bool:
        keywords = ["仅充电", "允许", "确定", "始终允许", "我知道了", "USB调试确认", "继续安装", "取消", "USB", "USB 用于", "传输文件"]
        safe_device = (self.device_id or "default").replace(":", "_")
        account_tag = getattr(self, "_current_account", "") or ""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        def record(keyword: str, x: int, y: int, screenshot_path: str):
            try:
                log_dir = Path(r"F:\Ollama_Safe_Backup") / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                report_path = log_dir / "last_report.md"
                report_path.write_text(
                    (report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else "")
                    + f"\n- SYSTEM_STABILITY_FIX | {ts} | device={safe_device} | account={account_tag} | keyword={keyword} | ({x},{y}) | {screenshot_path}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

        try:
            xml_data = ""
            try:
                xml_data = self.get_ui_dump(max_retries=1) or ""
            except Exception:
                xml_data = ""

            if xml_data:
                for kw in keywords:
                    coords = self.find_element_by_text(kw, xml_data=xml_data)
                    if coords:
                        super().tap(coords[0], coords[1])
                        logger.info(f"✨ [视觉清场] 成功点掉系统干扰弹窗: {kw} @ ({coords[0]},{coords[1]})")
                        record(kw, int(coords[0]), int(coords[1]), "XML")
                        time.sleep(0.5)
                        try:
                            xml2 = self.get_ui_dump(max_retries=1) or ""
                        except Exception:
                            xml2 = ""
                        still = False
                        if xml2:
                            for kw2 in keywords:
                                if self.find_element_by_text(kw2, xml_data=xml2):
                                    still = True
                                    break
                        return not still

            screenshot_path = str(self.screenshot_dir / "clean_popup.png")
            if not self.screenshot(screenshot_path):
                return False

            hit = None
            try:
                hit = find_popup_keyword(screenshot_path, ["确定", "取消", "仅充电", "始终允许", "允许", "我知道了", "继续安装", "USB调试确认"])
            except Exception:
                hit = None
            if hit is not None:
                super().tap(hit.x, hit.y)
                logger.info(f"[OCR] 自动点掉弹窗：{hit.keyword}")
                record(hit.keyword, int(hit.x), int(hit.y), screenshot_path)
                time.sleep(0.5)
                hit2 = None
                try:
                    hit2 = find_popup_keyword(screenshot_path, ["确定", "取消", "仅充电", "始终允许", "允许", "我知道了", "继续安装", "USB调试确认"])
                except Exception:
                    hit2 = None
                return hit2 is None

            usb = self.healer.detect_system_dialog(screenshot_path) or {}
            if usb.get("has_usb_dialog") and int(usb.get("x") or 0) > 0 and int(usb.get("y") or 0) > 0:
                x0, y0 = int(usb.get("x") or 0), int(usb.get("y") or 0)
                kw0 = (usb.get("target") or "USB_DIALOG").strip()
                super().tap(x0, y0)
                logger.info(f"✨ [视觉清场] 成功点掉系统干扰弹窗: {kw0} @ ({x0},{y0})")
                record(kw0, x0, y0, screenshot_path)
                time.sleep(0.5)
                return True

            d = self.healer.detect_confirm_dialog(screenshot_path) or {}
            if not (d.get("has_popup") or False):
                return False
            x, y = d.get("x"), d.get("y")
            kw = (d.get("keyword") or "").strip() or "POPUP"
            if not (isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0):
                return False
            super().tap(x, y)
            logger.info(f"✨ [视觉清场] 成功点掉系统干扰弹窗: {kw} @ ({x},{y})")
            record(kw, int(x), int(y), screenshot_path)
            time.sleep(0.5)
            if not self.screenshot(screenshot_path):
                return True
            d2 = self.healer.detect_confirm_dialog(screenshot_path) or {}
            return not (d2.get("has_popup") or False)
        except Exception:
            return False

    def _parse_publish_text_to_iso(self, publish_text: str, now: datetime) -> str:
        s = (publish_text or "").strip()
        if not s:
            return ""
        if re.fullmatch(r"\d{1,2}:\d{2}", s):
            hh, mm = [int(x) for x in s.split(":")]
            dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(凌晨|上午|中午|下午|晚上)\s*(\d{1,2})[:：](\d{2})", s)
        if m:
            tag, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
            if tag in ["下午", "晚上"] and hh < 12:
                hh += 12
            if tag == "中午" and hh == 0:
                hh = 12
            dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})(?:\s+(\d{1,2})[:：](\d{2}))?", s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hh = int(m.group(4) or 0)
            mm = int(m.group(5) or 0)
            return datetime(y, mo, d, hh, mm).strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(\d{1,2})[-/\.](\d{1,2})(?:\s+(\d{1,2})[:：](\d{2}))?", s)
        if m:
            mo, d = int(m.group(1)), int(m.group(2))
            hh = int(m.group(3) or 0)
            mm = int(m.group(4) or 0)
            return datetime(now.year, mo, d, hh, mm).strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(\d+)\s*小时前", s)
        if m:
            dt = now - timedelta(hours=int(m.group(1)))
            return dt.strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(\d+)\s*分钟前", s)
        if m:
            dt = now - timedelta(minutes=int(m.group(1)))
            return dt.strftime("%Y-%m-%d %H:%M")
        m = re.search(r"(\d+)\s*天前", s)
        if m:
            dt = (now - timedelta(days=int(m.group(1)))).replace(second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M")
        if "昨天" in s:
            dt = (now - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M")
        if "前天" in s:
            dt = (now - timedelta(days=2)).replace(hour=12, minute=0, second=0, microsecond=0)
            return dt.strftime("%Y-%m-%d %H:%M")
        return ""

    def audit_latest_video_time(self, platform: str, account: dict) -> dict:
        now = datetime.now()
        uid = str(account.get("uid") or "").strip()
        nickname = str(account.get("nickname") or "").strip()
        safe_device = (self.device_id or "default").replace(":", "_")
        out_dir = Path(r"F:\Ollama_Safe_Backup") / "logs" / "audit_latest_video"
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_path = out_dir / f"{platform}_{safe_device}_{uid}_grid.png"
        detail_path = out_dir / f"{platform}_{safe_device}_{uid}_detail.png"

        try:
            setattr(self, "_current_account", f"{platform}|{nickname or uid}")
        except Exception:
            pass

        self.visual_clean_popup()

        if not self.screenshot(str(grid_path)):
            return {"ok": False, "reason": "screencap_failed", "grid": str(grid_path)}

        grid = self.healer.analyze_video_grid(str(grid_path), platform) or {}
        items = grid.get("items") or []
        for _ in range(3):
            if items:
                break
            time.sleep(1.5)
            if self.screenshot(str(grid_path)):
                grid = self.healer.analyze_video_grid(str(grid_path), platform) or {}
                items = grid.get("items") or []

        chosen = None
        for it in items:
            if bool(it.get("pinned")):
                continue
            c = it.get("center") or {}
            x, y = c.get("x"), c.get("y")
            if isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0:
                chosen = (x, y, it)
                break
        if not chosen:
            sw = self._fallback_refresh_swipe(platform)
            if sw is not None:
                try:
                    self.run_adb(f"shell input swipe {sw.x1} {sw.y1} {sw.x2} {sw.y2} {sw.duration_ms}")
                    time.sleep(1.2)
                    if self.screenshot(str(grid_path)):
                        grid = self.healer.analyze_video_grid(str(grid_path), platform) or {}
                        items = grid.get("items") or []
                        for it in items:
                            if bool(it.get("pinned")):
                                continue
                            c = it.get("center") or {}
                            x, y = c.get("x"), c.get("y")
                            if isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0:
                                chosen = (x, y, it)
                                break
                except Exception:
                    pass
        if not chosen:
            return {"ok": False, "reason": "no_item", "grid": str(grid_path)}

        x, y, _ = chosen
        self.tap(int(x), int(y))
        time.sleep(2.0)
        if not self.screenshot(str(detail_path)):
            self.press_back(1)
            return {"ok": False, "reason": "detail_screencap_failed", "detail": str(detail_path)}

        pub = self.healer.read_publish_time(str(detail_path), platform) or {}
        publish_text = (pub.get("publish_text") or "").strip()
        if platform == "xiaohongshu" and not publish_text:
            self.run_adb("shell input swipe 540 1800 540 900 350")
            time.sleep(1.0)
            if self.screenshot(str(detail_path)):
                pub = self.healer.read_publish_time(str(detail_path), platform) or {}
                publish_text = (pub.get("publish_text") or "").strip()
        if platform == "xiaohongshu" and not publish_text:
            self.tap(400, 2100)
            time.sleep(0.8)
            if self.screenshot(str(detail_path)):
                pub = self.healer.read_publish_time(str(detail_path), platform) or {}
                publish_text = (pub.get("publish_text") or "").strip()

        iso = self._parse_publish_text_to_iso(publish_text, now)
        self.press_back(1)
        time.sleep(0.6)

        prev = str(account.get("last_video_time") or "").strip()
        is_new = bool(iso and (not prev or iso > prev))
        if is_new:
            account["last_video_time"] = iso
            account["sticky_n"] = int(account.get("sticky_n") or 0) + 1

        try:
            idx_path = Path(r"F:\Ollama_Safe_Backup") / "logs" / "latest_video_audit.json"
            prev_json = json.loads(idx_path.read_text(encoding="utf-8", errors="ignore") or "{}") if idx_path.exists() else {}
            prev_json[f"{platform}|{uid}"] = {
                "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                "platform": platform,
                "uid": uid,
                "nickname": nickname,
                "publish_text": publish_text,
                "iso": iso,
                "is_new": is_new,
                "grid": str(grid_path),
                "detail": str(detail_path),
            }
            idx_path.write_text(json.dumps(prev_json, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return {"ok": True, "iso": iso, "publish_text": publish_text, "is_new": is_new, "detail": str(detail_path)}

