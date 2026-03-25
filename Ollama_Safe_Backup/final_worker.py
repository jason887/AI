import io
import re
import subprocess
import time
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# --- 环境修复：禁用报错引擎与加速插件 ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_use_onednn'] = '0' 

# --- 配置区 ---
ADB_PATH = r"D:\scrcpy\scrcpy-win64-v2.4\adb.exe"
DEVICE_ID = "c43451370a20"
TARGET_PKG = "com.ss.android.ugc.aweme"
TARGET_URL = "https://v.douyin.com/vqZSY3YvJHI/"
TEMP_XML = Path("ui_dump.xml")

def adb(args: list[str], timeout: int = 25):
    # 核心修复：所有 adb 命令强制带上 -s DEVICE_ID，防止多设备干扰
    return subprocess.run([ADB_PATH, "-s", DEVICE_ID] + args, capture_output=True, timeout=timeout)

def tap(x: int, y: int):
    adb(["shell", "input", "tap", str(int(x)), str(int(y))])

def click_by_xml(text_content: str) -> bool:
    """基于 XML 节点的动态坐标定位"""
    adb(["shell", "uiautomator", "dump", "/data/local/tmp/ui.xml"])
    adb(["pull", "/data/local/tmp/ui.xml", str(TEMP_XML)])
    if not TEMP_XML.exists(): return False
    try:
        tree = ET.parse(str(TEMP_XML))
        for node in tree.iter("node"):
            t, d = node.get("text", ""), node.get("content-desc", "")
            if text_content in t or text_content in d:
                bounds = node.get("bounds", "")
                m = re.findall(r"\d+", bounds)
                if len(m) == 4:
                    x = (int(m[0]) + int(m[2])) // 2
                    y = (int(m[1]) + int(m[3])) // 2
                    tap(x, y)
                    return True
    except: pass
    return False

def harvest_comments():
    """OCR 评论采集（已修复初始化参数）"""
    print("【4】正在初始化 OCR 引擎...")
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
    # 修正：移除废弃的 use_angle_cls，确保不报错
    ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
    
    results = {}
    print("【5】开始循环滑动采集评论内容...")
    for i in range(20):
        p = adb(["exec-out", "screencap", "-p"]).stdout
        if p:
            try:
                img = Image.open(io.BytesIO(p)).convert("RGB")
                res = ocr.ocr(np.array(img))
                if res and res[0]:
                    for line in res[0]:
                        txt = line[1][0]
                        if len(txt) > 3:
                            results[txt] = results.get(txt, 0) + 1
            except: pass
        
        # 物理滑动翻页
        adb(["shell", "input", "swipe", "540", "1600", "540", "800", "400"])
        time.sleep(2)
        print(f"进度: 第 {i+1} 次滑动, 已捕获独特内容: {len(results)} 条")

    with open("final_comments.txt", "w", encoding="utf-8") as f:
        for k in results.keys(): f.write(f"{k}\n")
    print("【成功】所有高赞评论已保存至 final_comments.txt")

def main():
    print("【1】暴力重启抖音...")
    adb(["shell", "am", "force-stop", TARGET_PKG])
    time.sleep(1)
    adb(["shell", "monkey", "-p", TARGET_PKG, "-c", "android.intent.category.LAUNCHER", "1"])
    time.sleep(8)
    
    print("【2】物理输入 URL 跳转...")
    # 沿用你成功的 XML 查找 + 坐标保底逻辑
    if not click_by_xml("搜索"): tap(980, 120)
    time.sleep(3)
    tap(500, 120) 
    
    # 沿用你成功的清空逻辑
    for _ in range(25): adb(["shell", "input", "keyevent", "67"])
    
    # 物理输入 URL
    adb(["shell", "input", "text", TARGET_URL])
    time.sleep(2)
    
    if not click_by_xml("搜索"): adb(["shell", "input", "keyevent", "66"])
    time.sleep(8) # 增加等待视频加载的时间
    
    print("【3】展开评论区并切换排序...")
    if not click_by_xml("评论"): tap(960, 1150)
    time.sleep(4)
    
    # 尝试切换到热门/点赞排序
    if not (click_by_xml("点赞") or click_by_xml("热门")):
        pass 
    time.sleep(2)
    
    # 执行采集
    harvest_comments()

if __name__ == "__main__":
    main()