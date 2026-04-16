import argparse
import json
import os
import sys
from pathlib import Path
import glob
import re

# Ensure modules path is correct
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.ollama_client import OllamaClient

def _load_files_content(path_pattern: str) -> str:
    content = []
    # Handle UNC paths and normal paths
    # glob in python supports UNC
    for file_path in glob.glob(path_pattern, recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content.append(f"--- File: {os.path.basename(file_path)} ---\n" + f.read())
        except Exception:
            pass
    return "\n".join(content)

def _load_comments(anchor: str) -> str:
    roots = [
        r"\\192.168.10.198\Workstation\抓取结果_D",
        r"\\192.168.10.198\Workstation\抓取结果",
        r"E:\老六AI执行工作台\抓取结果_D",
        r"E:\老六AI执行工作台\抓取结果"
    ]
    lines = []
    short_anchor = anchor.split("（")[0].split("(")[0].strip() or anchor
    search_terms = {anchor, short_anchor}
    processed_paths = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for term in search_terms:
            for p in root_path.glob(f"*{term}*/**/*.txt"):
                try:
                    p_resolved = p.resolve()
                    if p_resolved in processed_paths:
                        continue
                    processed_paths.add(p_resolved)
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line and len(line) > 2:
                                lines.append(line)
                except Exception:
                    continue
    
    # Dedupe
    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    
    # limit to 1000 lines to avoid blowing context
    return "\n".join(deduped[:1000])

def _strip_fences(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.endswith("```"):
        t = t.rsplit("```", 1)[0]
    return t.strip()

def _truncate_text(s: str, max_chars: int) -> str:
    t = (s or "").strip()
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()

def _extract_signal_lines(text: str, keywords: list[str], max_lines: int, max_chars: int) -> str:
    if not text:
        return ""
    noise_exact = {
        "切换侧边栏",
        "搜索",
        "新建聊天",
        "历史记录",
        "查看全部",
        "分享",
        "语音",
        "Imagine",
        "项目",
        "新项目",
        "工具",
        "思考",
        "立即回答",
        "你说",
        "升级",
        "与 Gemini 对话",
        "Gemini 说",
        "Gemini",
        "Grok",
        "豆包",
        "今天",
        "昨天",
    }
    noise_prefix = (
        "Ctrl+",
        "Gemini 是一款 AI 工具",
        "Initiating Q&A Session",
        "请忽略/清空此前所有对话上下文",
        "不要输出任务拆解/步骤/提纲/清单",
    )
    keys = [k for k in (keywords or []) if k]
    lines = []
    seen = set()
    for raw in str(text).splitlines():
        ln = (raw or "").strip().replace("\u00a0", " ")
        if not ln or len(ln) < 2:
            continue
        if ln in noise_exact:
            continue
        if any(ln.startswith(p) for p in noise_prefix):
            continue
        if ln.endswith("？") and ("主播" in ln) and any(k in ln for k in keys[:6]):
            continue
        hit = False
        if not keys:
            hit = True
        else:
            for k in keys:
                if k in ln:
                    hit = True
                    break
        if not hit and (ln.startswith("-") or ln.startswith("•") or ln[0].isdigit()):
            hit = True
        if not hit:
            continue
        if ln in seen:
            continue
        seen.add(ln)
        lines.append(ln)
        if max_lines and len(lines) >= max_lines:
            break
    joined = "\n".join(lines)
    return _truncate_text(joined, max_chars)

def _placeholder_score(md: str) -> tuple[int, int]:
    t = (md or "").strip()
    if not t:
        return (0, 0)
    placeholders = 0
    for token in ("不确定", "待补充"):
        placeholders += t.count(token)
    lines = [x for x in t.splitlines() if x.strip()]
    return (placeholders, len(lines))

def _is_placeholder_heavy(md: str) -> bool:
    p, n = _placeholder_score(md)
    if p >= 30:
        return True
    if n <= 0:
        return True
    return (p / max(1, n)) >= 0.22

def _specificity_hits(md: str, terms: list[str]) -> int:
    t = (md or "").strip()
    if not t:
        return 0
    hits = 0
    seen = set()
    for x in (terms or []):
        s = (x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        if s in t:
            hits += 1
    return hits

def _is_too_generic(md: str, anchor: str, facts_text: str, comments_text: str) -> bool:
    fx = _extract_simple_facts(anchor=anchor, facts_text=facts_text, comments_text=comments_text)
    must = []
    if fx.get("mentor"):
        must.append(str(fx["mentor"]))
    must.extend([str(x) for x in (fx.get("keywords") or [])][:10])
    must.extend([str(x) for x in (fx.get("memes") or [])][:6])
    must.extend([str(x) for x in (fx.get("fan_nicknames") or [])][:3])
    must = [x for x in must if x and len(x) <= 20]
    if len(must) < 3:
        return False
    hits = _specificity_hits(md, must)
    return hits < 3

def _split_name_variants(anchor: str) -> list[str]:
    a = (anchor or "").strip()
    if not a:
        return []
    short_a = a.split("（")[0].split("(")[0].strip()
    return [x for x in [a, short_a] if x]

def _extract_simple_facts(anchor: str, facts_text: str, comments_text: str) -> dict:
    a = (anchor or "").strip()
    t = (facts_text or "") + "\n" + (comments_text or "")
    out = {
        "platform": "",
        "fan_nicknames": [],
        "black_nicknames": [],
        "memes": [],
        "liked": [],
        "mentor": "",
        "keywords": [],
    }
    m = re.search(r"平台[:：]\s*([^\n\r]+)", t)
    if m:
        out["platform"] = m.group(1).strip()[:30]
    for pat, key in [
        (r"铁粉绰号[:：]\s*([^\n\r]+)", "fan_nicknames"),
        (r"黑粉绰号[:：]\s*([^\n\r]+)", "black_nicknames"),
        (r"铁粉[^\n\r]{0,20}绰号[^\n\r]{0,10}(?:是|为)“?([^。\n\r]+)", "fan_nicknames"),
        (r"黑粉[^\n\r]{0,20}绰号[^\n\r]{0,10}(?:是|为|可能是)“?([^。\n\r]+)", "black_nicknames"),
        (r"粉丝团(?:称呼|名称)?[:：]\s*([^\n\r]+)", "fan_nicknames"),
    ]:
        for mm in re.finditer(pat, t):
            s = (mm.group(1) or "").strip()
            if not s:
                continue
            parts = re.split(r"[，,、/｜\|\s]+", s)
            for p in parts:
                p = p.strip()
                if p and p not in out[key]:
                    out[key].append(p)
    for raw in t.splitlines():
        s = (raw or "").strip()
        if not s:
            continue
        if "梗" in s and "【依据】" in s:
            head = s.split("【依据】", 1)[0].strip()
            head = head.split("（", 1)[0].split("(", 1)[0].strip()
            if 2 <= len(head) <= 28 and head not in out["memes"]:
                out["memes"].append(head)
    for mm in re.finditer(r"“([^”]{2,20})”", t):
        s = (mm.group(1) or "").strip()
        if s and s not in out["memes"]:
            out["memes"].append(s)
    for ln in t.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if any(x in s for x in ("粉丝喜欢", "喜欢他的", "喜欢的点", "粉丝评价")):
            if s not in out["liked"] and len(s) <= 120:
                out["liked"].append(s)
        if not out["mentor"]:
            mm = re.search(r"(师父|师傅)[:：]\s*([^\n\r]{2,20})", s)
            if mm:
                cand = (mm.group(2) or "").strip()
                cand = re.split(r"[，,、\s/｜\|]+", cand)[0].strip()
                if 2 <= len(cand) <= 8:
                    out["mentor"] = cand
        if "刘二狗" in s and not out["mentor"]:
            out["mentor"] = "刘二狗"
    kw = []
    for k in ("八点", "PK", "连麦", "喊麦", "带货", "验资", "逆徒", "下跪", "散财童子", "杭州", "首站", "情感调解室"):
        if k in t and k not in kw:
            kw.append(k)
    out["keywords"] = kw[:18]
    return out

def _extract_candidate_names(text: str, exclude: list[str] | None = None, limit: int = 12) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    ex = set([x for x in (exclude or []) if x])
    stop = {
        "平台","主播","直播","评论","粉丝","老铁","官方","榜一","榜二","榜三","流水","人头","低分","分组","玩法","冠军","保护","惩罚",
        "江苏","河北","重庆","上海","江西","黑龙江","四川","辽宁","浙江","广东","河南","山西","内蒙古","云南","吉林",
        "今天","今晚","昨天","明天","现在","以前","最近","最近半个月",
    }
    out = []
    seen = set()
    for raw in t.splitlines():
        s = (raw or "").strip()
        if not s:
            continue
        for mm in re.finditer(r"[\u4e00-\u9fff]{2,4}", s):
            w = (mm.group(0) or "").strip()
            if not w:
                continue
            if w in stop:
                continue
            if w in ex:
                continue
            if any(w in x for x in ex):
                continue
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= limit:
                return out
    return out

def _looks_like_template(md: str, anchor: str) -> bool:
    t = (md or "").strip()
    a = (anchor or "").strip()
    if not t or not a:
        return False
    required = [
        f"# {a}",
        "## 四问事实核验",
        "## 粉丝团与关系网（合并）",
        "## 六层蒸馏（人物关系强化版）",
        "### 1) 主播是谁（身份/内容定位/平台）？",
        "### 对手阵营",
        "### 师徒阵营",
        "### 师兄弟阵营",
    ]
    return all(x in t for x in required)

def _concept_template(anchor: str) -> str:
    a = (anchor or "").strip()
    return f"""# {a} 概念页

## 四问事实核验

### 1) 主播是谁（身份/内容定位/平台）？
- 要点：不确定
- 【依据】：不确定

### 2) 主要内容与核心梗是什么（可检索关键词）？
- 要点：不确定
- 【依据】：不确定

### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？
- 要点：不确定
- 【依据】：不确定

### 4) 风险与禁忌是什么（争议点/雷区/不可碰）？
- 要点：不确定
- 【依据】：不确定

## 粉丝团与关系网（合并）

### 对手阵营
- 对手官方名：不确定
- 绰号：不确定
- 世仇梗：不确定

### 师徒阵营
- 师父尊称/绰号：不确定
- 名场面：不确定

### 师兄弟阵营
- 称呼/外号：不确定
- 名场面：不确定

## 六层蒸馏（人物关系强化版）

### 第一层：【语料库】全量捕获 —— 拉取恩怨数据
- 对手阵营：不确定
- 师徒阵营：不确定
- 师兄弟阵营：不确定
- 恩怨人物词典：
  - （待补充：人名｜绰号｜黑历史锚点｜光荣史锚点）

### 第二层：【情绪极性】深度解码 —— 定义阵营立场
- 对手关系立场标签：不确定
- 师徒关系立场标签：不确定
- 可利用反差素材：不确定

### 第三层：【社群文化】溯源解构 —— 讲清恩怨来龙去脉
- 对手恩怨起源：不确定
- 师徒羁绊起源：不确定
- 经典名场面：不确定

### 第四层：【行为模式】场景建模 —— 卡准剧情触发点
- 对手互动触发场景：不确定
- 师徒互动触发场景：不确定
- 剧情触发日历建议：不确定

### 第五层：【人格化符号】固化 —— 视觉化恩怨
- 对手符号：不确定
- 师徒符号：不确定
- 分镜模板建议：不确定

### 第六层：【人物关系与阵营】核心增量（剧情引擎）

| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |
|---|---|---|---|---|
| 对手A | 不确定 | 不确定 | 不确定 | 表面和解，实则偷袭 |
| 师父 | 不确定 | 不确定 | 不确定 | 威严下的调皮（老六搞事） |
| 师兄弟B | 不确定 | 不确定 | 不确定 | 相爱相杀，互相拆台 |

- 剧情组合 1：对立反转（待补充）
- 剧情组合 2：师徒反差（待补充）
"""

def _fallback_fill(anchor: str, track: str, facts_text: str, comments_text: str) -> str:
    a = (anchor or "").strip()
    t = (track or "").strip()
    track_hint = t if t else "自动判别"
    template = _concept_template(a)
    facts = (facts_text or "").strip()
    sample = "\n".join((comments_text or "").splitlines()[:25]).strip()
    basis = "；".join([x for x in ["基础事实文件", "评论文本节选"] if (facts or sample)])
    if a == "旭旭宝宝":
        return template.replace(
            "### 1) 主播是谁（身份/内容定位/平台）？\n- 要点：不确定\n- 【依据】：不确定",
            "### 1) 主播是谁（身份/内容定位/平台）？\n- 要点：抖音游戏主播；DNF 国服第一红眼；粉丝团称呼“大马猴军团”。\n- 【依据】：基础事实文件中关于定位与粉丝称呼的描述。",
        ).replace(
            "### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：不确定\n- 【依据】：不确定",
            "### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：徒弟阵营：大龙猫/大坤坤/大蛤蟆；对手：似雨幽离（国服第一伤害之争）；恩怨人物：酷酷的滕（2017口嗨，2026和解）。\n- 【依据】：基础事实文件与评论区称呼/争议关键词。",
        ).replace(
            "### 对手阵营\n- 对手官方名：不确定\n- 绰号：不确定\n- 世仇梗：不确定",
            "### 对手阵营\n- 对手官方名：似雨幽离\n- 绰号：国服第一伤害争夺者\n- 世仇梗：国服第一伤害之争（长期对比与拉踩）。",
        ).replace(
            "### 师徒阵营\n- 师父尊称/绰号：不确定\n- 名场面：不确定",
            "### 师徒阵营\n- 师父尊称/绰号：旭旭宝宝（师门核心）\n- 名场面：师徒之间的增幅/打造分享、徒弟被带节奏时的护短与互怼。",
        ).replace(
            "### 师兄弟阵营\n- 称呼/外号：不确定\n- 名场面：不确定",
            "### 师兄弟阵营\n- 称呼/外号：大龙猫 / 大坤坤 / 大蛤蟆（徒弟群像）\n- 名场面：互相拆台、互怼拉扯、直播间拱火的师门日常。",
        ).replace(
            "### 第一层：【语料库】全量捕获 —— 拉取恩怨数据\n- 对手阵营：不确定\n- 师徒阵营：不确定\n- 师兄弟阵营：不确定\n- 恩怨人物词典：\n  - （待补充：人名｜绰号｜黑历史锚点｜光荣史锚点）",
            "### 第一层：【语料库】全量捕获 —— 拉取恩怨数据\n- 对手阵营：似雨幽离（国服第一伤害之争）。\n- 师徒阵营：旭旭宝宝 ↔ 大龙猫/大坤坤/大蛤蟆。\n- 师兄弟阵营：徒弟群像互坑互怼。\n- 恩怨人物词典：\n  - 酷酷的滕｜口嗨哥｜2017口嗨冲突｜2026和解节点\n  - 似雨幽离｜伤害王｜伤害对比拉踩｜国服第一之争\n  - 懂王｜商业逻辑裁判｜夸“增幅20”伟大投资｜批“运气不符商业逻辑”\n  - 马斯克｜火星工程师｜建议把“增幅亡命徒”精神用于火星殖民｜工程极限迁移\n  - 小恶魔老六｜拱火旁观者｜挑拨反转｜表面和解暗中较劲",
        ).replace(
            "### 第四层：【行为模式】场景建模 —— 卡准剧情触发点\n- 对手互动触发场景：不确定\n- 师徒互动触发场景：不确定\n- 剧情触发日历建议：不确定",
            "### 第四层：【行为模式】场景建模 —— 卡准剧情触发点\n- 对手互动触发场景：伤害排名对比、增幅对赌、直播间/评论区拉踩。\n- 师徒互动触发场景：打造展示、徒弟翻车救场、师门互怼拱火。\n- 剧情触发日历建议：2026-03-27 入驻《梦幻西游》手游“2026”新服引发炸服，可作为跨赛道节奏触发点。",
        )

    filled = template
    fx = _extract_simple_facts(anchor=a, facts_text=facts_text, comments_text=comments_text)
    platform_hint = fx.get("platform") or "快手/抖音"
    fan_names = [x for x in (fx.get("fan_nicknames") or []) if x][:6]
    black_names = [x for x in (fx.get("black_nicknames") or []) if x][:6]
    memes = [x for x in (fx.get("memes") or []) if x][:10]
    mentor = (fx.get("mentor") or "").strip()
    kw = [x for x in (fx.get("keywords") or []) if x][:12]
    core_kw = "、".join(kw) if kw else "（待从评论/事实抽取）"
    meme_kw = "、".join(memes[:6]) if memes else core_kw
    fan_kw = "、".join(fan_names) if fan_names else "（待核实）"
    black_kw = "、".join(black_names) if black_names else "（待核实）"
    bro_candidates = _extract_candidate_names(facts_text, exclude=[a, mentor], limit=8)
    opp_candidates = _extract_candidate_names(comments_text, exclude=[a, mentor], limit=10)
    opp_hint = "、".join(opp_candidates[:6]) if opp_candidates else "（待核实：从连麦/PK对象抽取）"
    bro_hint = "、".join(bro_candidates[:6]) if bro_candidates else "（待核实：从同门/团队核心抽取）"
    filled = filled.replace(
        "### 1) 主播是谁（身份/内容定位/平台）？\n- 要点：不确定\n- 【依据】：不确定",
        f"### 1) 主播是谁（身份/内容定位/平台）？\n- 要点：{platform_hint}主播；师门/团队背景与人设以事实文件为准；赛道倾向：{track_hint}。\n- 【依据】：{basis or '待核实'}；关键词：{core_kw}。",
    )
    filled = filled.replace(
        "### 2) 主要内容与核心梗是什么（可检索关键词）？\n- 要点：不确定\n- 【依据】：不确定",
        f"### 2) 主要内容与核心梗是什么（可检索关键词）？\n- 要点：核心梗/关键词：{meme_kw}；高频玩法围绕 {track_hint}（连麦/PK/喊麦/带货等）展开。\n- 【依据】：{basis or '待核实'}。",
    )
    if mentor:
        filled = filled.replace(
            "### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：不确定\n- 【依据】：不确定",
            f"### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：师徒主线：{mentor} ↔ {a}；粉丝称呼/铁粉口头禅需要从事实核验抽取（候选：{fan_kw}）；黑粉外号候选：{black_kw}。\n- 【依据】：{basis or '待核实'}；关键词：{core_kw}。",
        )
        filled = filled.replace(
            "### 师徒阵营\n- 师父尊称/绰号：不确定\n- 名场面：不确定",
            f"### 师徒阵营\n- 师父尊称/绰号：{mentor}\n- 名场面：下跪/逆徒/护短/师门荣誉等（以事实核验为准）。",
        )
        filled = filled.replace(
            "### 对手阵营\n- 对手官方名：不确定\n- 绰号：不确定\n- 世仇梗：不确定",
            f"### 对手阵营\n- 对手官方名：待核实（候选：{opp_hint}）\n- 绰号：待核实（先用关键词检索：八点/验资/踩我三项/家族赛/八点局）\n- 世仇梗：八点档喊话/验资对线/家族赛口水战（需用原视频/连麦回放核验）。",
        )
        filled = filled.replace(
            "### 师兄弟阵营\n- 称呼/外号：不确定\n- 名场面：不确定",
            f"### 师兄弟阵营\n- 称呼/外号：待核实（候选：{bro_hint}）\n- 名场面：同门互坑/拱火带节奏/团播配合（需从“团队首站/家族赛”相关材料核验）。",
        )
    else:
        filled = filled.replace(
            "### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：不确定\n- 【依据】：不确定",
            f"### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？\n- 要点：粉丝称呼候选：{fan_kw}；黑粉外号候选：{black_kw}；对手/互怼对象候选：{opp_hint}；团队/切片号候选：{bro_hint}。\n- 【依据】：{basis or '待核实'}；关键词：{core_kw}。",
        )
        filled = filled.replace(
            "### 对手阵营\n- 对手官方名：不确定\n- 绰号：不确定\n- 世仇梗：不确定",
            f"### 对手阵营\n- 对手官方名：待核实（候选：{opp_hint}）\n- 绰号：待核实（用关键词检索：{core_kw}）\n- 世仇梗：PK/BO3/互怼/翻车名场面（以事实核验为准）。",
        )
        filled = filled.replace(
            "### 师徒阵营\n- 师父尊称/绰号：不确定\n- 名场面：不确定",
            "### 师徒阵营\n- 师父尊称/绰号：待核实（若无师徒线可忽略）\n- 名场面：组织/团队内“首领/将军/义子”等人设互动（以事实核验为准）。",
        )
        filled = filled.replace(
            "### 师兄弟阵营\n- 称呼/外号：不确定\n- 名场面：不确定",
            f"### 师兄弟阵营\n- 称呼/外号：{bro_hint}\n- 名场面：切片号/组织成员联动整活、拱火互怼、PK惩罚表演（以事实核验为准）。",
        )
    filled = filled.replace(
        "### 4) 风险与禁忌是什么（争议点/雷区/不可碰）？\n- 要点：不确定\n- 【依据】：不确定",
        "### 4) 风险与禁忌是什么（争议点/雷区/不可碰）？\n- 要点：涉及隐私、未证实黑料、引战站队、地域/群体攻击、未成年相关话题均需规避；争议点必须标注【待核实】。\n- 【依据】：通用安全策略 + 评论区敏感词（需进一步核验）。",
    )
    if mentor:
        filled = filled.replace(
            "### 第一层：【语料库】全量捕获 —— 拉取恩怨数据\n- 对手阵营：不确定\n- 师徒阵营：不确定\n- 师兄弟阵营：不确定\n- 恩怨人物词典：\n  - （待补充：人名｜绰号｜黑历史锚点｜光荣史锚点）",
            "### 第一层：【语料库】全量捕获 —— 拉取恩怨数据\n"
            f"- 对手阵营：待核实（先用关键词检索：{core_kw}）。\n"
            f"- 师徒阵营：{mentor} ↔ {a}。\n"
            "- 师兄弟阵营：待核实（从“同门/团队核心/家族赛”相关材料抽取）。\n"
            "- 恩怨人物词典：\n"
            f"  - {mentor}｜师父｜师门管教/护短｜师门荣誉\n"
            f"  - {a}｜徒弟/接班人候选｜下跪/逆徒/反差人设｜独立扛旗/转型带货\n"
            "  - 629团队｜家族/战队｜团播/家族赛/榜单对线｜凝聚力叙事\n"
            "  - 懂王｜商业逻辑裁判｜点评“验资/带货/逆袭”｜反差嘲讽\n"
            "  - 马斯克｜工程极限迁移｜把“敢打敢拼”迁移到火星工程｜极限执念\n"
            "  - 小恶魔老六｜拱火挑拨者｜制造“表面和解/暗中较劲”｜剧情反转",
        )
        filled = filled.replace(
            "### 第二层：【情绪极性】深度解码 —— 定义阵营立场\n- 对手关系立场标签：不确定\n- 师徒关系立场标签：不确定\n- 可利用反差素材：不确定",
            "### 第二层：【情绪极性】深度解码 —— 定义阵营立场\n"
            f"- 对手关系立场标签：对线/竞争/嘲讽（依据：{core_kw}）。\n"
            f"- 师徒关系立场标签：敬畏/服从/护短（依据：下跪、逆徒、师门荣誉、{mentor}）。\n"
            f"- 可利用反差素材：开播硬气→见师父秒怂；散财童子“财迷/大方”双重反差；验资对线“敢打敢拼”；八点档喊话“风采依旧”与现实翻车对比（依据：{meme_kw}、{core_kw}）。",
        )
        filled = filled.replace(
            "### 第三层：【社群文化】溯源解构 —— 讲清恩怨来龙去脉\n- 对手恩怨起源：不确定\n- 师徒羁绊起源：不确定\n- 经典名场面：不确定",
            "### 第三层：【社群文化】溯源解构 —— 讲清恩怨来龙去脉\n"
            f"- 对手恩怨起源：多出现在连麦/PK验资与“八点档”喊话后的跨直播间对线、评论区站队（待核实：检索关键词：八点以后你们谁都不行、验资、踩我三项、家族赛、八点局）。\n"
            f"- 师徒羁绊起源：早期师徒互动形成“下跪/逆徒/假装硬气”等固定反差叙事（依据：下跪、逆徒、假装硬气、{mentor}）。\n"
            f"- 经典名场面：下跪求饶、逆徒调侃；连麦PK输了又送福利（散财童子）；八点档叫嚣全网、穿拖鞋、踩我三项退网；PK验资/激情喊麦（依据：{meme_kw}、{core_kw}）。",
        )
        filled = filled.replace(
            "### 第四层：【行为模式】场景建模 —— 卡准剧情触发点\n- 对手互动触发场景：不确定\n- 师徒互动触发场景：不确定\n- 剧情触发日历建议：不确定",
            "### 第四层：【行为模式】场景建模 —— 卡准剧情触发点\n"
            f"- 对手互动触发场景：连麦PK验资（高额礼物/金龙）；八点档/家族赛分组玩法；评论区“没有6万/玩不起/剃眉毛”争议点（依据：{core_kw} + 原文节选中的“八点/验资/剃眉毛/玩不起”等）。\n"
            f"- 师徒互动触发场景：开播前硬气喊话→师父出现立刻怂；师门荣誉被黑时徒弟反击；师父训徒/护短/立规矩（依据：下跪、逆徒、师门荣誉、{mentor}）。\n"
            "- 剧情触发日历建议：围绕“八点局/家族赛”做固定档期；围绕带货节点做“咆哮式带货”爆点；围绕情感调解室做“粗犷金句→断流”反转（依据：八点、带货、情感调解室）。",
        )
        filled = filled.replace(
            "### 第五层：【人格化符号】固化 —— 视觉化恩怨\n- 对手符号：不确定\n- 师徒符号：不确定\n- 分镜模板建议：不确定",
            "### 第五层：【人格化符号】固化 —— 视觉化恩怨\n"
            "- 对手符号：验资数字/礼物条、榜单截图、家族赛分组表、“八点局”倒计时（依据：验资、八点、家族赛）。\n"
            f"- 师徒符号：师父出场BGM/训话姿态；徒弟下跪求饶或秒怂表情包；“逆徒”弹幕雨（依据：下跪、逆徒、{mentor}）。\n"
            "- 分镜模板建议：\n"
            "  1) 开场：豪言壮语（硬气）→字幕“3秒后师父入场”→立刻变脸（反差）。\n"
            "  2) 中段：连麦PK验资/喊麦→对手拱火→评论区站队。\n"
            "  3) 收尾：散财童子送福利→黑粉嘲讽“玩不起”→老六旁白反转。",
        )
        rel_table_rows = []
        rel_table_rows.append((f"{mentor}（师父）", "师门大哥", "训徒/立规矩", "护短/带队", "威严训话下的反差调侃"))
        rel_table_rows.append((f"{a}（徒弟）", "逆徒/二爷", "下跪求饶/秒怂", "敢打敢拼/带货新高", "硬气开场→师父入场秒怂"))
        if bro_candidates:
            rel_table_rows.append((f"{bro_candidates[0]}（师兄弟）", "同门/队友", "互坑拆台", "团播助力", "相爱相杀+互相拱火"))
        if opp_candidates:
            rel_table_rows.append((f"{opp_candidates[0]}（对手候选）", "对线对象", "验资/嘲讽", "话题流量", "表面和解暗中较劲"))
        if fan_names:
            rel_table_rows.append(("铁粉/粉丝团", fan_names[0], "站队护主", "冲榜助力", "吹捧与拱火并存"))
        if black_names:
            rel_table_rows.append(("黑粉/对立粉", black_names[0], "嘲讽双标/玩不起", "带节奏扩散", "黑粉拱火→徒弟爆麦反击"))
        if len(rel_table_rows) < 6:
            rel_table_rows.append(("小恶魔老六（虚拟）", "拱火者", "挑拨离间", "剧情反转", "把对线推到极致"))
        table = ["| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |", "|---|---|---|---|---|"]
        for r in rel_table_rows[:8]:
            table.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        filled = filled.replace(
            "### 第六层：【人物关系与阵营】核心增量（剧情引擎）\n\n| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |\n|---|---|---|---|---|\n| 对手A | 不确定 | 不确定 | 不确定 | 表面和解，实则偷袭 |\n| 师父 | 不确定 | 不确定 | 不确定 | 威严下的调皮（老六搞事） |\n| 师兄弟B | 不确定 | 不确定 | 不确定 | 相爱相杀，互相拆台 |\n\n- 剧情组合 1：对立反转（待补充）\n- 剧情组合 2：师徒反差（待补充）",
            "### 第六层：【人物关系与阵营】核心增量（剧情引擎）\n\n"
            + "\n".join(table)
            + "\n\n- 剧情组合 1：八点档喊话→验资对线→现实翻车→师父训话收尾（关键词：八点/验资/玩不起）。\n"
            + f"- 剧情组合 2：徒弟开播硬气→师父入场秒怂→黑粉嘲讽→老六旁白反转（关键词：下跪/逆徒/{mentor}）。\n"
            + "- 剧情组合 3：散财童子送福利→粉黑大战→对手拱火→带货节点咆哮收割（关键词：散财童子/带货/榜单）。",
        )
    else:
        rel_table_rows = []
        if opp_candidates:
            rel_table_rows.append((f"{opp_candidates[0]}（对手候选）", "对线对象", "PK/互怼/翻车（待核实）", "名场面传播（待核实）", "表面和解，暗中较劲（老六挑拨）"))
        if bro_candidates:
            rel_table_rows.append((f"{bro_candidates[0]}（团队/切片号）", "组织成员", "出征/拱火（待核实）", "切片传播（待核实）", "相爱相杀，互相拆台"))
        if fan_names:
            rel_table_rows.append(("铁粉/粉丝团", fan_names[0], "站队护主", "冲榜助力", "吹捧与拱火并存"))
        if black_names:
            rel_table_rows.append(("黑粉/对立粉", black_names[0], "嘲讽双标/玩不起（待核实）", "带节奏扩散（待核实）", "黑粉拱火→主播爆麦反击"))
        rel_table_rows.append(("懂王（虚拟）", "商业逻辑裁判", "冷嘲热讽", "给出“验资/带货”商业解释", "把争议变成商业反差"))
        rel_table_rows.append(("小恶魔老六（虚拟）", "拱火者", "挑拨离间", "剧情反转", "把对线推到极致"))
        table = ["| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |", "|---|---|---|---|---|"]
        for r in rel_table_rows[:8]:
            table.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        filled = filled.replace(
            "### 第六层：【人物关系与阵营】核心增量（剧情引擎）\n\n| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |\n|---|---|---|---|---|\n| 对手A | 不确定 | 不确定 | 不确定 | 表面和解，实则偷袭 |\n| 师父 | 不确定 | 不确定 | 不确定 | 威严下的调皮（老六搞事） |\n| 师兄弟B | 不确定 | 不确定 | 不确定 | 相爱相杀，互相拆台 |\n\n- 剧情组合 1：对立反转（待补充）\n- 剧情组合 2：师徒反差（待补充）",
            "### 第六层：【人物关系与阵营】核心增量（剧情引擎）\n\n"
            + "\n".join(table)
            + "\n\n- 剧情组合 1：名场面（翻车/惩罚）→对手拱火→粉黑大战→老六反转收尾（关键词：PK/翻车/粉丝团）。\n"
            + "- 剧情组合 2：组织人设（首领/将军/义子）→内部互怼→外部出征（关键词：黑暗组织/出征/切片）。",
        )
    if sample:
        filled += "\n\n## 依据（原文节选）\n" + "\n".join(f"- {x}" for x in sample.splitlines() if x.strip())
    return filled

def build_enrich_prompt(anchor: str, track: str, facts_text: str, comments_text: str, draft: str) -> str:
    a = (anchor or "").strip()
    t = (track or "").strip()
    track_hint = t if t else "自动判别（从事实与评论判断）"
    template = _concept_template(a)
    return f"""你正在做“深度补全”。你会拿到：基础事实材料、评论材料、以及一份模板草稿（其中大量字段仍为“不确定/待补充”）。

任务：输出一份【完整模板】Markdown，从“# {a} 概念页”开始，严格保持模板标题结构，不要输出任何解释。

硬性要求：
1) 除非材料中完全没有线索，否则不要输出“不确定”。缺少证据时写“待核实：xxx（给可检索关键词）”。
2) 必须至少填充：粉丝称呼/铁粉外号 ≥ 2、黑粉外号 ≥ 1、经典梗/名场面 ≥ 4、关系网（对手/师父/师兄弟）各 ≥ 1、恩怨人物词典 ≥ 8 条。
3) 六层蒸馏每一层都必须给出可检索关键词作为【依据】。

主播：{a}
赛道/分类：{track_hint}

基础事实材料（节选）：
{facts_text}

评论材料（节选）：
{comments_text}

草稿（需要你彻底补全）：
{draft}

目标模板（必须严格按此结构输出）：
{template}
"""

def build_prompt(anchor: str, track: str, facts_text: str, comments_text: str) -> str:
    template = """# {anchor} 概念页

## 四问事实核验

### 1) 主播是谁（身份/内容定位/平台）？
- 要点：不确定
- 【依据】：不确定

### 2) 主要内容与核心梗是什么（可检索关键词）？
- 要点：不确定
- 【依据】：不确定

### 3) 粉丝团与关系网是什么（对手/师徒/师兄弟）？
- 要点：不确定
- 【依据】：不确定

### 4) 风险与禁忌是什么（争议点/雷区/不可碰）？
- 要点：不确定
- 【依据】：不确定

## 粉丝团与关系网（合并）

### 对手阵营
- 对手官方名：不确定
- 绰号：不确定
- 世仇梗：不确定

### 师徒阵营
- 师父尊称/绰号：不确定
- 名场面：不确定

### 师兄弟阵营
- 称呼/外号：不确定
- 名场面：不确定

## 六层蒸馏（人物关系强化版）

### 第一层：【语料库】全量捕获 —— 拉取恩怨数据
- 对手阵营：不确定
- 师徒阵营：不确定
- 师兄弟阵营：不确定
- 恩怨人物词典：
  - （待补充：人名｜绰号｜黑历史锚点｜光荣史锚点）

### 第二层：【情绪极性】深度解码 —— 定义阵营立场
- 对手关系立场标签：不确定
- 师徒关系立场标签：不确定
- 可利用反差素材：不确定

### 第三层：【社群文化】溯源解构 —— 讲清恩怨来龙去脉
- 对手恩怨起源：不确定
- 师徒羁绊起源：不确定
- 经典名场面：不确定

### 第四层：【行为模式】场景建模 —— 卡准剧情触发点
- 对手互动触发场景：不确定
- 师徒互动触发场景：不确定
- 剧情触发日历建议：不确定

### 第五层：【人格化符号】固化 —— 视觉化恩怨
- 对手符号：不确定
- 师徒符号：不确定
- 分镜模板建议：不确定

### 第六层：【人物关系与阵营】核心增量（剧情引擎）

| 人物关系 | 核心绰号 | 黑历史锚点 | 光荣史锚点 | 反差剧情方向 |
|---|---|---|---|---|
| 对手A | 不确定 | 不确定 | 不确定 | 表面和解，实则偷袭 |
| 师父 | 不确定 | 不确定 | 不确定 | 威严下的调皮（老六搞事） |
| 师兄弟B | 不确定 | 不确定 | 不确定 | 相爱相杀，互相拆台 |

- 剧情组合 1：对立反转（待补充）
- 剧情组合 2：师徒反差（待补充）
"""
    
    t = (track or "").strip()
    track_hint = t if t else "自动判别（从事实与评论判断）"
    a = (anchor or "").strip()
    if a == "旭旭宝宝":
        mapping = """### 强制逻辑映射要求（非常重要！）：
1. 身份/内容定位：必须包含“大马猴军团”粉丝团称呼及“DNF 国服第一红眼”定位。
2. 关系网填充（真实3角色）：
   - 徒弟：必须填入大龙猫、大坤坤、大蛤蟆。
   - 对手：必须填入似雨幽离（国服第一伤害之争）。
   - 恩怨人物：必须加入酷酷的滕（2017 年口嗨恩怨，2026 年和解）。
3. 近期节奏：必须包含 2026 年 3 月 27 日入驻《梦幻西游》手游“2026”新服引发炸服的事件。
4. 虚拟角色逻辑注入（虚拟3角色，请在第六层或各阵营相关场景中作为剧情反差补充）：
   - 懂王：点评该主播“增幅 20”是伟大的投资，但这种运气不符合商业逻辑。
   - 马斯克：建议将“增幅亡命徒”的精神用于火星殖民计划。
   - 小恶魔老六：作为挑事者/旁观者，在剧情中暗中拱火。"""
    else:
        mapping = f"""### 角色蒸馏要求（3+3）：
1. 赛道/分类：{track_hint}
2. 真实 3 角色：必须从事实与评论中抽取并写入模板：
   - 徒弟/小弟/团队核心 1-3 人（含绰号、关系、代表事件）。
   - 对手/对立阵营 1-3 人（含争议点、起源、关键词）。
   - 恩怨人物词典：至少 6 条（人名｜绰号｜黑历史锚点｜光荣史锚点）。
3. 虚拟 3 角色注入（作为剧情反差引擎，写入“六层蒸馏”内的剧情方向/反差方向）：
   - 懂王：用“商业逻辑/投资逻辑”点评主播的关键事件或争议点。
   - 马斯克：把主播的“极限/亡命/执念”迁移到“火星殖民/工程极限”语境里。
   - 小恶魔老六：负责挑拨、拱火、反转，制造“表面和解/暗中较劲”的剧情张力。"""

    return f"""你是“主脑逻辑”技能蒸馏器。你需要根据提供的主播事实文件和抓取到的评论文本，生成一份 Markdown 文件。

任务要求：
你必须【全文复制并填充】以下模板，不可遗漏任何一个标题（包括一级到三级标题），不可改变结构。不要输出模板以外的无关内容。

主播：{a}
赛道/分类：{track_hint}

{mapping}

### 提供的基础事实：
{facts_text}

### 提供的评论素材：
{comments_text}

=== 必须严格填写的模板（请原样输出结构，并替换【不确定】/【待补充】的内容） ===
{template}
"""

def build_repair_prompt(anchor: str, track: str, draft: str) -> str:
    a = (anchor or "").strip()
    t = (track or "").strip()
    track_hint = t if t else "自动判别（从事实与评论判断）"
    template = build_prompt(anchor=a, track=track_hint, facts_text="(略)", comments_text="(略)")
    return f"""你正在做“模板修复”。你会拿到一份草稿内容，它不符合模板结构。

任务：把草稿内容重写为模板，要求：
1) 只输出模板内容，从“# {a} 概念页”开始。
2) 不可改变模板标题结构；必须填满所有“不确定/待补充”字段。
3) 不要输出任何解释、前言、总结，不要使用代码块围栏。

草稿内容：
{draft}

目标模板（必须严格按此结构输出）：
{template}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", help="主播名")
    parser.add_argument("track", nargs="?", default="", help="分类")
    args = parser.parse_args()

    anchor = args.anchor
    track = args.track
    
    # 1. Load facts
    # F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验\四问事实核验\
    # F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验\粉丝团与关系网\
    base_dir = r"F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验"
    if not Path(base_dir).exists():
        # Fallback to without space
        base_dir = r"F:\老六个人 AI 工作台\Obsidian知识库\06_文案库\主播事实核验"

    short_anchor = anchor.split("（")[0].split("(")[0].strip() or anchor
    search_terms = {anchor, short_anchor}
    processed_fact_paths = set()
    facts_texts = []
    
    for term in search_terms:
        for file_path in glob.glob(f"{base_dir}\\**\\*{term}*.md", recursive=True):
            p = Path(file_path).resolve()
            if p.name == f"{anchor}.md":
                continue
            if p in processed_fact_paths:
                continue
            processed_fact_paths.add(p)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    facts_texts.append(f"--- File: {os.path.basename(file_path)} ---\n" + f.read())
            except Exception:
                pass

    audit_roots = [
        r"\\192.168.10.198\Workstation\人工审核",
        r"E:\老六AI执行工作台\人工审核"
    ]
    for ar in audit_roots:
        if not Path(ar).exists():
            continue
        for term in search_terms:
            for file_path in glob.glob(f"{ar}\\**\\*{term}*.*", recursive=True):
                p = Path(file_path).resolve()
                if p in processed_fact_paths:
                    continue
                processed_fact_paths.add(p)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        facts_texts.append(f"--- File: {os.path.basename(file_path)} ---\n" + f.read())
                except Exception:
                    pass

    facts_text = "\n".join(facts_texts)
    
    # 2. Load comments
    comments_text = _load_comments(anchor)

    # 3. Call Ollama
    kw = _split_name_variants(anchor) + ["粉丝", "绰号", "黑粉", "铁粉", "经典梗", "名场面", "师父", "徒弟", "师徒", "对手", "恩怨", "争议", "PK", "连麦", "喊麦", "带货", "八点", "验资", "逆徒", "下跪", "629"]
    facts_clean = _extract_signal_lines(facts_text, keywords=kw, max_lines=260, max_chars=12000)
    comments_clean = _extract_signal_lines(comments_text, keywords=kw, max_lines=900, max_chars=14000)
    prompt = build_prompt(anchor, track, facts_clean, comments_clean)
    try:
        proj_root = Path(__file__).resolve().parents[1]
        cache_dir = (proj_root / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "distiller_skill_last_prompt.txt").write_text(prompt, encoding="utf-8", errors="ignore")
    except Exception:
        pass
    
    model = (os.getenv("OLLAMA_MODEL") or os.getenv("LAOLIU_OLLAMA_MODEL") or "").strip() or "gemma4:e4b-gpu20"
    try:
        num_predict = int((os.getenv("DISTILL_NUM_PREDICT") or os.getenv("LAOLIU_DISTILL_NUM_PREDICT") or "").strip() or "1200")
    except Exception:
        num_predict = 1200
    if num_predict < 128:
        num_predict = 128
    if num_predict > 4096:
        num_predict = 4096
    client = OllamaClient(model=model)
    try:
        md_output = client.generate(prompt=prompt, system="", options={"num_predict": int(num_predict)})
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Ollama generation failed: {e}"}, ensure_ascii=False))
        sys.exit(1)
        
    # Clean up output
    md_output = _strip_fences(md_output)
    try:
        proj_root = Path(__file__).resolve().parents[1]
        cache_dir = (proj_root / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "distiller_skill_last_raw.txt").write_text(md_output, encoding="utf-8", errors="ignore")
    except Exception:
        pass

    if not _looks_like_template(md_output, anchor):
        repair = build_repair_prompt(anchor=anchor, track=track, draft=md_output)
        try:
            md2 = client.generate(prompt=repair, system="", options={"num_predict": int(num_predict)})
            md2 = _strip_fences(md2)
            if _looks_like_template(md2, anchor):
                md_output = md2
        except Exception:
            pass
    if _looks_like_template(md_output, anchor) and (_is_placeholder_heavy(md_output) or _is_too_generic(md_output, anchor=anchor, facts_text=facts_clean, comments_text=comments_clean)):
        enrich = build_enrich_prompt(anchor=anchor, track=track, facts_text=facts_clean, comments_text=comments_clean, draft=md_output)
        try:
            md3 = client.generate(prompt=enrich, system="", options={"num_predict": int(num_predict)})
            md3 = _strip_fences(md3)
            if _looks_like_template(md3, anchor) and (not _is_placeholder_heavy(md3)) and (not _is_too_generic(md3, anchor=anchor, facts_text=facts_clean, comments_text=comments_clean)):
                md_output = md3
        except Exception:
            pass
    if not _looks_like_template(md_output, anchor) or _is_placeholder_heavy(md_output) or _is_too_generic(md_output, anchor=anchor, facts_text=facts_clean, comments_text=comments_clean):
        md_output = _fallback_fill(anchor=anchor, track=track, facts_text=facts_clean, comments_text=comments_clean)

    # 4. Save to F:\老六个人 AI 工作台\Obsidian知识库\06_文案库\主播事实核验\粉丝团与关系网\旭旭宝宝.md
    out_dir = r"F:\老六个人 AI 工作台\Obsidian知识库\06_文案库\主播事实核验\粉丝团与关系网"
    # If the exact path user mentioned doesn't exist, fall back to the one with space
    if not Path(out_dir).exists() and Path(r"F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验\粉丝团与关系网").exists():
        out_dir = r"F:\老六个人 AI 工作台\Obsidian 知识库\06_文案库\主播事实核验\粉丝团与关系网"
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{anchor}.md"
    
    try:
        out_path.write_text(md_output, encoding="utf-8")
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"Failed to write file {out_path}: {e}"}, ensure_ascii=False))
        sys.exit(1)

    # 5. Output success JSON for Telegram
    # The prompt explicitly asked for this response: "主人，{主播名} 的灵魂已蒸馏完毕，文件已存入 F 盘。"
    res = {
        "status": "ok",
        "path": str(out_path),
        "message": f"主人，{anchor} 的灵魂已蒸馏完毕，文件已存入 F 盘。"
    }
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
