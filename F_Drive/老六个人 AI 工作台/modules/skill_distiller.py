from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from modules.ollama_client import OllamaClient


TRACK_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "游戏主播": {
        "互坑": [
            "故意立 flag，然后安排翻车",
            "对手嘲讽：也就这点水平，也就只会嘴硬",
            "挖坑：敢不敢 SOLO？不敢就是怂",
            "挑拨：你师傅都比你打得稳",
            "搞心态：又在找借口，输不起是吧",
            "带节奏：上次就是你坑队友，别装了",
            "阴阳怪气：哇，好厉害哦，不会真有人觉得这很秀吧",
        ],
        "节奏": [
            "主播急眼了",
            "主播破防了",
            "主播又开始嘴硬",
            "主播又在找借口",
            "主播被打脸了",
            "主播又被吊打了",
        ],
    },
    "娱乐/脱口秀/PK": {
        "互坑": [
            "嘲讽对方没票、没大哥、拉了",
            "挖坑：有本事别跑，继续 PK",
            "挑拨：你师门都不帮你，混得真差",
            "恶搞：你刚才那表情太丑了，我截表情包了",
            "使坏：我故意逗你，看你急不急",
            "带节奏：大家快看，他急了他急了",
        ],
        "节奏": [
            "主播破防名场面",
            "主播被气得语无伦次",
            "主播被怼得没话说",
            "主播被恶搞到脸红",
            "主播被对面拿捏了",
        ],
    },
    "带货主播": {
        "互坑": [
            "嘲讽：你那价格也叫福利？坑粉丝呢",
            "挖坑：敢不敢把底价亮出来？不敢就是暴利",
            "使坏：我故意问你敏感问题，看你怎么圆",
            "挑拨：你粉丝都觉得你卖贵了",
            "搞心态：又在演戏，又在剧本",
            "带节奏：大家别信，他就是割韭菜",
        ],
        "节奏": [
            "又开始演戏了",
            "剧本又来了",
            "主播被问住了",
            "主播心虚了",
            "主播开始转移话题",
        ],
    },
    "军事/历史/知识": {
        "互坑": [
            "抬杠：你这观点不对，史料不是这么写的",
            "挖坑：你敢不敢为你说的话负责？",
            "使坏：我故意挑你逻辑漏洞",
            "嘲讽：也就只会抄别人观点",
            "带节奏：又在博眼球，又在制造焦虑",
        ],
        "节奏": [
            "主播被打脸",
            "主播逻辑漏洞",
            "主播又在瞎分析",
            "主播被问得哑口无言",
        ],
    },
    "财经主播": {
        "互坑": [
            "挖坑：你上次预测错了，怎么解释？",
            "使坏：故意问敏感股票，让他违规",
            "嘲讽：你自己都没赚还教别人？",
            "带节奏：又在割韭菜，又在卖课",
            "挑拨：你就是马后炮，涨了才说",
        ],
        "节奏": [
            "主播又被打脸",
            "主播又在马后炮",
            "主播又在制造焦虑",
            "主播不敢正面回答",
        ],
    },
}


def normalize_track(track: str) -> str:
    t = (track or "").strip()
    if not t:
        return ""
    t = t.replace("赛道", "").strip()
    aliases = {
        "游戏": "游戏主播",
        "游戏主播": "游戏主播",
        "娱乐": "娱乐/脱口秀/PK",
        "脱口秀": "娱乐/脱口秀/PK",
        "pk": "娱乐/脱口秀/PK",
        "PK": "娱乐/脱口秀/PK",
        "娱乐/脱口秀/PK": "娱乐/脱口秀/PK",
        "带货": "带货主播",
        "带货主播": "带货主播",
        "军事": "军事/历史/知识",
        "历史": "军事/历史/知识",
        "知识": "军事/历史/知识",
        "军事/历史/知识": "军事/历史/知识",
        "财经": "财经主播",
        "财经主播": "财经主播",
    }
    return aliases.get(t, t)


def track_templates_markdown() -> str:
    parts: list[str] = []
    for k, v in TRACK_TEMPLATES.items():
        parts.append(f"### {k}")
        parts.append("")
        parts.append("互坑/使坏桥段：")
        parts.extend([f"- {x}" for x in v.get("互坑", [])])
        parts.append("")
        parts.append("评论区通用节奏：")
        parts.extend([f"- {x}" for x in v.get("节奏", [])])
        parts.append("")
    return "\n".join(parts).strip()


def _default_input_roots() -> list[Path]:
    candidates = [
        os.getenv("COMMENT_ROOT"),
        r"E:\老六AI执行工作台\抓取结果_D",
        r"E:\老六AI执行工作台\抓取结果",
        r"E:\抓取结果",
        r"\\192.168.10.198\Workstation\抓取结果_D",
        r"\\192.168.10.198\Workstation\抓取结果",
        r"\\SC-202501141403\Workstation\抓取结果_D",
        r"\\SC-202501141403\Workstation\抓取结果",
        str(Path(__file__).resolve().parents[1] / "抓取结果"),
    ]
    out: list[Path] = []
    for c in candidates:
        if not c:
            continue
        p = Path(str(c))
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _default_skills_root() -> Path:
    env_dir = os.getenv("SKILLS_DIR")
    if env_dir:
        return Path(env_dir)
    root = Path(__file__).resolve().parents[1]
    spaced = root / "Obsidian 知识库"
    if spaced.exists() and spaced.is_dir():
        return spaced / "Skills"
    return root / "Obsidian知识库" / "Skills"


def _sanitize_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80] if len(s) > 80 else s


def _clean_line(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(@\S+|回复\s*@\S+)\s*", "", t)
    t = re.sub(r"^[\d\W_]+", "", t)
    t = t.strip()
    if len(t) < 2:
        return ""
    if re.fullmatch(r"[!！?？。，,.、~…]+", t):
        return ""
    return t


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@dataclass
class DistillConfig:
    input_roots: list[Path]
    output_root: Path
    model: str
    base_url: str
    max_comment_lines: int = 500
    max_total_chars: int = 12000
    num_predict: int = 1200
    track: str = ""
    stream: bool = False


class SkillDistiller:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        self.client = OllamaClient(base_url=cfg.base_url, model=cfg.model)

    @classmethod
    def default(cls) -> "SkillDistiller":
        return cls(
            DistillConfig(
                input_roots=_default_input_roots(),
                output_root=_default_skills_root(),
                model=os.getenv("OLLAMA_MODEL") or "gemma",
                base_url=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
            )
        )

    def list_anchors(self) -> list[str]:
        names: set[str] = set()
        for root in self.cfg.input_roots:
            if not root.exists():
                continue
            for p in root.iterdir():
                if p.is_dir():
                    names.add(p.name)
        return sorted(names)

    def _find_anchor_dir(self, anchor: str) -> Path | None:
        for root in self.cfg.input_roots:
            p = root / anchor
            if p.exists() and p.is_dir():
                return p
        return None

    def _load_comments(self, anchor_dir: Path) -> list[str]:
        txts = sorted(anchor_dir.rglob("*.txt"))
        lines: list[str] = []
        for p in txts:
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for ln in raw.splitlines():
                c = _clean_line(ln)
                if c:
                    lines.append(c)

        lines = _dedupe_keep_order(lines)
        if self.cfg.max_comment_lines > 0:
            lines = lines[: int(self.cfg.max_comment_lines)]

        if self.cfg.max_total_chars > 0:
            out: list[str] = []
            total = 0
            for ln in lines:
                add = len(ln) + 1
                if total + add > int(self.cfg.max_total_chars):
                    break
                out.append(ln)
                total += add
            lines = out
        return lines

    def _build_prompt(self, anchor: str, comments: list[str]) -> str:
        joined = "\n".join(f"- {c}" for c in comments)
        track = normalize_track(self.cfg.track)
        track_hint = track if track else "自动判别（从评论判断最像哪一类）"
        templates = track_templates_markdown()
        return f"""你是“技能蒸馏器”。输入是一批直播间/短视频评论原文，你要从中蒸馏出主播的 Real 3 人设技能，并输出一份可直接落库的 SKILL.md。

输出要求：
1) 只输出 Markdown（不要代码块围栏），文件名固定 SKILL.md。
2) 内容必须可执行：包含口吻、禁忌、价值观、常用句式、互动策略、开场/收尾模板、以及 6-12 条“示例回复”（覆盖：怼黑粉、接梗、卖货、情绪安抚、转移话题、拉关注）。
3) 不能出现“根据评论/这批数据/我推测”等措辞，直接当作你已非常了解主播。
4) 不要输出任何 <think>/<thought> 标签本身。

赛道要求：
- 赛道：{track_hint}
- 你必须在最终 SKILL.md 里输出一个“## Track”小节，写清楚赛道归类与一句话理由（理由不得提“评论/数据”）。
- 你必须在最终 SKILL.md 里输出一个“## Track Pranks”小节，给出该赛道可复用的互坑模板：至少 8 条“互坑/使坏桥段”与至少 6 条“评论区通用节奏”，并结合该主播口吻改写成可直接复用的话术。

请以如下结构输出（必须包含这些标题）：
# {anchor} · SKILL

## Persona

## Voice & Tone

## Catchphrases

## Values

## Boundaries (Must NOT)

## Interaction Playbook

## Track

## Track Pranks

## Reply Patterns

## Examples

评论原文（去重后节选）：
{joined}

五大赛道通用互坑模板（参考用；若指定了赛道就只用对应赛道的模板；未指定就先归类再用模板）：
{templates}

<thought>先在脑内总结“人设核心”“情绪曲线”“禁忌雷区”，再写 SKILL.md。不要把 thought 输出到最终内容。</thought>
"""

    def distill(self, anchor: str) -> tuple[Path, str]:
        name = _sanitize_name(anchor)
        if not name:
            raise ValueError("anchor 不能为空")
        anchor_dir = self._find_anchor_dir(anchor)
        if anchor_dir is None:
            raise FileNotFoundError(f"未找到主播目录：{anchor}")

        comments = self._load_comments(anchor_dir)
        if not comments:
            raise RuntimeError(f"未读取到评论 txt：{anchor_dir}")

        prompt = self._build_prompt(anchor=name, comments=comments)
        options = {"num_predict": int(self.cfg.num_predict)} if int(self.cfg.num_predict or 0) > 0 else None
        md = self.client.generate(prompt=prompt, system="", options=options, model=self.cfg.model, stream=bool(self.cfg.stream))

        out_dir = (self.cfg.output_root / name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "SKILL.md"
        out_path.write_text(md.strip() + "\n", encoding="utf-8")
        return out_path, md
