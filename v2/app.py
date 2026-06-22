from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus, urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components

try:
    from openai import OpenAI
except ImportError:  # Keep the app usable enough to show a clear setup error.
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:  # Claude support is optional unless that provider is selected.
    Anthropic = None

try:  # PyYAML makes frontmatter parsing/dumping robust; we fall back to a tiny parser if absent.
    import yaml
except ImportError:
    yaml = None


DATA_DIR = Path("wiki_data")
KEYS_DIR = Path("keys")
PROMPT_DIR = Path("prmopt")
DEFAULT_PROMPT_NAME = "chinese_history_wiki_maintainer"
# Legacy default prompt name from the generic "Raw LLM Wiki" build. Kept so older
# installs migrate cleanly to the Chinese-history default without orphaned files.
LEGACY_PROMPT_NAME = "default_wiki_maintainer"
DEFAULT_WIKI_PROMPT = """你是一個專業的中國歷史維基百科維護者。
你的任務是幫助維護一個持久、相互連結的本地 Markdown 歷史維基。
核心規則：
- 除非使用者介面明確要求，否則僅返回純 Markdown 格式。
- 一律使用繁體中文撰寫；若輸入為簡體字，請轉換為繁體中文。
- 保持客觀、嚴謹的歷史學術基調（類似《史記》或現代歷史教科書的客觀陳述）。
- 遇到存在歷史爭議的事件或人物，必須明確列出不同的歷史觀點，不要捏造事實。
- 使用清晰的標題（如：背景、經過、結果、歷史評價）。
- 當提到已有的歷史事件、人物、朝代或地點時，請使用準確的專有名詞，以便系統自動產生連結。
- 總結必須簡明扼要，基於史實。
Markdown 樣式：
- 主要頁面以一段簡短的歷史背景概述開始。
- 頻繁使用無序列表來梳理時間線、重要戰役或關鍵人物。
- 避免使用原始 HTML。
"""
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "llama-3.2-1b-instruct"
DEFAULT_MAX_TOKENS = 4096

# --- Chinese historical NER configuration -------------------------------------
# Primary: a BERT-based Chinese NER checkpoint. Fallback: an Apache-2.0 model with
# an explicit PER/LOC/TIME label set. Both load through transformers' `pipeline`.
NER_MODEL_PRIMARY = "shibing624/bert4ner-base-chinese"
NER_MODEL_FALLBACK = "ckiplab/bert-base-chinese-ner"
NER_SCORE_THRESHOLD = 0.50
NER_MIN_ENTITY_LEN = 2
# Single characters that are still meaningful historical surnames/markers worth keeping.
NER_KEEP_SINGLE = frozenset("孔孟李王劉刘曹秦漢汉唐宋元明清晉晋隋")
# Normalize the many raw label spellings (BERT, OntoNotes, POS-style) to PER/LOC/TIME.
# Per the spec we keep only person / location / time; ORG, NORP, etc. are dropped so
# organizations are not mislabeled as locations.
NER_LABEL_MAP = {
    "PER": "PER", "PERSON": "PER", "NR": "PER", "NAME": "PER",
    "LOC": "LOC", "LOCATION": "LOC", "NS": "LOC", "GPE": "LOC", "FAC": "LOC", "ADDRESS": "LOC", "SCENE": "LOC",
    "TIME": "TIME", "DATE": "TIME", "T": "TIME", "TIMEX": "TIME",
}
NER_TYPE_LABELS = {"PER": "歷史人物 (Person)", "LOC": "地點 (Location)", "TIME": "時間/朝代 (Time)"}

# --- OCR configuration --------------------------------------------------------
# Three selectable engines: two offline (PaddleOCR / Tesseract) and one that
# reuses the already-configured chat provider's vision model.
OCR_ENGINE_PADDLE = "本地 PaddleOCR"
OCR_ENGINE_TESSERACT = "本地 Tesseract"
OCR_ENGINE_VISION = "Vision LLM (視覺模型)"
# PaddleOCR language code vs. Tesseract traineddata names.
OCR_PADDLE_LANG = "ch"
OCR_TESSERACT_LANG = "chi_sim+chi_tra"
OCR_IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
OCR_UPLOAD_TYPES = OCR_IMAGE_TYPES + ["pdf"]
OCR_PAGE_SEPARATOR = "\n\n----- 第 {n} 頁 -----\n\n"


# Dependency-free gazetteer: dynasties/eras used both by the regex fallback NER and
# to infer the timeline `year` when the LLM omits it. Years are approximate starts (CE; negative = BCE).
# Both Traditional and Simplified surface forms are listed so matching works whether
# the page text uses 漢朝/晉朝/遼朝 (Traditional) or 汉朝/晋朝/辽朝 (Simplified).
DYNASTY_TIMELINE: list[tuple[str, int]] = [
    ("夏朝", -2070), ("夏", -2070),
    ("商朝", -1600), ("商", -1600),
    ("西周", -1046), ("周朝", -1046), ("東周", -770), ("东周", -770), ("周", -1046),
    ("春秋", -770), ("戰國", -475), ("战国", -475),
    ("秦朝", -221), ("秦", -221),
    ("西漢", -202), ("西汉", -202), ("漢朝", -202), ("汉朝", -202),
    ("東漢", 25), ("东汉", 25), ("漢", -202), ("汉", -202),
    ("三國", 220), ("三国", 220), ("曹魏", 220),
    ("蜀漢", 221), ("蜀汉", 221), ("東吳", 229), ("东吴", 229),
    ("西晉", 265), ("西晋", 265), ("東晉", 317), ("东晋", 317),
    ("晉朝", 265), ("晋朝", 265), ("晉", 265), ("晋", 265),
    ("南北朝", 420), ("隋朝", 581), ("隋", 581),
    ("唐朝", 618), ("唐", 618),
    ("五代十國", 907), ("五代十国", 907), ("五代", 907),
    ("北宋", 960), ("宋朝", 960), ("南宋", 1127), ("宋", 960),
    ("遼朝", 916), ("辽朝", 916), ("遼", 916), ("辽", 916),
    ("金朝", 1115), ("西夏", 1038),
    ("元朝", 1271), ("元", 1271),
    ("明朝", 1368), ("明", 1368),
    ("清朝", 1636), ("清", 1636),
    ("中華民國", 1912), ("中华民国", 1912), ("民國", 1912), ("民国", 1912),
    ("中華人民共和國", 1949), ("中华人民共和国", 1949), ("新中國", 1949), ("新中国", 1949),
]

SOURCE_LOCAL = "Local model"
SOURCE_ONLINE = "Online API"
PROVIDER_LOCAL = "Local OpenAI-compatible"
PROVIDER_OPENAI = "OpenAI GPT"
PROVIDER_GEMINI = "Google Gemini"
PROVIDER_CLAUDE = "Anthropic Claude"
PROVIDER_NEWAPI = "third party provider"
PROVIDER_CUSTOM = "Custom OpenAI-compatible"
BACKEND_OPENAI_COMPATIBLE = "openai_compatible"
BACKEND_ANTHROPIC = "anthropic"
OPENAI_ENDPOINT_SUFFIXES = ("/chat/completions", "/completions", "/embeddings")
PROVIDER_CONFIGS = {
    PROVIDER_LOCAL: {
        "backend": BACKEND_OPENAI_COMPATIBLE,
        "base_url": DEFAULT_BASE_URL,
        "api_key": DEFAULT_API_KEY,
        "api_key_fallback": DEFAULT_API_KEY,
        "model": DEFAULT_MODEL,
        "auto_resolve_model": True,
        "help": "Use LM Studio, Ollama's OpenAI-compatible server, or another local /v1 chat endpoint.",
    },
    PROVIDER_OPENAI: {
        "backend": BACKEND_OPENAI_COMPATIBLE,
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "api_key_env": "OPENAI_API_KEY",
        "api_key_fallback": "",
        "model": "gpt-4o-mini",
        "auto_resolve_model": True,
        "help": "Use OpenAI-hosted GPT models with an OpenAI API key.",
    },
    PROVIDER_GEMINI: {
        "backend": BACKEND_OPENAI_COMPATIBLE,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "api_key_env": "GEMINI_API_KEY",
        "api_key_fallback": "",
        "model": "gemini-2.0-flash",
        "auto_resolve_model": True,
        "help": "Use Gemini's OpenAI-compatible endpoint with a Gemini API key.",
    },
    PROVIDER_CLAUDE: {
        "backend": BACKEND_ANTHROPIC,
        "base_url": "",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_fallback": "",
        "model": "claude-3-5-sonnet-latest",
        "auto_resolve_model": False,
        "help": "Use Anthropic's Claude Messages API with an Anthropic API key.",
    },
    PROVIDER_NEWAPI: {
        "backend": BACKEND_OPENAI_COMPATIBLE,
        "base_url": "https://www.juaiapi.com/v1",
        "api_key": "",
        "api_key_fallback": "",
        "model": "",
        "auto_resolve_model": True,
        "help": "Use JuAI or another New API proxy with an OpenAI-compatible /v1 endpoint. Leave Model blank to try models from /v1/models.",
    },
    PROVIDER_CUSTOM: {
        "backend": BACKEND_OPENAI_COMPATIBLE,
        "base_url": "https://www.juaiapi.com/v1",
        "api_key": "",
        "api_key_fallback": "",
        "model": "",
        "auto_resolve_model": True,
        "help": "Use any remote provider that exposes an OpenAI-compatible chat completions API. Set Base URL to the provider root or /v1 endpoint.",
    },
}
ONLINE_PROVIDERS = [PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_CLAUDE, PROVIDER_NEWAPI, PROVIDER_CUSTOM]


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(exist_ok=True)


def ensure_keys_dir() -> None:
    KEYS_DIR.mkdir(exist_ok=True)


def ensure_prompt_dir() -> None:
    PROMPT_DIR.mkdir(exist_ok=True)


def prompt_path(name: str) -> Path:
    safe_name = key_file_stem(name) or DEFAULT_PROMPT_NAME
    return PROMPT_DIR / f"{safe_name}.md"


def ensure_default_prompt() -> None:
    ensure_prompt_dir()
    path = prompt_path(DEFAULT_PROMPT_NAME)
    if not path.exists():
        path.write_text(DEFAULT_WIKI_PROMPT, encoding="utf-8")
    # One-time migration: retire the generic English default from older installs so
    # the prompt list isn't cluttered. Only removes it if untouched (still the stock
    # English maintainer text); customized legacy prompts are preserved.
    legacy_path = prompt_path(LEGACY_PROMPT_NAME)
    if legacy_path.exists():
        try:
            legacy_text = legacy_path.read_text(encoding="utf-8")
        except OSError:
            legacy_text = ""
        if legacy_text.lstrip().startswith("You are an autonomous Markdown wiki maintainer"):
            try:
                legacy_path.unlink()
            except OSError:
                pass


# Chinese / full-width punctuation that is legal on some filesystems but unsafe or
# ugly across operating systems. Replaced with "_" so Chinese titles stay readable.
_CJK_PUNCT_PATTERN = re.compile(
    "["
    + re.escape("：；，。、！？“”‘’「」『』《》〈〉（）【】〔〕［］｛｝｜＼／＜＞＊…·・")
    + "]"
)


def sanitize_title(raw_title: str) -> str:
    """Convert user input into a safe, cross-platform flat-file page title.

    Handles both ASCII-forbidden characters and Chinese/full-width punctuation
    (：，。、《》「」“” （） …) so Chinese titles produce valid filenames on
    Windows, macOS, and Linux while remaining human-readable.
    """
    title = raw_title.strip()
    # Treat the ideographic space (U+3000) as ordinary whitespace.
    title = title.replace("　", " ")
    # 1) ASCII characters that are illegal in Windows/Unix filenames.
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    # 2) Chinese / full-width punctuation -> "_" (keeps word separation).
    title = _CJK_PUNCT_PATTERN.sub("_", title)
    # 3) Collapse whitespace, then fold any run containing "_" into a single "_",
    #    and trim separators from the ends. Internal spaces in Latin titles survive.
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"\s*_[\s_]*", "_", title)
    title = title.strip(" ._")
    return title[:80] or "Untitled"


def page_path(title: str) -> Path:
    safe_title = sanitize_title(title)
    return DATA_DIR / f"{safe_title}.md"


def scan_pages() -> list[str]:
    ensure_data_dir()
    return sorted(path.stem for path in DATA_DIR.glob("*.md"))


def scan_prompts() -> list[str]:
    ensure_default_prompt()
    return sorted(path.stem for path in PROMPT_DIR.glob("*.md"))


def read_prompt(name: str) -> str:
    path = prompt_path(name)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_prompt(name: str, content: str) -> str:
    ensure_prompt_dir()
    safe_name = key_file_stem(name) or DEFAULT_PROMPT_NAME
    prompt_path(safe_name).write_text(content, encoding="utf-8")
    return safe_name


def create_prompt(name: str, content: str = "") -> tuple[str, bool]:
    ensure_prompt_dir()
    safe_name = key_file_stem(name) or "new_prompt"
    path = prompt_path(safe_name)
    if path.exists():
        return safe_name, False
    path.write_text(content or DEFAULT_WIKI_PROMPT, encoding="utf-8")
    return safe_name, True


def delete_prompt(name: str) -> bool:
    if name == DEFAULT_PROMPT_NAME:
        return False
    path = prompt_path(name)
    if not path.exists():
        return False
    path.unlink()
    return True


def get_active_prompt_name() -> str:
    prompts = scan_prompts()
    active = st.session_state.get("active_prompt", DEFAULT_PROMPT_NAME)
    if active not in prompts:
        active = DEFAULT_PROMPT_NAME if DEFAULT_PROMPT_NAME in prompts else prompts[0]
        st.session_state.active_prompt = active
    return active


def get_active_prompt() -> str:
    return read_prompt(get_active_prompt_name()).strip() or DEFAULT_WIKI_PROMPT.strip()


def key_file_stem(provider: str) -> str:
    return re.sub(r"_+", "_", sanitize_title(provider).lower().replace(" ", "_")).strip("_")


def api_key_path(provider: str) -> Path:
    return KEYS_DIR / f"{key_file_stem(provider)}.json"


def fallback_key_paths(provider: str) -> list[Path]:
    if provider == PROVIDER_NEWAPI:
        return [api_key_path(PROVIDER_CUSTOM), api_key_path(PROVIDER_OPENAI)]
    return []


def read_connection_file(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    raw_key = payload.get("api_key") or payload.get("key") or ""
    raw_url = payload.get("base_url") or payload.get("url") or ""
    if isinstance(raw_key, str):
        nested_connection = parse_api_key_input(raw_key)
        raw_key = nested_connection.get("api_key", raw_key)
        raw_url = raw_url or nested_connection.get("base_url", "")
    api_key = raw_key.strip() if isinstance(raw_key, str) else ""
    base_url = raw_url.strip() if isinstance(raw_url, str) else ""
    connection = {}
    if api_key:
        connection["api_key"] = api_key
    if base_url:
        connection["base_url"] = base_url
    return connection


def load_saved_connection(provider: str) -> dict[str, str]:
    path = api_key_path(provider)
    if path.exists():
        return read_connection_file(path)

    for fallback_path in fallback_key_paths(provider):
        connection = read_connection_file(fallback_path)
        if "juaiapi.com" in connection.get("base_url", ""):
            return connection
    return {}


def saved_connection_version(provider: str) -> int:
    version = 0
    for path in [api_key_path(provider), *fallback_key_paths(provider)]:
        try:
            version = max(version, path.stat().st_mtime_ns)
        except OSError:
            continue
    return version


def load_saved_api_key(provider: str) -> str:
    return load_saved_connection(provider).get("api_key", "")


def load_saved_base_url(provider: str) -> str:
    return load_saved_connection(provider).get("base_url", "")


def save_api_key(provider: str, api_key: str, base_url: str = "") -> None:
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("Cannot save an empty API key.")
    ensure_keys_dir()
    payload = {
        "provider": provider,
        "api_key": cleaned_key,
    }
    cleaned_url = base_url.strip()
    if cleaned_url:
        payload["base_url"] = normalize_base_url(cleaned_url)
    api_key_path(provider).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def delete_saved_api_key(provider: str) -> bool:
    path = api_key_path(provider)
    if not path.exists():
        return False
    path.unlink()
    return True


def parse_api_key_input(raw_value: str) -> dict[str, str]:
    cleaned = raw_value.strip()
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"api_key": cleaned}
    if not isinstance(payload, dict):
        return {"api_key": cleaned}

    raw_key = payload.get("api_key") or payload.get("key") or ""
    raw_url = payload.get("base_url") or payload.get("url") or ""
    connection: dict[str, str] = {}
    if isinstance(raw_key, str) and raw_key.strip():
        connection["api_key"] = raw_key.strip()
    if isinstance(raw_url, str) and raw_url.strip():
        connection["base_url"] = raw_url.strip()
    return connection


_FRONTMATTER_RE = re.compile(
    r"^﻿?---[ \t]*\r?\n(?P<fm>.*?)\r?\n---[ \t]*\r?\n?(?P<body>.*)$",
    re.DOTALL,
)
FRONTMATTER_FIELD_ORDER = ("title", "dynasty", "year", "tags")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        inner = value[1:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    return value


def _coerce_year(value: object) -> int | None:
    """Best-effort parse of a frontmatter year into an int (negative = BCE)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    year = int(match.group())
    # Honour an explicit BCE marker, but only one that genuinely denotes "before
    # common era" — NOT the bare 前 inside CE dynasty names like 前秦/前赵/前燕 or the
    # ordinary word 之前/以前. Require 公元前/西元前, an English BCE/BC token, or a 前
    # immediately adjacent to the digits (e.g. 前202年).
    if year > 0 and re.search(
        r"(公元前|西元前|(?<![之以])前\s*\d|\bBCE?\b|\bB\.C\.?)", text, re.IGNORECASE
    ):
        year = -year
    return year


def _split_inline_list(inner: str) -> list[str]:
    """Split an inline-list body `a, "b,c", d` on commas, honouring quoted items."""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for char in inner:
        if quote:
            buf.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            buf.append(char)
        elif char == ",":
            items.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    if buf:
        items.append("".join(buf))
    return [_strip_quotes(part) for part in items if _strip_quotes(part)]


def _parse_frontmatter_minimal(text: str) -> dict[str, object]:
    """Tiny YAML-subset parser used only when PyYAML is unavailable.

    Supports `key: scalar`, inline lists `key: [a, b]`, and block lists (`- item`).
    """
    meta: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        list_item = re.match(r"^\s*-\s+(.*)$", line)
        if list_item and current_list_key is not None:
            # Upgrade the placeholder (None, set when the `key:` header was read)
            # to a real list on the first item, then append.
            if not isinstance(meta.get(current_list_key), list):
                meta[current_list_key] = []
            meta[current_list_key].append(_strip_quotes(list_item.group(1)))
            continue
        field = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", line)
        if not field:
            continue
        key, value = field.group(1), field.group(2).strip()
        if value == "":
            # Either an empty scalar or the header of a following block list.
            meta[key] = None
            current_list_key = key
            continue
        current_list_key = None
        if value.startswith("[") and value.endswith("]"):
            meta[key] = _split_inline_list(value[1:-1])
        else:
            meta[key] = _strip_quotes(value)
    return meta


def parse_frontmatter_text(text: str) -> dict[str, object]:
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text)
            return loaded if isinstance(loaded, dict) else {}
        except yaml.YAMLError:
            return _parse_frontmatter_minimal(text)
    return _parse_frontmatter_minimal(text)


def _looks_like_frontmatter_block(fm_text: str) -> bool:
    """True only if every non-empty line looks like YAML (key:, list item, or scalar).

    Guards against treating a Markdown body that merely *starts* with a `---`
    thematic break as a frontmatter block (which would silently swallow content).
    """
    saw_key = False
    for raw_line in fm_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_\-]+\s*:(\s|$)", line):
            saw_key = True
            continue
        if line.startswith("- "):  # block-list item under a preceding key
            continue
        # Anything else (prose, headings, a lone "-", etc.) -> not frontmatter.
        return False
    return saw_key


def split_frontmatter(raw_text: str) -> tuple[dict[str, object], str]:
    """Split a stored page into (frontmatter dict, markdown body).

    A leading ``---`` block is only treated as frontmatter when its contents
    actually look like YAML key/value lines; otherwise the whole text (including a
    leading thematic break) is returned as the body so no content is lost.
    """
    if not raw_text:
        return {}, raw_text or ""
    match = _FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text
    fm_text = match.group("fm")
    if not _looks_like_frontmatter_block(fm_text):
        return {}, raw_text
    meta = parse_frontmatter_text(fm_text)
    if not isinstance(meta, dict) or not meta:
        return {}, raw_text
    return meta, match.group("body")


def _dump_scalar_minimal(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    needs_quotes = (
        text == ""
        or text.strip() != text
        or re.search(r'[:#\[\]{}"\',\n]', text) is not None
        or text[0] in "!&*?|>%@`-"
    )
    if needs_quotes:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def dump_frontmatter(meta: dict[str, object]) -> str:
    """Serialize metadata to a `---`-delimited YAML frontmatter block."""
    ordered: dict[str, object] = {}
    for key in FRONTMATTER_FIELD_ORDER:
        if key in meta:
            ordered[key] = meta[key]
    for key, value in meta.items():
        if key not in ordered:
            ordered[key] = value

    if yaml is not None:
        body = yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return f"---\n{body}---\n"

    lines: list[str] = []
    for key, value in ordered.items():
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(_dump_scalar_minimal(item) for item in value)
            lines.append(f"{key}: [{rendered}]")
        else:
            lines.append(f"{key}: {_dump_scalar_minimal(value)}".rstrip())
    return "---\n" + "\n".join(lines) + "\n---\n"


def infer_dynasty_year(*texts: str) -> tuple[str | None, int | None]:
    """Find the earliest-mentioned dynasty/era across the given texts via the gazetteer."""
    for text in texts:
        if not text:
            continue
        for name, year in DYNASTY_TIMELINE:
            if name in text:
                return name, year
    return None, None


def default_page_meta(title: str, body: str = "") -> dict[str, object]:
    dynasty, year = infer_dynasty_year(title, body)
    meta: dict[str, object] = {"title": title}
    meta["dynasty"] = dynasty or ""
    meta["year"] = year
    tags = ["歷史"]
    if dynasty:
        tags.append(dynasty)
    meta["tags"] = tags
    return meta


def serialize_page(meta: dict[str, object], body: str) -> str:
    frontmatter = dump_frontmatter(meta)
    body = (body or "").strip("\n")
    if body:
        return f"{frontmatter}\n{body}\n"
    return frontmatter


def read_page_raw(title: str) -> str:
    """Full stored file contents, including frontmatter (used by the editor)."""
    path = page_path(title)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_page(title: str) -> str:
    """Markdown body only, with any frontmatter stripped (used for display/AI/NER)."""
    _, body = split_frontmatter(read_page_raw(title))
    return body


def read_page_meta(title: str) -> dict[str, object]:
    meta, _ = split_frontmatter(read_page_raw(title))
    return meta


def write_page(
    title: str,
    content: str,
    meta: dict[str, object] | None = None,
    *,
    merge_existing: bool = True,
) -> None:
    """Persist a page, always writing YAML frontmatter at the top.

    Frontmatter precedence (low -> high): inferred defaults, existing on-disk
    frontmatter (when merge_existing), frontmatter embedded in `content`, and the
    explicit `meta` override. The markdown body is taken from `content`.
    """
    ensure_data_dir()
    embedded_meta, body = split_frontmatter(content)
    resolved: dict[str, object] = {}
    if merge_existing:
        resolved.update(read_page_meta(title))
    resolved.update(embedded_meta)
    if meta:
        resolved.update(meta)

    # Backfill any missing core fields from inference so frontmatter is always complete.
    defaults = default_page_meta(title, body)
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    resolved["title"] = title  # Title always tracks the (sanitized) page name.
    if resolved.get("dynasty") in (None, ""):
        dynasty, year = infer_dynasty_year(title, body)
        if dynasty:
            resolved["dynasty"] = dynasty
            resolved.setdefault("year", year)
    resolved["year"] = _coerce_year(resolved.get("year"))

    page_path(title).write_text(serialize_page(resolved, body), encoding="utf-8")


def delete_page(title: str) -> bool:
    path = page_path(title)
    if not path.exists():
        return False
    path.unlink()
    return True


def create_page(title: str) -> tuple[str, bool]:
    ensure_data_dir()
    safe_title = sanitize_title(title)
    path = page_path(safe_title)
    if path.exists():
        return safe_title, False
    path.write_text(serialize_page(default_page_meta(safe_title), ""), encoding="utf-8")
    return safe_title, True


def get_query_page() -> str | None:
    try:
        value = st.query_params.get("page")
    except Exception:
        value = st.experimental_get_query_params().get("page", [None])[0]
    if isinstance(value, list):
        value = value[0] if value else None
    return value if value else None


def set_query_page(title: str) -> None:
    try:
        st.query_params["page"] = title
    except Exception:
        st.experimental_set_query_params(page=title)


def clear_query_page() -> None:
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    st.experimental_rerun()


def init_state(pages: list[str]) -> None:
    st.session_state.setdefault("current_page", None)
    st.session_state.setdefault("edit_mode", False)
    st.session_state.setdefault("show_new_page", False)
    st.session_state.setdefault("related_topics", [])
    st.session_state.setdefault("active_prompt", DEFAULT_PROMPT_NAME)
    st.session_state.setdefault("show_new_prompt", False)
    st.session_state.setdefault("extracted_entities", [])
    st.session_state.setdefault("extracted_entities_page", None)
    st.session_state.setdefault("ner_method", None)
    st.session_state.setdefault("ocr_text", "")
    st.session_state.setdefault("ocr_source_name", None)

    query_page = get_query_page()
    if query_page in pages:
        st.session_state.current_page = query_page

    if st.session_state.current_page not in pages:
        st.session_state.current_page = pages[0] if pages else None
        st.session_state.edit_mode = False


def select_page(title: str) -> None:
    st.session_state.current_page = title
    st.session_state.edit_mode = False
    st.session_state.related_topics = []
    st.session_state.extracted_entities = []
    st.session_state.extracted_entities_page = None
    set_query_page(title)
    rerun()


def build_system_prompt(task_prompt: str) -> str:
    base_prompt = get_active_prompt()
    return f"{base_prompt}\n\nTask-specific instruction:\n{task_prompt.strip()}"


def get_provider_config(provider: str) -> dict[str, object]:
    return PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS[PROVIDER_LOCAL])


def normalize_base_url(base_url: str, default: str = DEFAULT_BASE_URL) -> str:
    candidate = (base_url or default).strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        for suffix in OPENAI_ENDPOINT_SUFFIXES:
            if path.endswith(suffix):
                path = path[: -len(suffix)].rstrip("/") or "/v1"
                return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))
        if path in ("", "/"):
            return urlunsplit((parsed.scheme, parsed.netloc, "/v1", parsed.query, parsed.fragment))
    return candidate or default


def resolve_base_url(provider: str, base_url: str) -> str:
    config = get_provider_config(provider)
    saved_base_url = load_saved_base_url(provider)
    default_base_url = str(config.get("base_url") or DEFAULT_BASE_URL)
    return normalize_base_url(base_url or saved_base_url, default_base_url)


def resolve_api_key(provider: str, api_key: str) -> str:
    config = get_provider_config(provider)
    raw_key = parse_api_key_input(api_key or "").get("api_key", "")
    saved_key = load_saved_api_key(provider)
    env_name = str(config.get("api_key_env") or "")
    env_key = os.getenv(env_name, "") if env_name else ""
    fallback = str(config.get("api_key_fallback") or "")
    resolved = raw_key or saved_key or env_key or fallback
    if not resolved:
        raise RuntimeError(
            f"Missing API key for {provider}. Enter one in the sidebar"
            + (f", save it in `keys/`, or set {env_name}." if env_name else " or save it in `keys/`.")
        )
    return resolved


def make_openai_client(provider: str, base_url: str, api_key: str):
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Run `pip install openai`.")
    return OpenAI(base_url=resolve_base_url(provider, base_url), api_key=resolve_api_key(provider, api_key))


def make_anthropic_client(provider: str, api_key: str):
    if Anthropic is None:
        raise RuntimeError("The `anthropic` package is not installed. Run `pip install anthropic`.")
    return Anthropic(api_key=resolve_api_key(provider, api_key))


def is_embedding_model(model_id: str) -> bool:
    return "embed" in model_id.lower()


def get_available_model_ids(client) -> list[str]:
    try:
        response = client.models.list()
    except Exception:
        return []

    model_items = getattr(response, "data", [])
    model_ids: list[str] = []
    for item in model_items:
        model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())
    return model_ids


def test_openai_compatible_connection(provider: str, base_url: str, api_key: str) -> list[str]:
    client = make_openai_client(provider, base_url, api_key)
    return get_available_model_ids(client)


def resolve_model(
    client,
    requested_model: str,
    fallback_model: str,
    *,
    auto_resolve: bool,
) -> str:
    requested = (requested_model or "").strip() or fallback_model
    if not requested:
        raise RuntimeError("Missing model name. Enter a model in the sidebar.")
    if not auto_resolve:
        return requested

    available_models = get_available_model_ids(client)
    if not available_models or requested in available_models:
        return requested

    chat_models = [model_id for model_id in available_models if not is_embedding_model(model_id)]
    return chat_models[0] if chat_models else requested


def get_model_candidates(client, requested_model: str, fallback_model: str, *, auto_resolve: bool) -> list[str]:
    requested = (requested_model or "").strip()
    candidates: list[str] = []
    for model_id in (requested, fallback_model):
        if model_id and model_id not in candidates:
            candidates.append(model_id)

    if auto_resolve:
        for model_id in get_available_model_ids(client):
            if model_id not in candidates and not is_embedding_model(model_id):
                candidates.append(model_id)

    if not candidates:
        raise RuntimeError("Missing model name. Enter a model in the sidebar.")
    return candidates


def should_try_next_model(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        signal in message
        for signal in (
            "no access to model",
            "model_not_found",
            "does not have access",
            "not have access",
            "invalid model",
            "model_not_supported",
        )
    )


def format_model_retry_error(provider: str, attempted_models: list[str], exc: Exception) -> RuntimeError:
    attempted = ", ".join(attempted_models)
    return RuntimeError(
        f"{provider} rejected the attempted model(s): {attempted}. "
        "Choose a model your token can access, or use a saved New API connection whose token exposes models. "
        f"Last error: {exc}"
    )


def extract_anthropic_text(response) -> str:
    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def stream_chat(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
):
    config = get_provider_config(provider)
    backend = config.get("backend")
    fallback_model = str(config.get("model") or DEFAULT_MODEL)

    if backend == BACKEND_ANTHROPIC:
        client = make_anthropic_client(provider, api_key)
        with client.messages.stream(
            model=(model or fallback_model).strip(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
        return

    client = make_openai_client(provider, base_url, api_key)
    attempted_models: list[str] = []
    candidates = get_model_candidates(
        client,
        model,
        fallback_model,
        auto_resolve=bool(config.get("auto_resolve_model")),
    )
    for candidate_model in candidates:
        attempted_models.append(candidate_model)
        try:
            stream = client.chat.completions.create(
                model=candidate_model,
                temperature=temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return
        except Exception as exc:
            if not should_try_next_model(exc) or candidate_model == candidates[-1]:
                if should_try_next_model(exc):
                    raise format_model_retry_error(provider, attempted_models, exc) from exc
                raise


def complete_chat(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
) -> str:
    config = get_provider_config(provider)
    backend = config.get("backend")
    fallback_model = str(config.get("model") or DEFAULT_MODEL)

    if backend == BACKEND_ANTHROPIC:
        client = make_anthropic_client(provider, api_key)
        response = client.messages.create(
            model=(model or fallback_model).strip(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return extract_anthropic_text(response)

    client = make_openai_client(provider, base_url, api_key)
    attempted_models: list[str] = []
    candidates = get_model_candidates(
        client,
        model,
        fallback_model,
        auto_resolve=bool(config.get("auto_resolve_model")),
    )
    for candidate_model in candidates:
        attempted_models.append(candidate_model)
        try:
            response = client.chat.completions.create(
                model=candidate_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if not should_try_next_model(exc) or candidate_model == candidates[-1]:
                if should_try_next_model(exc):
                    raise format_model_retry_error(provider, attempted_models, exc) from exc
                raise
    return ""


def complete_vision_chat(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    image_b64: str,
    image_mime: str = "image/png",
) -> str:
    """Send a single image plus a text instruction to a vision-capable chat model.

    Supports both the Anthropic Messages API (native image blocks) and any
    OpenAI-compatible chat endpoint (``image_url`` data-URI content parts). Used by
    the OCR "Vision LLM" engine to transcribe text from an uploaded image.
    """
    config = get_provider_config(provider)
    backend = config.get("backend")
    fallback_model = str(config.get("model") or DEFAULT_MODEL)

    if backend == BACKEND_ANTHROPIC:
        client = make_anthropic_client(provider, api_key)
        response = client.messages.create(
            model=(model or fallback_model).strip(),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_mime,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        return extract_anthropic_text(response)

    client = make_openai_client(provider, base_url, api_key)
    data_uri = f"data:{image_mime};base64,{image_b64}"
    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    attempted_models: list[str] = []
    candidates = get_model_candidates(
        client,
        model,
        fallback_model,
        auto_resolve=bool(config.get("auto_resolve_model")),
    )
    for candidate_model in candidates:
        attempted_models.append(candidate_model)
        try:
            response = client.chat.completions.create(
                model=candidate_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if not should_try_next_model(exc) or candidate_model == candidates[-1]:
                if should_try_next_model(exc):
                    raise format_model_retry_error(provider, attempted_models, exc) from exc
                raise
    return ""


def format_llm_error(exc: Exception, provider: str, base_url: str) -> str:
    config = get_provider_config(provider)
    if config.get("backend") == BACKEND_ANTHROPIC:
        return f"Claude request failed for `{provider}`. Check your API key, model, and `anthropic` package. Details: {exc}"
    if should_try_next_model(exc):
        return (
            f"LLM request failed for `{provider}` at `{resolve_base_url(provider, base_url)}` because the token "
            f"does not have access to the selected model. Change the Model field to one allowed by this API token. "
            f"Details: {exc}"
        )
    return (
        f"LLM request failed for `{provider}` at `{resolve_base_url(provider, base_url)}`. "
        f"Check the API key, model, provider endpoint, and network access. Details: {exc}"
    )


def markdown_escape_link_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _autolink_subpattern(title: str) -> str:
    """Exact-match sub-pattern for one title.

    Chinese text has no whitespace word boundaries, so we match the title string
    exactly. We only add a boundary guard when an *edge* character is ASCII
    alphanumeric (so a Latin title like ``Han`` won't link inside ``Handan``); CJK
    edges get no guard, which is what makes Chinese auto-linking work.
    """
    escaped = re.escape(title)
    first, last = title[:1], title[-1:]
    left = r"(?<![A-Za-z0-9_])" if first.isascii() and (first.isalnum() or first == "_") else ""
    right = r"(?![A-Za-z0-9_])" if last.isascii() and (last.isalnum() or last == "_") else ""
    return f"{left}{escaped}{right}"


def autolink_markdown(markdown_text: str, page_titles: list[str], current_title: str | None) -> str:
    """Lightweight wiki linking while avoiding code blocks, code spans, and existing links.

    Uses exact string matching over the page titles (no ``\\w`` word boundaries, which
    break on Chinese), longest title first so nested titles like ``唐朝`` win over ``唐``,
    and relies on ``protected_pattern`` to avoid touching existing links/code.
    """
    titles = sorted(
        [title for title in page_titles if title and title != current_title],
        key=len,
        reverse=True,
    )
    if not markdown_text or not titles:
        return markdown_text

    pattern = re.compile("|".join(_autolink_subpattern(title) for title in titles))
    protected_pattern = re.compile(r"(```.*?```|`[^`\n]+`|\[[^\]]+\]\([^)]+\))", re.DOTALL)

    def link_segment(segment: str) -> str:
        def replace(match: re.Match[str]) -> str:
            title = match.group(0)  # zero-width guards mean group 0 is exactly the title
            label = markdown_escape_link_label(title)
            return f"[**{label}**](?page={quote_plus(title)})"

        return pattern.sub(replace, segment)

    parts = protected_pattern.split(markdown_text)
    for index, part in enumerate(parts):
        if not protected_pattern.fullmatch(part):
            parts[index] = link_segment(part)
    return "".join(parts)


def parse_topic_suggestions(raw_text: str) -> list[str]:
    topics: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*[-*+\d.)]+\s*", "", line).strip()
        cleaned = cleaned.strip("`#:- ")
        if cleaned:
            topics.append(sanitize_title(cleaned))
        if len(topics) == 5:
            break
    return topics


# =============================================================================
# Chinese historical Named Entity Recognition (NER)
#
# Three interchangeable backends, tried in this order by `method="auto"`:
#   1. "bert"      - a BERT-based Chinese NER model via HuggingFace transformers
#                    (best quality; requires `pip install transformers torch`).
#   2. "llm"       - reuse the already-configured chat model to extract entities
#                    (works out of the box, no extra ML dependencies).
#   3. "gazetteer" - pure-regex dynasty/era/year matching (no network, no deps;
#                    contributes TIME entities and known dynasty names only).
# =============================================================================

# True only if transformers AND a usable backend (torch or tensorflow) are
# importable. We check without importing (find_spec) so startup stays fast and never
# pulls in torch unless NER is actually used. Requiring a backend here means the
# "bert" tier is only auto-selected when it can really run.
#
# NOTE: these are evaluated lazily via has_module()/HAS_* below so that a package
# installed *during* the session (via the Dependencies panel) is detected on the
# next rerun without restarting the app.


def has_module(name: str) -> bool:
    """Runtime check for an importable module, resilient to mid-session installs.

    Unlike a module-load-time constant, this re-queries the import system every
    call (after invalidating caches), so packages installed through the in-app
    Dependencies panel become visible on the next Streamlit rerun.
    """
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_transformers() -> bool:
    return has_module("transformers") and (has_module("torch") or has_module("tensorflow"))


# Backwards-compatible module-level snapshots. These are recomputed once per rerun
# in main() via refresh_capability_flags() so the rest of the code can keep reading
# the familiar HAS_* names while still reflecting freshly installed packages.
HAS_TRANSFORMERS = has_transformers()

# --- OCR capability detection -------------------------------------------------
# Same lazy "is the backend installed?" pattern as HAS_TRANSFORMERS. The heavy
# libraries are imported only inside the OCR functions, never at module load.
HAS_PADDLEOCR = has_module("paddleocr")
HAS_TESSERACT = has_module("pytesseract")
HAS_PIL = has_module("PIL")
HAS_PYMUPDF = has_module("fitz") or has_module("pymupdf")


def refresh_capability_flags() -> None:
    """Re-evaluate the HAS_* snapshots so mid-session installs take effect.

    Called near the top of each rerun. We invalidate importlib's finder caches
    first so a just-pip-installed package is actually discoverable.
    """
    importlib.invalidate_caches()
    global HAS_TRANSFORMERS, HAS_PADDLEOCR, HAS_TESSERACT, HAS_PIL, HAS_PYMUPDF
    HAS_TRANSFORMERS = has_transformers()
    HAS_PADDLEOCR = has_module("paddleocr")
    HAS_TESSERACT = has_module("pytesseract")
    HAS_PIL = has_module("PIL")
    HAS_PYMUPDF = has_module("fitz") or has_module("pymupdf")


# --- In-app dependency installer ----------------------------------------------
# Maps a friendly feature key to the pip requirement(s) + the import name used to
# verify success. paddlepaddle is flagged as having no wheel on Python >= 3.13.
PADDLE_HAS_WHEEL = sys.version_info < (3, 13)
DEPENDENCY_SPECS: dict[str, dict[str, object]] = {
    "pdf": {
        "label": "PDF 支援 (PyMuPDF)",
        "packages": ["pymupdf"],
        "check": "fitz",
        "help": "讓 OCR 面板可以上傳並辨識 PDF（會逐頁轉成圖片）。",
    },
    "tesseract": {
        "label": "Tesseract OCR 引擎",
        "packages": ["pytesseract"],
        "check": "pytesseract",
        "help": "離線 OCR 引擎。注意：仍需另外安裝 Tesseract 主程式與 chi_sim/chi_tra 語言包。",
    },
    "paddle": {
        "label": "PaddleOCR 引擎（中文最佳）",
        "packages": ["paddleocr", "paddlepaddle"],
        "check": "paddleocr",
        "help": "對中文辨識效果最好、完全離線。體積較大，下載需數分鐘。",
    },
    "ner": {
        "label": "本地 BERT 中文 NER (transformers + torch)",
        "packages": ["transformers", "torch"],
        "check": "transformers",
        "help": "本地實體識別模型，首次使用另會下載約 400MB 模型權重。torch 體積較大。",
    },
}


def install_packages(packages: list[str]) -> tuple[bool, str]:
    """pip-install the given packages into the running interpreter.

    Returns (success, combined_output). Uses the same Python that runs Streamlit
    (sys.executable) so the packages land in the active virtualenv.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # torch/paddle can be large; allow up to 30 min.
        )
    except subprocess.TimeoutExpired:
        return False, "安裝逾時（超過 30 分鐘）。請改用終端機手動安裝。"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"無法啟動 pip：{exc}"
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Keep the tail; pip output can be very long.
    tail = "\n".join(output.strip().splitlines()[-25:])
    return proc.returncode == 0, tail

_CJK_RANGE = "㐀-䶿一-鿿豈-﫿"
_ZH_YEAR_PATTERNS = [
    # Western-style years, optionally BCE: 公元前202年 / 公元25年 / 1912年
    re.compile(r"(?:公元前|公元|西元前|西元)?\d{1,4}\s*年"),
    # Reign-era + year in Chinese numerals: 贞观十九年 / 建安元年 / 開元二十九年
    re.compile(rf"[{_CJK_RANGE}]{{2,4}}(?:元年|[〇零一二三四五六七八九十百千两]{{1,6}}年)"),
]


def normalize_ner_label(raw_label: str) -> str | None:
    """Strip any BIO/BIOES prefix (B-/I-/E-/S-) and map to PER/LOC/TIME, else None."""
    if not raw_label:
        return None
    tag = raw_label.split("-", 1)[-1].strip().upper()
    return NER_LABEL_MAP.get(tag)


def clean_zh_token(word: str) -> str:
    """Remove subword markers and the inter-character spaces HF inserts for CJK."""
    return word.replace(" ##", "").replace("##", "").replace(" ", "").strip()


def _looks_like_entity(surface: str, norm_type: str, *, allow_short: bool = False) -> bool:
    """Reject obvious junk: empty, too short, or non-CJK noise.

    Single characters are only accepted for PER (kept surnames like 李/王) or when
    ``allow_short`` is set (e.g. curated gazetteer dynasty names such as 夏/商/周).
    """
    surface = surface.strip()
    if not surface:
        return False
    # Must contain at least one CJK character (filters stray punctuation/latin tokens).
    if not re.search(rf"[{_CJK_RANGE}]", surface):
        return False
    if len(surface) < NER_MIN_ENTITY_LEN:
        if allow_short:
            return True
        if norm_type == "PER" and surface in NER_KEEP_SINGLE:
            return True
        return False
    return True


@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_id: str):
    """Build and cache a transformers NER pipeline (one shared instance per model).

    Cached with @st.cache_resource (NOT cache_data) because the model is a large,
    unpicklable singleton that must be shared across reruns/sessions. Heavy imports
    happen here, inside the NER code path, so the app boots without torch installed.
    """
    from transformers import (  # noqa: PLC0415 - intentional lazy import
        AutoModelForTokenClassification,
        AutoTokenizer,
        BertTokenizerFast,
        pipeline,
    )

    if model_id.startswith("ckiplab/"):
        # CKIP's model card requires the base-repo fast tokenizer for correct offsets.
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    # aggregation_strategy="simple" groups B-/I- runs and exposes start/end offsets.
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU; set to 0 for the first CUDA GPU.
    )


def _postprocess_entities(raw_entities, text: str, score_threshold: float) -> list[dict[str, object]]:
    """Normalize labels, recover exact surface text from offsets, filter and dedupe."""
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, object]] = []
    for entity in raw_entities:
        score = float(entity.get("score", 1.0))
        if score < score_threshold:
            continue
        label = entity.get("entity_group") or entity.get("entity") or ""
        norm = normalize_ner_label(label)
        if norm is None:
            continue
        start, end = entity.get("start"), entity.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            surface = text[start:end].strip()
        else:
            surface = clean_zh_token(str(entity.get("word", "")))
        if not _looks_like_entity(surface, norm):
            continue
        key = (norm, surface)
        if key in seen:
            continue
        seen.add(key)
        results.append({"text": surface, "type": norm, "score": round(score, 4)})
    return results


def _extract_entities_bert(content: str, model_id: str, score_threshold: float) -> list[dict[str, object]]:
    ner = load_ner_pipeline(model_id)
    raw_entities = ner(content)
    return _postprocess_entities(raw_entities, content, score_threshold)


def _extract_entities_gazetteer(content: str) -> list[dict[str, object]]:
    """Dependency-free fallback: match known dynasties/eras and year expressions as TIME."""
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, object]] = []

    def add(surface: str, norm_type: str, *, allow_short: bool = False) -> None:
        surface = surface.strip()
        key = (norm_type, surface)
        if surface and key not in seen and _looks_like_entity(surface, norm_type, allow_short=allow_short):
            seen.add(key)
            results.append({"text": surface, "type": norm_type, "score": 1.0})

    # Collect matched dynasty names, then drop any that is a substring of another
    # matched name (so "夏朝" wins over a bare "夏"); curated names may be single-char.
    matched = {name for name, _year in DYNASTY_TIMELINE if name in content}
    for name in matched:
        if any(name != other and name in other for other in matched):
            continue
        add(name, "TIME", allow_short=True)
    for pattern in _ZH_YEAR_PATTERNS:
        for match in pattern.finditer(content):
            add(match.group(0), "TIME")
    return results


def _parse_llm_entities(raw_text: str) -> list[dict[str, object]]:
    """Parse the LLM entity response: prefer a JSON array, fall back to `TYPE: text` lines."""
    text = raw_text.strip()
    # Strip a ```json ... ``` fence if the model wrapped its answer.
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add(surface: object, label: object) -> None:
        surface = str(surface or "").strip()
        norm = normalize_ner_label(str(label or "").strip()) or "PER"
        key = (norm, surface)
        if surface and key not in seen and _looks_like_entity(surface, norm):
            seen.add(key)
            results.append({"text": surface, "type": norm, "score": 1.0})

    def add_item(item: object) -> None:
        if isinstance(item, dict):
            add(
                item.get("text") or item.get("name") or item.get("entity") or item.get("word"),
                item.get("type") or item.get("label") or item.get("category"),
            )
        elif isinstance(item, str):
            add(item, "PER")

    # Prefer a JSON value embedded anywhere in the response (array or object).
    json_match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            for item in data:
                add_item(item)
            return results
        if isinstance(data, dict):
            # Either a wrapper like {"entities": [...]} or a single entity dict.
            list_values = [value for value in data.values() if isinstance(value, list)]
            if list_values:
                for value in list_values:
                    for item in value:
                        add_item(item)
                return results
            if any(field in data for field in ("text", "name", "entity", "word")):
                add_item(data)
                return results

    # Line-based fallback: "PER: 李世民" or "李世民 (PER)" or bare "李世民".
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*[-*+\d.)]+\s*", "", line).strip().strip("`")
        if not cleaned:
            continue
        labelled = re.match(r"^\s*([A-Za-z一-鿿/]+)\s*[:：]\s*(.+)$", cleaned)
        paren = re.match(r"^(.*?)\s*[\(（]\s*([A-Za-z]+)\s*[\)）]\s*$", cleaned)
        if labelled and normalize_ner_label(labelled.group(1)):
            for piece in re.split(r"[、,，;；]", labelled.group(2)):
                add(piece, labelled.group(1))
        elif paren:
            add(paren.group(1), paren.group(2))
        else:
            add(cleaned, "PER")
    return results


def _extract_entities_llm(content: str, llm_settings: dict[str, object]) -> list[dict[str, object]]:
    """Use the configured chat model to extract PER/LOC/TIME entities as JSON."""
    system_prompt = build_system_prompt(
        "你是一個中文命名實體識別(NER)引擎。只輸出 JSON，不要任何解釋或 Markdown。"
    )
    user_prompt = (
        "從下面的中國歷史文本中擷取專有名詞實體。只保留三類："
        "PER(歷史人物)、LOC(地點/古蹟/地名)、TIME(朝代/年號/紀年)。"
        "嚴格返回一個 JSON 陣列，每個元素形如 "
        '{"text": "實體原文", "type": "PER|LOC|TIME"}。'
        "不要包含普通名詞、官職或書名，不要重複，不要輸出程式碼區塊標記。\n\n"
        f"文本：\n{content}"
    )
    raw = complete_chat(
        provider=str(llm_settings["provider"]),
        base_url=str(llm_settings["base_url"]),
        api_key=str(llm_settings["api_key"]),
        model=str(llm_settings["model"]),
        temperature=0.0,
        max_tokens=int(llm_settings.get("max_tokens", DEFAULT_MAX_TOKENS)),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return _parse_llm_entities(raw)


def extract_entities_detailed(
    content: str,
    *,
    method: str = "auto",
    model_id: str = NER_MODEL_PRIMARY,
    score_threshold: float = NER_SCORE_THRESHOLD,
    llm_settings: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Extract entities and report which backend was actually used.

    Returns ``(entities, method_used)`` where each entity is
    ``{"text": str, "type": "PER"|"LOC"|"TIME", "score": float}``.
    """
    text = (content or "").strip()
    if not text:
        return [], "none"

    chosen = method
    if chosen == "auto":
        chosen = "bert" if HAS_TRANSFORMERS else ("llm" if llm_settings else "gazetteer")

    # Try the chosen backend, then cascade BERT -> LLM -> gazetteer so EVERY caller
    # (not just the Streamlit UI) degrades gracefully when a backend is unavailable.
    if chosen == "bert":
        try:
            return _extract_entities_bert(text, model_id, score_threshold), "bert"
        except Exception:
            chosen = "llm" if llm_settings else "gazetteer"

    if chosen == "llm" and llm_settings:
        try:
            return _extract_entities_llm(text, llm_settings), "llm"
        except Exception:
            chosen = "gazetteer"

    return _extract_entities_gazetteer(text), "gazetteer"


def extract_historical_entities(content: str) -> list[str]:
    """Parse Chinese historical text and return a unique list of entity strings.

    This is the spec-facing entry point. It auto-selects the best available
    backend (BERT model if installed and runnable, else the dependency-free
    gazetteer) and degrades gracefully on failure, returning deduplicated entity
    surface forms (historical figures `[PER]`, sites `[LOC]`, and eras/dynasties
    `[TIME]`) suitable for `create_page()`.
    """
    entities, _method = extract_entities_detailed(content)
    seen: set[str] = set()
    ordered: list[str] = []
    for entity in entities:
        surface = str(entity["text"])
        if surface not in seen:
            seen.add(surface)
            ordered.append(surface)
    return ordered


def create_entity_pages(
    entities: list[dict[str, object]],
    *,
    skip_title: str | None = None,
) -> tuple[list[str], list[str]]:
    """Create a wiki page per extracted entity. Returns (created_titles, existing_titles).

    ``skip_title`` (typically the current page) is excluded so a page does not
    list itself as a freshly "created"/"existing" entity.
    """
    created: list[str] = []
    existing: list[str] = []
    handled: set[str] = set()
    skip_safe = sanitize_title(skip_title) if skip_title else None
    for entity in entities:
        raw = str(entity.get("text", "")).strip()
        safe = sanitize_title(raw)
        if not safe or safe == "Untitled" or safe in handled or safe == skip_safe:
            continue
        handled.add(safe)
        title, was_created = create_page(safe)
        (created if was_created else existing).append(title)
    return created, existing


# =============================================================================
# OCR — extract Chinese text from uploaded images / PDFs
#
# Three interchangeable engines selected in the UI:
#   1. PaddleOCR  - offline, strong on Chinese (`pip install paddleocr paddlepaddle`).
#   2. Tesseract  - offline, needs the chi_sim/chi_tra traineddata + the binary.
#   3. Vision LLM - reuses the configured chat provider's vision model (no extra deps).
# PDFs are rasterized page-by-page with PyMuPDF and each page is OCR'd in turn.
# Every heavy library is imported lazily inside its function so the app boots
# without any OCR dependency installed.
# =============================================================================


def available_ocr_engines() -> list[str]:
    """Engines we can actually offer right now, based on installed packages.

    Vision LLM is always offered (it only needs the already-required chat client).
    The two offline engines appear only when their backend import is present.
    """
    engines: list[str] = []
    if HAS_PADDLEOCR:
        engines.append(OCR_ENGINE_PADDLE)
    if HAS_TESSERACT:
        engines.append(OCR_ENGINE_TESSERACT)
    engines.append(OCR_ENGINE_VISION)
    return engines


def _pil_image_from_bytes(data: bytes):
    """Decode raw bytes into an RGB PIL image (raises if Pillow is missing)."""
    if not HAS_PIL:
        raise RuntimeError("需要 Pillow 套件才能讀取圖片。請執行 `pip install Pillow`。")
    from PIL import Image  # noqa: PLC0415 - intentional lazy import

    image = Image.open(io.BytesIO(data))
    # Normalize to RGB so every downstream engine gets a consistent mode.
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _pdf_to_images(data: bytes, *, dpi: int = 200) -> list[bytes]:
    """Rasterize each PDF page to PNG bytes via PyMuPDF for per-page OCR."""
    if not (has_module("fitz") or has_module("pymupdf")):
        raise RuntimeError(
            "需要 PyMuPDF 才能處理 PDF。請在「依賴套件」面板點一下安裝，"
            "或執行 `pip install pymupdf`，亦可改為上傳圖片。"
        )
    try:  # PyMuPDF exposes itself as `fitz`; newer versions also as `pymupdf`.
        import fitz  # noqa: PLC0415 - intentional lazy import (PyMuPDF)
    except ImportError:
        import pymupdf as fitz  # noqa: PLC0415

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[bytes] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise RuntimeError("這個 PDF 沒有任何頁面。")
            for page in doc:
                pixmap = page.get_pixmap(matrix=matrix)
                pages.append(pixmap.tobytes("png"))
    except RuntimeError:
        raise
    except Exception as exc:  # corrupt/encrypted PDF, etc.
        raise RuntimeError(f"無法讀取 PDF：{exc}") from exc
    return pages


@st.cache_resource(show_spinner=False)
def load_paddle_ocr(lang: str = OCR_PADDLE_LANG):
    """Build and cache one PaddleOCR instance (heavy; shared across reruns)."""
    from paddleocr import PaddleOCR  # noqa: PLC0415 - intentional lazy import

    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def _ocr_image_paddle(image_bytes: bytes) -> str:
    """Run PaddleOCR on one image and join recognized lines top-to-bottom."""
    import numpy as np  # noqa: PLC0415 - intentional lazy import

    ocr = load_paddle_ocr()
    image = _pil_image_from_bytes(image_bytes).convert("RGB")
    result = ocr.ocr(np.array(image), cls=True)
    lines: list[str] = []
    # PaddleOCR returns [[ [box, (text, score)], ... ]] (one list per image page).
    for block in result or []:
        for entry in block or []:
            try:
                text = entry[1][0]
            except (IndexError, TypeError):
                continue
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def _ocr_image_tesseract(image_bytes: bytes) -> str:
    """Run Tesseract on one image using the Simplified+Traditional traineddata."""
    import pytesseract  # noqa: PLC0415 - intentional lazy import

    image = _pil_image_from_bytes(image_bytes)
    text = pytesseract.image_to_string(image, lang=OCR_TESSERACT_LANG)
    return "\n".join(line for line in text.splitlines() if line.strip())


def _ocr_image_vision(image_bytes: bytes, llm_settings: dict[str, object]) -> str:
    """Transcribe one image with the configured provider's vision model."""
    system_prompt = build_system_prompt(
        "你是一個專業的繁體中文 OCR 文字辨識引擎，擅長辨識中國歷史文獻與古籍圖片。"
    )
    user_prompt = (
        "請辨識並輸出這張圖片中的所有文字。"
        "只輸出辨識到的純文字內容，保留原本的段落與換行，"
        "不要添加任何說明、標題或 Markdown 標記。"
        "若原文為簡體字，請轉換為繁體中文。"
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return complete_vision_chat(
        provider=str(llm_settings["provider"]),
        base_url=str(llm_settings["base_url"]),
        api_key=str(llm_settings["api_key"]),
        model=str(llm_settings["model"]),
        temperature=0.0,
        max_tokens=int(llm_settings.get("max_tokens", DEFAULT_MAX_TOKENS)),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_b64=image_b64,
        image_mime="image/png",
    ).strip()


def _ocr_single_image(image_bytes: bytes, engine: str, llm_settings: dict[str, object]) -> str:
    if engine == OCR_ENGINE_PADDLE:
        return _ocr_image_paddle(image_bytes)
    if engine == OCR_ENGINE_TESSERACT:
        return _ocr_image_tesseract(image_bytes)
    return _ocr_image_vision(image_bytes, llm_settings)


def run_ocr(
    *,
    file_bytes: bytes,
    file_name: str,
    engine: str,
    llm_settings: dict[str, object],
) -> str:
    """OCR an uploaded image or PDF and return the combined recognized text.

    Images are processed directly; PDFs are split into per-page PNGs (PyMuPDF) and
    each page is OCR'd then joined with a numbered separator.
    """
    if not file_bytes:
        return ""
    is_pdf = file_name.lower().endswith(".pdf")
    if is_pdf:
        page_images = _pdf_to_images(file_bytes)
        if not page_images:
            return ""
        if len(page_images) == 1:
            return _ocr_single_image(page_images[0], engine, llm_settings).strip()
        sections: list[str] = []
        for index, page_bytes in enumerate(page_images, start=1):
            page_text = _ocr_single_image(page_bytes, engine, llm_settings).strip()
            sections.append(OCR_PAGE_SEPARATOR.format(n=index).strip("\n") + "\n\n" + page_text)
        return "\n\n".join(sections).strip()
    return _ocr_single_image(file_bytes, engine, llm_settings).strip()


def _dependency_installed(spec_key: str) -> bool:
    spec = DEPENDENCY_SPECS.get(spec_key, {})
    check = str(spec.get("check") or "")
    if spec_key == "ner":
        return has_transformers()
    return has_module(check) if check else False


def render_dependency_button(spec_key: str, *, key_prefix: str = "dep") -> None:
    """One feature row: status + an Install button that pip-installs in-process."""
    spec = DEPENDENCY_SPECS.get(spec_key)
    if not spec:
        return
    label = str(spec["label"])
    packages = list(spec["packages"])  # type: ignore[arg-type]
    installed = _dependency_installed(spec_key)

    no_wheel = spec_key == "paddle" and not PADDLE_HAS_WHEEL
    col_info, col_btn = st.columns([0.62, 0.38])
    with col_info:
        status = "✅ 已安裝" if installed else "⬇️ 未安裝"
        st.markdown(f"**{label}** — {status}")
        st.caption(str(spec.get("help") or ""))
        if no_wheel and not installed:
            st.caption(
                f"⚠️ 目前的 Python {sys.version_info.major}.{sys.version_info.minor} "
                "沒有 paddlepaddle 的安裝檔，無法自動安裝。建議改用 Python 3.10–3.12，"
                "或改用 Tesseract / Vision LLM 引擎。"
            )
    with col_btn:
        disabled = installed or (no_wheel)
        btn_label = "已安裝" if installed else ("無法安裝" if no_wheel else "安裝")
        if st.button(
            btn_label,
            key=f"{key_prefix}_install_{spec_key}",
            use_container_width=True,
            disabled=disabled,
        ):
            with st.spinner(f"正在安裝 {', '.join(packages)} …（可能需要數分鐘）"):
                ok, output = install_packages(packages)
            refresh_capability_flags()
            if ok and _dependency_installed(spec_key):
                st.success(f"{label} 安裝完成！")
                rerun()
            elif ok:
                st.warning(
                    f"{', '.join(packages)} 安裝指令已完成，但尚未偵測到模組，"
                    "請重新整理頁面或重啟應用程式。"
                )
                with st.expander("安裝輸出 (log)"):
                    st.code(output or "(無輸出)")
            else:
                st.error(f"{label} 安裝失敗。")
                with st.expander("錯誤訊息 (log)", expanded=True):
                    st.code(output or "(無輸出)")


def render_dependency_panel(*, key_prefix: str = "dep") -> None:
    """Full dependencies manager: an install row per optional feature."""
    st.caption(
        "下列為選用功能套件。點「安裝」會用目前的 Python 環境 "
        f"(`{Path(sys.executable).name}`) 自動 pip 安裝。"
    )
    all_ok = all(_dependency_installed(k) for k in DEPENDENCY_SPECS)
    if all_ok:
        st.success("所有選用套件皆已安裝。")
    for spec_key in DEPENDENCY_SPECS:
        render_dependency_button(spec_key, key_prefix=key_prefix)
        st.divider()


def render_sidebar(pages: list[str], current_content: str) -> dict[str, object]:
    st.sidebar.title("中國歷史維基 📜")

    with st.sidebar.expander("模型設定 (Model Configuration)", expanded=True):
        source = st.radio(
            "模型來源 (Model source)",
            [SOURCE_LOCAL, SOURCE_ONLINE],
            horizontal=True,
            help="在本地 OpenAI 相容伺服器與線上 API 供應商之間切換。",
        )
        provider = PROVIDER_LOCAL
        if source == SOURCE_ONLINE:
            provider = st.selectbox("API 供應商 (Provider)", ONLINE_PROVIDERS)

        config = get_provider_config(provider)
        st.caption(str(config.get("help") or ""))
        if provider in (PROVIDER_NEWAPI, PROVIDER_CUSTOM):
            st.info(
                "若使用自訂的 OpenAI 相容 API，請在 Base URL 貼上供應商網址。"
                "你也可以把 New API 的 JSON 貼進 API Key 欄位，再按「儲存金鑰」。"
            )

        backend = config.get("backend")
        saved_connection = load_saved_connection(provider)
        base_url = ""
        if backend == BACKEND_OPENAI_COMPATIBLE:
            default_base_url = saved_connection.get("base_url") or str(config.get("base_url") or DEFAULT_BASE_URL)
            base_url_widget_key = f"base_url_{provider}"
            saved_version_key = f"saved_conn_version_{provider}"
            current_saved_version = saved_connection_version(provider)
            if (
                base_url_widget_key not in st.session_state
                or st.session_state.get(saved_version_key) != current_saved_version
            ):
                st.session_state[base_url_widget_key] = default_base_url
                st.session_state[saved_version_key] = current_saved_version
            base_url = st.text_input(
                "Base URL",
                key=base_url_widget_key,
                help=(
                    "請使用供應商根網址或 OpenAI 相容的 /v1 網址，"
                    "不要包含 /chat/completions 之類的端點路徑。"
                ),
            )
        else:
            st.caption("Claude 使用 Anthropic 原生 Messages API，因此不需要 Base URL。")

        env_name = str(config.get("api_key_env") or "")
        key_help = "本地伺服器通常會忽略這個值，但 OpenAI SDK 要求非空字串。"
        if env_name:
            key_help = (
                f"在此貼上金鑰、儲存到 {api_key_path(provider)}，"
                f"或在啟動 Streamlit 前設定 {env_name} 環境變數。"
            )
        elif provider in (PROVIDER_NEWAPI, PROVIDER_CUSTOM):
            key_help = (
                "貼上原始 API 金鑰，或像這樣的 New API JSON 物件："
                '{"_type":"newapi_channel_conn","key":"sk-...","url":"https://www.juaiapi.com"}。'
            )
        saved_api_key = saved_connection.get("api_key", "")
        default_api_key = saved_api_key or str(config.get("api_key") or "")
        api_key_widget_key = f"api_key_{provider}"
        if api_key_widget_key not in st.session_state:
            st.session_state[api_key_widget_key] = default_api_key
        api_key = st.text_input(
            "API Key",
            type="password",
            key=api_key_widget_key,
            help=key_help,
        )
        key_status = f"金鑰檔案：`{api_key_path(provider)}`"
        if saved_api_key:
            st.caption(f"{key_status} 已為此供應商載入。")
        else:
            st.caption(f"此供應商尚無已儲存的金鑰。{key_status}")

        col_save_key, col_delete_key = st.columns(2)
        if col_save_key.button("儲存金鑰 (Save Key)", key=f"save_key_{provider}", use_container_width=True):
            try:
                parsed_connection = parse_api_key_input(api_key)
                parsed_api_key = parsed_connection.get("api_key", "")
                parsed_base_url = parsed_connection.get("base_url", "")
                save_api_key(provider, parsed_api_key, parsed_base_url or base_url)
                if parsed_base_url and backend == BACKEND_OPENAI_COMPATIBLE:
                    st.session_state[f"base_url_{provider}"] = normalize_base_url(parsed_base_url)
                    st.session_state[api_key_widget_key] = parsed_api_key
                st.success(f"已儲存 {provider} 的 API 金鑰。")
                rerun()
            except ValueError as exc:
                st.warning(str(exc))
        if col_delete_key.button(
            "刪除金鑰 (Delete Key)",
            key=f"delete_key_{provider}",
            disabled=not saved_api_key,
            use_container_width=True,
        ):
            if delete_saved_api_key(provider):
                st.session_state[api_key_widget_key] = str(config.get("api_key") or "")
                if backend == BACKEND_OPENAI_COMPATIBLE:
                    st.session_state[f"base_url_{provider}"] = str(config.get("base_url") or DEFAULT_BASE_URL)
                st.success(f"已刪除 {provider} 的已儲存 API 金鑰。")
                rerun()
        if backend == BACKEND_OPENAI_COMPATIBLE and st.button(
            "測試連線 (Test Provider)",
            key=f"test_provider_{provider}",
            use_container_width=True,
        ):
            try:
                parsed_connection = parse_api_key_input(api_key)
                test_api_key = parsed_connection.get("api_key", api_key)
                test_base_url = parsed_connection.get("base_url", base_url)
                model_ids = test_openai_compatible_connection(provider, test_base_url, test_api_key)
                chat_models = [model_id for model_id in model_ids if not is_embedding_model(model_id)]
                if chat_models:
                    preview = ", ".join(chat_models[:8])
                    st.success(f"連線成功。可用的對話模型包括：{preview}")
                else:
                    st.warning("已連線，但 /v1/models 未返回任何非嵌入模型。")
            except Exception as exc:
                st.error(format_llm_error(exc, provider, base_url))
        model = st.text_input(
            "模型 (Model)",
            value=str(config.get("model") or ""),
            key=f"model_{provider}",
            help=(
                "請使用所選供應商提供的對話模型 ID。"
                "若為本地／自訂的 OpenAI 相容供應商，可留空以嘗試 /v1/models 返回的模型。"
            ),
        )
        temperature = st.slider("溫度 (Temperature)", 0.0, 2.0, 0.7, 0.05)
        max_tokens = st.number_input(
            "最大輸出 Token 數 (Max output tokens)",
            min_value=256,
            max_value=32768,
            value=DEFAULT_MAX_TOKENS,
            step=256,
            help="Claude 會直接使用此值；其他需要輸出上限的供應商也會沿用此設定。",
        )

    with st.sidebar.expander("提示詞設定 (Prompt Setting)", expanded=False):
        prompts = scan_prompts()
        active_prompt = get_active_prompt_name()
        selected_prompt = st.selectbox(
            "使用中的提示詞 (Active prompt)",
            prompts,
            index=prompts.index(active_prompt),
            help="每次 AI 維基任務前都會先讀取這個提示詞。",
        )
        if selected_prompt != active_prompt:
            st.session_state.active_prompt = selected_prompt
            rerun()
        st.caption(f"使用中：`prmopt/{selected_prompt}.md`")

    with st.sidebar.expander("實體識別 NER 設定", expanded=False):
        if HAS_TRANSFORMERS:
            st.caption("已偵測到 transformers，使用本地 BERT 中文 NER 模型。")
            model_choices = [NER_MODEL_PRIMARY, NER_MODEL_FALLBACK]
            ner_model = st.selectbox(
                "NER 模型 (NER model)",
                model_choices,
                index=0,
                help="首選 Apache-2.0 模型；CKIP 模型實體更細但為 GPL-3.0。首次使用會下載約 400MB。",
            )
            st.session_state["ner_model_id"] = ner_model
            threshold = st.slider(
                "分數門檻 (Score threshold)",
                0.0,
                1.0,
                float(st.session_state.get("ner_threshold", NER_SCORE_THRESHOLD)),
                0.05,
                help="低=召回更多（更雜），高=更精確。建議先用 0.5 在你的文本上試調。",
            )
            st.session_state["ner_threshold"] = threshold
        else:
            st.info(
                "未安裝 `transformers`/`torch`。實體擷取將使用目前的 LLM 模型，"
                "或在無 API 時使用內建朝代詞典。"
            )
            st.caption("可在下方「依賴套件」面板一鍵安裝本地 BERT NER。")
            render_dependency_button("ner", key_prefix="sidebar_ner")
            st.session_state.setdefault("ner_model_id", NER_MODEL_PRIMARY)
            st.session_state.setdefault("ner_threshold", NER_SCORE_THRESHOLD)

    with st.sidebar.expander("依賴套件 (Dependencies)", expanded=False):
        render_dependency_panel(key_prefix="sidebar")

    if st.sidebar.button("建立新頁面 (Create New Page)", use_container_width=True):
        st.session_state.show_new_page = not st.session_state.show_new_page

    if st.session_state.show_new_page:
        with st.sidebar.form("new_page_form", clear_on_submit=True):
            new_title = st.text_input("新頁面標題 (New page title)")
            submitted = st.form_submit_button("建立頁面 (Create Page)", use_container_width=True)
            if submitted:
                safe_title, created = create_page(new_title)
                st.session_state.show_new_page = False
                if not created:
                    st.sidebar.warning(f"`{safe_title}` 已存在，將直接開啟。")
                select_page(safe_title)

    st.sidebar.divider()
    st.sidebar.subheader("頁面 (Pages)")
    if not pages:
        st.sidebar.caption("尚無頁面。請先建立一個頁面開始使用。")
    else:
        chronological = st.sidebar.toggle(
            "按年代排序 (Sort by year)",
            value=st.session_state.get("sidebar_chronological", False),
            key="sidebar_chronological",
            help="按 frontmatter 中的 year 欄位從早到晚排列頁面。",
        )
        ordered_pages = pages
        if chronological:
            entries = collect_timeline_entries(pages)
            # Earliest first; undated pages (year is None) sink to the bottom.
            entries.sort(key=lambda e: (e["year"] is None, e["year"] if e["year"] is not None else 0))
            ordered_pages = [str(entry["title"]) for entry in entries]
        for title in ordered_pages:
            marker = "* " if title == st.session_state.current_page else "- "
            if st.sidebar.button(f"{marker}{title}", key=f"page_{title}", use_container_width=True):
                select_page(title)

    st.sidebar.divider()
    with st.sidebar.expander("AI 助手 (AI Agent)", expanded=True):
        current_page = st.session_state.current_page
        if not current_page:
            st.caption("建立或選擇一個頁面以使用 AI 工具。")
        else:
            parsed_current_connection = parse_api_key_input(api_key)
            request_api_key = parsed_current_connection.get("api_key", api_key)
            request_base_url = parsed_current_connection.get("base_url", base_url)
            draft_prompt = st.text_area(
                "草稿提示 (Draft prompt)",
                value=f"請撰寫一篇關於「{current_page}」的實用且結構清晰的維基頁面。",
                height=110,
            )
            is_empty = not current_content.strip()
            if not is_empty:
                st.caption("只有空白頁面才能產生草稿。")
            if st.button("產生草稿 (Generate Draft)", disabled=not is_empty, use_container_width=True):
                generate_draft(
                    title=current_page,
                    prompt=draft_prompt,
                    provider=provider,
                    base_url=request_base_url,
                    api_key=request_api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=int(max_tokens),
                    pages=pages,
                )

    parsed_return_connection = parse_api_key_input(api_key)
    return {
        "provider": provider,
        "base_url": parsed_return_connection.get("base_url", base_url),
        "api_key": parsed_return_connection.get("api_key", api_key),
        "model": model,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
    }


def infer_page_metadata_llm(
    *,
    title: str,
    body: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> dict[str, object]:
    """Best-effort: ask the model for {dynasty, year, tags} frontmatter. Returns {} on failure."""
    system_prompt = build_system_prompt(
        "你是一個歷史元資料擷取器。只輸出 JSON，不要解釋或 Markdown 程式碼區塊。"
    )
    user_prompt = (
        f"根據以下歷史維基頁面，判斷它最相關的朝代、起始公元年份(公元前用負數)和2-5個標籤。"
        f'嚴格返回 JSON：{{"dynasty": "朝代名", "year": 數字或null, "tags": ["標籤1","標籤2"]}}。\n\n'
        f"標題：{title}\n\n正文：\n{body[:2000]}"
    )
    try:
        raw = complete_chat(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=min(int(max_tokens), 512),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()
    except Exception:
        return {}

    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    meta: dict[str, object] = {}
    dynasty = data.get("dynasty")
    if isinstance(dynasty, str) and dynasty.strip():
        meta["dynasty"] = dynasty.strip()
    year = _coerce_year(data.get("year"))
    if year is not None:
        meta["year"] = year
    tags = data.get("tags")
    if isinstance(tags, list):
        clean_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if clean_tags:
            meta["tags"] = clean_tags[:6]
    return meta


def generate_draft(
    *,
    title: str,
    prompt: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    pages: list[str],
) -> None:
    st.subheader("正在產生草稿 (Generating Draft)")
    placeholder = st.empty()
    content = ""
    system_prompt = build_system_prompt(
        "你是一個嚴謹的維基撰寫助手。只返回純 Markdown，並一律使用繁體中文。"
        "不要把整篇答案包在程式碼區塊裡。"
    )
    user_prompt = (
        f"請撰寫一個標題為 `{title}` 的 Markdown 維基頁面。\n\n"
        f"使用者要求：\n{prompt}\n\n"
        "使用清晰的標題、簡潔的說明與具體的細節，並以繁體中文撰寫。"
    )
    try:
        for token in stream_chat(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ):
            content += token
            placeholder.markdown(autolink_markdown(content, pages, title))
        body = content.strip()
        # Enrich frontmatter (dynasty/year/tags) via the model; falls back to the
        # gazetteer-based defaults inside write_page() if this returns nothing.
        meta = infer_page_metadata_llm(
            title=title,
            body=body,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )
        write_page(title, body + "\n", meta=meta or None)
        st.session_state.edit_mode = False
        st.success("草稿已儲存。")
        rerun()
    except Exception as exc:
        st.error(format_llm_error(exc, provider, base_url))


def prepend_summary(
    *,
    title: str,
    content: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    if not content.strip():
        st.warning("此頁面為空。請先產生或撰寫內容。")
        return

    system_prompt = build_system_prompt("你是一個簡潔的維基編輯。只返回 Markdown，並使用繁體中文。")
    user_prompt = (
        f"請用一段簡短的文字加上 3-5 個重點條列，摘要這個標題為 `{title}` 的維基頁面。\n\n"
        f"{content}"
    )
    try:
        with st.spinner("正在摘要頁面 (Summarizing)..."):
            summary = complete_chat(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ).strip()
        new_content = f"## AI 摘要\n\n{summary}\n\n---\n\n{content.lstrip()}"
        write_page(title, new_content)
        st.success("摘要已加到頁面頂部。")
        rerun()
    except Exception as exc:
        st.error(format_llm_error(exc, provider, base_url))


def suggest_related_topics(
    *,
    title: str,
    content: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    if not content.strip():
        st.warning("此頁面為空。請先產生或撰寫內容。")
        return

    system_prompt = build_system_prompt(
        "你負責建議簡潔的維基頁面標題。只返回純文字清單，每行一個標題，並使用繁體中文。"
    )
    user_prompt = (
        "根據這個頁面，建議 3-5 個接下來值得建立的相關維基頁面。"
        "使用簡短的標題式詞語，不要加任何說明。\n\n"
        f"目前頁面：{title}\n\n{content}"
    )
    try:
        with st.spinner("正在尋找相關主題 (Finding related topics)..."):
            raw = complete_chat(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        st.session_state.related_topics = parse_topic_suggestions(raw)
    except Exception as exc:
        st.error(format_llm_error(exc, provider, base_url))


def render_related_topics(existing_pages: list[str]) -> None:
    topics = st.session_state.get("related_topics", [])
    if not topics:
        return

    st.subheader("建議的相關主題 (Suggested Related Topics)")
    for topic in topics:
        col_title, col_action = st.columns([0.72, 0.28])
        exists = topic in existing_pages
        col_title.markdown(f"- **{topic}**" + (" *(已存在)*" if exists else ""))
        if col_action.button("開啟" if exists else "建立", key=f"topic_{topic}"):
            if not exists:
                create_page(topic)
            select_page(topic)


def extract_and_create_entities(
    *,
    title: str,
    content: str,
    llm_settings: dict[str, object],
) -> None:
    """Run NER on the page, auto-create a page per entity, and stash results for display."""
    if not content.strip():
        st.warning("此頁面為空。請先產生或撰寫內容。")
        return

    # method="auto" picks bert/llm/gazetteer and cascades through them internally,
    # so we get the full three-tier fallback (not just bert -> gazetteer).
    try:
        with st.spinner("正在辨識歷史實體 (Extracting historical entities)..."):
            entities, used = extract_entities_detailed(
                content,
                method="auto",
                model_id=str(st.session_state.get("ner_model_id", NER_MODEL_PRIMARY)),
                score_threshold=float(st.session_state.get("ner_threshold", NER_SCORE_THRESHOLD)),
                llm_settings=llm_settings,
            )
    except Exception as exc:  # Last-resort guard; cascade above already handles most failures.
        st.warning(f"主要 NER 引擎失敗，已回退到詞典比對。Primary NER failed ({exc}); using gazetteer.")
        entities, used = extract_entities_detailed(content, method="gazetteer")

    # Per the spec: automatically create a page for every extracted entity
    # (excluding this page itself).
    created, existing = create_entity_pages(entities, skip_title=title)
    st.session_state.extracted_entities = entities
    st.session_state.extracted_entities_page = title
    st.session_state.ner_method = used

    if not entities:
        st.info("未辨識到實體。No historical entities were detected on this page.")
        return
    st.success(
        f"辨識到 {len(entities)} 個實體 (引擎: {used})。"
        f"新建 {len(created)} 頁，已存在 {len(existing)} 頁。"
    )
    rerun()


def render_extracted_entities(existing_pages: list[str]) -> None:
    """Show the most recent NER results for the current page, grouped by type."""
    if st.session_state.get("extracted_entities_page") != st.session_state.get("current_page"):
        return
    entities = st.session_state.get("extracted_entities", [])
    if not entities:
        return

    used = st.session_state.get("ner_method")
    st.subheader("辨識出的歷史實體 (Extracted Entities)")
    if used:
        st.caption(f"NER 引擎: `{used}`")
    grouped: dict[str, list[dict[str, object]]] = {"PER": [], "LOC": [], "TIME": []}
    for entity in entities:
        grouped.setdefault(str(entity.get("type", "PER")), []).append(entity)

    columns = st.columns(3)
    for column, type_key in zip(columns, ("PER", "LOC", "TIME")):
        with column:
            st.markdown(f"**{NER_TYPE_LABELS.get(type_key, type_key)}**")
            items = grouped.get(type_key, [])
            if not items:
                st.caption("—")
            for entity in items:
                surface = str(entity.get("text", ""))
                safe = sanitize_title(surface)
                exists = safe in existing_pages
                if st.button(
                    f"{'📄' if exists else '➕'} {surface}",
                    key=f"entity_{type_key}_{safe}",
                    use_container_width=True,
                ):
                    if not exists:
                        create_page(safe)
                    select_page(safe)
    if st.button("清除實體列表 (Clear)", key="clear_entities"):
        st.session_state.extracted_entities = []
        st.session_state.extracted_entities_page = None
        rerun()


def render_delete_page(pages: list[str], current_page: str) -> None:
    with st.expander("刪除頁面 (Delete Page)"):
        st.warning("這會永久刪除此頁面的 Markdown 檔案。")
        confirmation = st.text_input(
            "輸入頁面標題以確認刪除",
            key=f"delete_confirm_{current_page}",
            placeholder=current_page,
        )
        if st.button(
            "刪除此頁面 (Delete This Page)",
            type="primary",
            disabled=confirmation != current_page,
            use_container_width=True,
        ):
            deleted = delete_page(current_page)
            remaining_pages = [page for page in scan_pages() if page != current_page]
            st.session_state.current_page = remaining_pages[0] if remaining_pages else None
            st.session_state.edit_mode = False
            st.session_state.related_topics = []
            if st.session_state.current_page:
                set_query_page(st.session_state.current_page)
            else:
                clear_query_page()
            if deleted:
                st.success(f"已刪除 `{current_page}`。")
            else:
                st.warning(f"`{current_page}` 已不存在。")
            rerun()


def render_prompt_settings() -> None:
    st.header("提示詞設定 (Prompt Setting)")
    st.caption("提示詞以 Markdown 檔案儲存在 `prmopt/`，每次 AI 維基任務前都會讀取。")

    prompts = scan_prompts()
    active_prompt = get_active_prompt_name()
    selected_prompt = st.selectbox(
        "要使用的提示詞 (Prompt to use)",
        prompts,
        index=prompts.index(active_prompt),
        key="prompt_settings_selector",
    )
    if selected_prompt != active_prompt:
        st.session_state.active_prompt = selected_prompt
        rerun()

    prompt_content = read_prompt(selected_prompt)
    edited_prompt = st.text_area(
        "提示詞 Markdown",
        value=prompt_content,
        height=520,
        key=f"prompt_editor_{selected_prompt}",
        help="在 AI 收到具體的草稿／摘要／主題任務前，先教它如何維護這個維基。",
    )

    col_save, col_reset, col_delete = st.columns(3)
    if col_save.button("儲存提示詞 (Save)", type="primary", use_container_width=True):
        saved_name = write_prompt(selected_prompt, edited_prompt)
        st.session_state.active_prompt = saved_name
        st.success(f"已儲存 `prmopt/{saved_name}.md`。")
        rerun()
    if col_reset.button(
        "還原預設 (Reset Default)",
        disabled=selected_prompt != DEFAULT_PROMPT_NAME,
        use_container_width=True,
    ):
        write_prompt(DEFAULT_PROMPT_NAME, DEFAULT_WIKI_PROMPT)
        st.success("已還原預設提示詞。")
        rerun()
    if col_delete.button(
        "刪除提示詞 (Delete)",
        disabled=selected_prompt == DEFAULT_PROMPT_NAME,
        use_container_width=True,
    ):
        if delete_prompt(selected_prompt):
            st.session_state.active_prompt = DEFAULT_PROMPT_NAME
            st.success(f"已刪除 `prmopt/{selected_prompt}.md`。")
            rerun()

    st.divider()
    st.subheader("建立提示詞 (Create Prompt)")
    with st.form("new_prompt_form", clear_on_submit=True):
        new_prompt_name = st.text_input("新提示詞名稱 (New prompt name)", placeholder="research_wiki_maintainer")
        new_prompt_seed = st.text_area(
            "初始提示詞 (Initial prompt)",
            value=DEFAULT_WIKI_PROMPT,
            height=220,
        )
        submitted = st.form_submit_button("建立並使用 (Create and Use)", use_container_width=True)
        if submitted:
            safe_name, created = create_prompt(new_prompt_name, new_prompt_seed)
            st.session_state.active_prompt = safe_name
            if not created:
                st.warning(f"`prmopt/{safe_name}.md` 已存在，已切換至該提示詞。")
            else:
                st.success(f"已建立 `prmopt/{safe_name}.md`。")
            rerun()


def format_year(year: object) -> str:
    """Render an integer year as a human-readable CE/BCE string (negative = BCE)."""
    parsed = _coerce_year(year)
    if parsed is None:
        return "未知 (unknown)"
    if parsed < 0:
        return f"公元前 {abs(parsed)} 年 (BCE)"
    return f"公元 {parsed} 年 (CE)"


def collect_timeline_entries(pages: list[str]) -> list[dict[str, object]]:
    """Read frontmatter for every page and return entries enriched with a sortable year."""
    entries: list[dict[str, object]] = []
    for title in pages:
        meta = read_page_meta(title)
        year = _coerce_year(meta.get("year"))
        dynasty = meta.get("dynasty") or ""
        if year is None:
            # Backfill from the gazetteer using title + dynasty so undated pages still place.
            inferred_name, inferred_year = infer_dynasty_year(str(dynasty), title)
            year = inferred_year
            if not dynasty and inferred_name:
                dynasty = inferred_name
        tags = meta.get("tags")
        entries.append(
            {
                "title": title,
                "year": year,
                "dynasty": dynasty,
                "tags": tags if isinstance(tags, (list, tuple)) else [],
            }
        )
    return entries


def render_timeline_view(pages: list[str]) -> None:
    """Chronologically sorted timeline of all pages, parsed from YAML frontmatter."""
    st.header("歷史時間線 (Timeline)")
    st.caption("按頁面 frontmatter 中的 `year` 欄位排序。未標註年份的頁面列在末尾。")

    if not pages:
        st.info("還沒有頁面。請先建立並產生一些歷史頁面。")
        return

    entries = collect_timeline_entries(pages)
    dated = [entry for entry in entries if entry["year"] is not None]
    undated = [entry for entry in entries if entry["year"] is None]
    dated.sort(key=lambda entry: int(entry["year"]))  # ascending: earliest first

    if dated:
        # Compact dataframe overview for quick scanning.
        try:
            st.dataframe(
                [
                    {
                        "年份 Year": format_year(entry["year"]),
                        "朝代 Dynasty": entry["dynasty"] or "—",
                        "頁面 Page": entry["title"],
                    }
                    for entry in dated
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception:
            pass

        st.divider()
        last_dynasty = None
        for entry in dated:
            dynasty = str(entry["dynasty"] or "")
            if dynasty and dynasty != last_dynasty:
                st.markdown(f"### {dynasty}")
                last_dynasty = dynasty
            col_year, col_page = st.columns([0.3, 0.7])
            col_year.markdown(f"**{format_year(entry['year'])}**")
            if col_page.button(entry["title"], key=f"timeline_{entry['title']}", use_container_width=True):
                select_page(entry["title"])
            tags = entry["tags"]
            if tags:
                col_page.caption(" ".join(f"`{tag}`" for tag in tags))

    if undated:
        st.divider()
        st.subheader("未標註年份 (Undated)")
        for entry in undated:
            if st.button(entry["title"], key=f"timeline_undated_{entry['title']}", use_container_width=True):
                select_page(entry["title"])


def render_page_metadata(meta: dict[str, object]) -> None:
    """Render frontmatter (dynasty / year / tags) as compact captions above a page."""
    if not meta:
        return
    bits: list[str] = []
    dynasty = meta.get("dynasty")
    if dynasty:
        bits.append(f"朝代 (Dynasty): **{dynasty}**")
    year = meta.get("year")
    if year not in (None, ""):
        bits.append(f"年份 (Year): **{format_year(year)}**")
    if bits:
        st.caption(" · ".join(bits))
    tags = meta.get("tags")
    if isinstance(tags, (list, tuple)) and tags:
        st.caption("標籤 (Tags): " + " ".join(f"`{tag}`" for tag in tags))


def render_ocr_panel(current_page: str, pages: list[str], llm_settings: dict[str, object]) -> None:
    """OCR uploader + editable preview, embedded in the wiki page area.

    Lets the user upload an image/PDF, pick an engine, recognize the text, edit it,
    then append it to the current page or create a brand-new page from it.
    """
    with st.expander("🖼️ OCR 圖片文字辨識 (Image / PDF OCR)", expanded=False):
        engines = available_ocr_engines()
        notes: list[str] = []
        if not HAS_PADDLEOCR:
            notes.append("未安裝 PaddleOCR")
        if not HAS_TESSERACT:
            notes.append("未安裝 Tesseract")
        if not HAS_PYMUPDF:
            notes.append("未安裝 PyMuPDF（PDF 需要）")
        if notes:
            st.caption("提示：" + "；".join(notes) + "。可在側邊欄「依賴套件」面板一鍵安裝。")

        engine = st.selectbox(
            "辨識引擎 (OCR engine)",
            engines,
            key="ocr_engine_select",
            help="本地引擎完全離線；Vision LLM 會使用左側設定的視覺模型。",
        )

        # Always allow selecting a PDF so the user is never blocked at the picker.
        # If PyMuPDF is missing we surface a one-click install instead of silently
        # dropping the file type (the previous behaviour that made PDFs un-selectable).
        uploaded = st.file_uploader(
            "上傳圖片或 PDF (Upload image / PDF)",
            type=OCR_UPLOAD_TYPES,
            key="ocr_uploader",
            help="支援 PNG / JPG / WEBP / BMP / TIFF / PDF。",
        )

        is_pdf = uploaded is not None and uploaded.name.lower().endswith(".pdf")
        if is_pdf and not HAS_PYMUPDF:
            st.warning("偵測到 PDF，但尚未安裝 PyMuPDF。請先安裝以啟用 PDF 辨識：")
            render_dependency_button("pdf", key_prefix="ocr_pdf")

        if uploaded is not None and HAS_PIL and not is_pdf:
            try:
                st.image(uploaded.getvalue(), caption=uploaded.name, use_container_width=True)
            except Exception:
                pass

        run_disabled = uploaded is None or (is_pdf and not HAS_PYMUPDF)
        if st.button("開始辨識 (Run OCR)", type="primary", use_container_width=True, disabled=run_disabled):
            if engine == OCR_ENGINE_VISION and not (llm_settings.get("api_key") or llm_settings.get("provider")):
                st.warning("Vision LLM 需要可用的模型設定，請先在左側設定供應商與 API 金鑰。")
            else:
                try:
                    with st.spinner("正在辨識文字 (Running OCR)..."):
                        text = run_ocr(
                            file_bytes=uploaded.getvalue(),
                            file_name=uploaded.name,
                            engine=engine,
                            llm_settings=llm_settings,
                        )
                    st.session_state.ocr_text = text
                    st.session_state.ocr_source_name = uploaded.name
                    # Bump the nonce so the text area re-reads the new value (a keyed
                    # widget otherwise ignores the `value=` argument on later runs).
                    st.session_state.ocr_nonce = st.session_state.get("ocr_nonce", 0) + 1
                    rerun()
                except Exception as exc:
                    st.error(f"OCR 失敗：{exc}")

        nonce = st.session_state.get("ocr_nonce", 0)
        ocr_text = st.text_area(
            "辨識結果（可編輯）(Recognized text — editable)",
            value=st.session_state.get("ocr_text", ""),
            height=260,
            key=f"ocr_text_area_{nonce}",
        )

        col_append, col_new, col_clear = st.columns(3)
        if col_append.button(
            "附加到本頁 (Append to page)",
            use_container_width=True,
            disabled=not ocr_text.strip(),
        ):
            existing_raw = read_page_raw(current_page)
            _, existing_body = split_frontmatter(existing_raw)
            separator = "\n\n" if existing_body.strip() else ""
            new_body = f"{existing_body.rstrip()}{separator}{ocr_text.strip()}\n"
            write_page(current_page, new_body)
            st.success(f"已將辨識文字附加到 `{current_page}`。")
            st.session_state.ocr_text = ""
            st.session_state.ocr_source_name = None
            st.session_state.ocr_nonce = nonce + 1
            rerun()

        default_new_title = ""
        source_name = st.session_state.get("ocr_source_name")
        if source_name:
            default_new_title = sanitize_title(Path(str(source_name)).stem)
        new_title = col_new.text_input(
            "新頁面標題 (New page title)",
            value=default_new_title,
            key=f"ocr_new_title_{nonce}",
            label_visibility="collapsed",
            placeholder="新頁面標題",
        )
        if col_new.button(
            "建立新頁面 (Create page)",
            use_container_width=True,
            disabled=not ocr_text.strip() or not new_title.strip(),
        ):
            safe_title, created = create_page(new_title)
            write_page(safe_title, ocr_text.strip() + "\n")
            st.session_state.ocr_text = ""
            st.session_state.ocr_source_name = None
            st.session_state.ocr_nonce = nonce + 1
            if not created:
                st.warning(f"`{safe_title}` 已存在，已寫入其內容並開啟。")
            select_page(safe_title)

        if col_clear.button("清除 (Clear)", use_container_width=True):
            st.session_state.ocr_text = ""
            st.session_state.ocr_source_name = None
            st.session_state.ocr_nonce = nonce + 1
            rerun()


def render_wiki_page(pages: list[str], llm_settings: dict[str, object]) -> None:
    current_page = st.session_state.current_page

    if not current_page:
        st.info("從側邊欄建立一個頁面，開始你的本地中國歷史維基。Create a page from the sidebar to begin.")
        return

    raw_text = read_page_raw(current_page)
    meta, content = split_frontmatter(raw_text)
    st.caption(f"`wiki_data/{current_page}.md`")
    st.header(current_page)
    render_page_metadata(meta)

    st.session_state.edit_mode = st.toggle("編輯頁面 (Edit Page)", value=st.session_state.edit_mode)

    if st.session_state.edit_mode:
        st.caption("提示：檔案頂部的 YAML frontmatter（--- 之間）可在此直接編輯。")
        edited = st.text_area(
            "Markdown 原始碼（含 frontmatter）",
            value=raw_text,
            height=520,
            key=f"editor_{current_page}",
        )
        col_save, col_cancel = st.columns(2)
        if col_save.button("儲存變更 (Save Changes)", type="primary", use_container_width=True):
            # The editor shows the full file; persist verbatim without re-merging
            # on-disk frontmatter so manual frontmatter edits are honoured.
            write_page(current_page, edited, merge_existing=False)
            st.session_state.edit_mode = False
            st.success("頁面已儲存。")
            rerun()
        if col_cancel.button("取消 (Cancel)", use_container_width=True):
            st.session_state.edit_mode = False
            rerun()
        return

    col_summary, col_topics, col_entities = st.columns(3)
    if col_summary.button("摘要本頁 (Summarize)", use_container_width=True):
        prepend_summary(title=current_page, content=content, **llm_settings)
    if col_topics.button("建議相關主題 (Related Topics)", use_container_width=True):
        suggest_related_topics(title=current_page, content=content, **llm_settings)
    entity_help = (
        "使用 BERT 中文 NER 模型擷取實體。" if HAS_TRANSFORMERS
        else "未安裝 transformers，將改用 LLM 擷取實體。"
    )
    if col_entities.button(
        "擷取並建立實體頁面 (Extract Entities)",
        use_container_width=True,
        help=entity_help,
    ):
        extract_and_create_entities(title=current_page, content=content, llm_settings=llm_settings)

    render_ocr_panel(current_page, pages, llm_settings)
    render_related_topics(pages)
    render_extracted_entities(pages)
    st.divider()

    if content.strip():
        st.markdown(autolink_markdown(content, pages, current_page))
    else:
        st.info("此頁面為空。請使用側邊欄的 AI Agent 產生草稿，或開啟「編輯頁面」手動撰寫，亦可使用上方的 OCR 從圖片匯入文字。")

    st.divider()
    render_delete_page(pages, current_page)


# =============================================================================
# Relationship Graph — an interactive knowledge network of the whole wiki
#
# Two complementary kinds of edges, recomputed from the .md files on every render
# so new/edited/deleted pages are always reflected:
#   * PAGE -> OBJECT : every page links to its dynasty, its tags, and (optionally,
#     via NER/LLM) the historical people/places/times it mentions. Two pages that
#     share an object are therefore connected *through* that object node — this is
#     the "what object relates these two wikis" view.
#   * PAGE -> PAGE   : a direct edge when one page's body mentions another page's
#     title (the same matching used by the in-text auto-linker).
#
# Rendered with vis-network (vis.js) loaded from a CDN inside an st.components
# iframe, so NO extra Python package is required — it works even on a minimal /
# locked-down Python where paddle/torch wheels are unavailable.
# =============================================================================

# Node "kind" -> display colour (community colouring, like the reference image).
GRAPH_KIND_COLORS = {
    "page": "#4F9DFF",      # wiki pages — bright blue hubs
    "dynasty": "#F4B740",   # dynasties/eras — amber
    "tag": "#9B8CFF",       # tags — violet
    "PER": "#FF6B8B",       # people — pink/red
    "LOC": "#3FC68A",       # places — green
    "TIME": "#F4B740",      # times — amber (shared with dynasty family)
}
GRAPH_KIND_LABELS = {
    "page": "維基頁面 (Page)",
    "dynasty": "朝代 (Dynasty)",
    "tag": "標籤 (Tag)",
    "PER": "歷史人物 (Person)",
    "LOC": "地點 (Location)",
    "TIME": "時間/年號 (Time)",
}


def _page_mention_edges(pages: list[str], bodies: dict[str, str]) -> list[tuple[str, str]]:
    """Direct page->page edges: page A mentions page B's title in its body.

    Reuses the auto-linker's exact-match sub-patterns (longest title first, with
    ASCII edge guards) so Chinese titles match without word boundaries.
    """
    edges: list[tuple[str, str]] = []
    titles = sorted([t for t in pages if t], key=len, reverse=True)
    if len(titles) < 2:
        return edges
    patterns = {t: re.compile(_autolink_subpattern(t)) for t in titles}
    for source in titles:
        body = bodies.get(source, "")
        if not body:
            continue
        for target in titles:
            if target == source:
                continue
            if patterns[target].search(body):
                edges.append((source, target))
    return edges


def build_relationship_graph(
    pages: list[str],
    *,
    include_dynasty: bool = True,
    include_tags: bool = True,
    include_mentions: bool = True,
    entity_map: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Build a {nodes, edges, stats} graph dict from the wiki's Markdown files.

    Pure and dependency-free: reads frontmatter + body for every page and derives
    nodes/edges deterministically. ``entity_map`` (optional) injects NER/LLM entity
    object-nodes (PER/LOC/TIME) keyed by page title for the AI-enriched view.
    """
    nodes: dict[str, dict[str, object]] = {}
    edges: dict[tuple[str, str], dict[str, object]] = {}
    degree: dict[str, int] = {}

    def node_id(kind: str, name: str) -> str:
        return f"{kind}::{name}"

    def add_node(kind: str, name: str, **extra) -> str:
        nid = node_id(kind, name)
        if nid not in nodes:
            nodes[nid] = {"id": nid, "kind": kind, "label": name, **extra}
            degree[nid] = 0
        return nid

    def add_edge(a: str, b: str, label: str = "", *, kind: str = "rel") -> None:
        if a == b:
            return
        key = (a, b) if a <= b else (b, a)
        if key not in edges:
            edges[key] = {"from": key[0], "to": key[1], "label": label, "kind": kind}
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

    bodies: dict[str, str] = {}
    # Track which surface names are used as dynasties so a tag with the same name
    # routes to the single dynasty hub instead of spawning a duplicate tag node.
    dynasty_names: set[str] = set()
    if include_dynasty:
        for title in pages:
            meta, _ = split_frontmatter(read_page_raw(title))
            d = str(meta.get("dynasty") or "").strip()
            if d:
                dynasty_names.add(d)

    for title in pages:
        raw = read_page_raw(title)
        meta, body = split_frontmatter(raw)
        bodies[title] = body
        page_nid = add_node("page", title, dynasty=str(meta.get("dynasty") or ""))

        if include_dynasty:
            dynasty = str(meta.get("dynasty") or "").strip()
            if dynasty:
                d_nid = add_node("dynasty", dynasty)
                add_edge(page_nid, d_nid, "朝代", kind="dynasty")

        if include_tags:
            tags = meta.get("tags")
            if isinstance(tags, (list, tuple)):
                for tag in tags:
                    tag_str = str(tag).strip()
                    # Skip the ubiquitous 歷史/历史 tag — it links everything and adds noise.
                    if not tag_str or tag_str in ("歷史", "历史"):
                        continue
                    # Merge a tag that names a dynasty into that dynasty hub.
                    if tag_str in dynasty_names:
                        if include_dynasty:
                            add_edge(page_nid, add_node("dynasty", tag_str), "朝代", kind="dynasty")
                        else:
                            add_edge(page_nid, add_node("tag", tag_str), "標籤", kind="tag")
                    else:
                        t_nid = add_node("tag", tag_str)
                        add_edge(page_nid, t_nid, "標籤", kind="tag")

        if entity_map and title in entity_map:
            for ent in entity_map[title]:
                ent_text = str(ent.get("text", "")).strip()
                ent_type = str(ent.get("type", "PER")).strip().upper()
                if not ent_text or ent_type not in ("PER", "LOC", "TIME"):
                    continue
                # If the entity is itself a page, the mention edge will cover it.
                if ent_text in pages:
                    continue
                e_nid = add_node(ent_type, ent_text)
                add_edge(page_nid, e_nid, GRAPH_KIND_LABELS.get(ent_type, ent_type), kind=ent_type)

    if include_mentions:
        for source, target in _page_mention_edges(pages, bodies):
            add_edge(node_id("page", source), node_id("page", target), "提及", kind="mention")

    # Attach degree (used for node sizing) and finalize node colours.
    node_list: list[dict[str, object]] = []
    for nid, node in nodes.items():
        kind = str(node["kind"])
        node["value"] = degree.get(nid, 0)
        node["color"] = GRAPH_KIND_COLORS.get(kind, "#9AA0A6")
        node_list.append(node)

    stats = {
        "pages": sum(1 for n in node_list if n["kind"] == "page"),
        "objects": sum(1 for n in node_list if n["kind"] != "page"),
        "edges": len(edges),
    }
    return {"nodes": node_list, "edges": list(edges.values()), "stats": stats}


def _vis_network_html(graph: dict[str, object], *, height_px: int = 640) -> str:
    """Render the graph dict to a self-contained vis-network HTML document."""
    vis_nodes = []
    for node in graph["nodes"]:  # type: ignore[index]
        kind = str(node["kind"])
        label = str(node["label"])
        is_page = kind == "page"
        vis_nodes.append(
            {
                "id": node["id"],
                "label": label,
                "title": f"{GRAPH_KIND_LABELS.get(kind, kind)}: {label}",
                "color": node["color"],
                "value": int(node["value"]) + (3 if is_page else 1),
                "shape": "dot",
                "kind": kind,
                "font": {
                    "color": "#E8EAED",
                    "size": 18 if is_page else 13,
                    "strokeWidth": 3,
                    "strokeColor": "#0E1117",
                },
            }
        )
    vis_edges = []
    for edge in graph["edges"]:  # type: ignore[index]
        is_mention = edge.get("kind") == "mention"
        vis_edges.append(
            {
                "from": edge["from"],
                "to": edge["to"],
                "label": str(edge.get("label") or ""),
                "dashes": bool(is_mention),
                "color": {"color": "#5A6270", "highlight": "#4F9DFF", "opacity": 0.7},
            }
        )

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #0E1117; }}
  #graph {{ width: 100%; height: {height_px}px; background: #0E1117; }}
  .vis-tooltip {{
    background: #1C1F26 !important; color: #E8EAED !important;
    border: 1px solid #3A3F4B !important; border-radius: 6px !important;
    padding: 6px 10px !important; font-family: sans-serif !important;
  }}
</style>
</head>
<body>
<div id="graph"></div>
<script type="text/javascript">
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const container = document.getElementById('graph');
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    nodes: {{
      shape: 'dot',
      scaling: {{ min: 8, max: 46, label: {{ enabled: true, min: 12, max: 26 }} }},
      borderWidth: 2,
      color: {{ border: '#0E1117' }}
    }},
    edges: {{
      smooth: {{ type: 'continuous' }},
      width: 1.2,
      font: {{ color: '#9AA0A6', size: 11, strokeWidth: 0, align: 'middle' }},
      arrows: {{ to: {{ enabled: false }} }}
    }},
    physics: {{
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{ gravitationalConstant: -55, centralGravity: 0.012, springLength: 120, springConstant: 0.08, avoidOverlap: 0.6 }},
      stabilization: {{ iterations: 220 }},
      maxVelocity: 30
    }},
    interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: false, keyboard: false, multiselect: false }}
  }};
  const network = new vis.Network(container, data, options);
  // Surface a click so the parent page could read it (best-effort; no hard dep).
  network.on('click', function(params) {{
    if (params.nodes && params.nodes.length) {{
      const id = params.nodes[0];
      try {{ window.parent.postMessage({{ type: 'graph_node_click', id: id }}, '*'); }} catch (e) {{}}
    }}
  }});
</script>
</body>
</html>"""


def _collect_entity_map_for_graph(
    pages: list[str],
    llm_settings: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Run NER/LLM entity extraction across all pages for the AI-enriched graph."""
    entity_map: dict[str, list[dict[str, object]]] = {}
    progress = st.progress(0.0, text="正在分析頁面實體 …")
    total = max(len(pages), 1)
    for index, title in enumerate(pages, start=1):
        body = read_page(title)
        if body.strip():
            try:
                entities, _used = extract_entities_detailed(
                    body,
                    method="auto",
                    model_id=str(st.session_state.get("ner_model_id", NER_MODEL_PRIMARY)),
                    score_threshold=float(st.session_state.get("ner_threshold", NER_SCORE_THRESHOLD)),
                    llm_settings=llm_settings,
                )
                entity_map[title] = entities
            except Exception:
                entity_map[title] = []
        progress.progress(index / total, text=f"正在分析頁面實體 … ({index}/{total})")
    progress.empty()
    return entity_map


def _embed_html(html_doc: str, *, height: int) -> None:
    """Embed a self-contained HTML+JS document in a sandboxed iframe.

    Prefers ``components.html`` (a srcdoc iframe), which reliably allows the
    CDN-hosted vis.js script to load. Falls back to ``st.iframe`` with a data: URL
    only if that call is unavailable in this Streamlit version.
    """
    try:
        components.html(html_doc, height=height, scrolling=False)
    except Exception:
        data_url = "data:text/html;base64," + base64.b64encode(html_doc.encode("utf-8")).decode("ascii")
        st.iframe(data_url, height=height, width="stretch")


def render_relationship_graph(pages: list[str], llm_settings: dict[str, object]) -> None:
    """The Relationship Graph tab: filters + legend, the network canvas, details."""
    st.header("關係圖譜 (Knowledge Graph)")
    st.caption(
        "依據所有頁面的內容自動建立。頁面之間透過共享的朝代、標籤或彼此的提及而相連；"
        "新增或更新頁面後，重新整理本頁即會自動更新。"
    )

    if not pages:
        st.info("還沒有頁面。請先建立並產生一些歷史頁面。")
        return

    col_filters, col_graph, col_detail = st.columns([0.2, 0.6, 0.2])

    with col_filters:
        st.subheader("篩選 (Filters)")
        include_dynasty = st.checkbox("朝代連結", value=True, key="graph_dyn")
        include_tags = st.checkbox("標籤連結", value=True, key="graph_tags")
        include_mentions = st.checkbox("頁面互相提及", value=True, key="graph_mentions")
        st.divider()
        st.subheader("AI 深度分析")
        st.caption("額外用 NER/LLM 擷取人物、地點、時間，作為連結節點（較慢）。")
        use_ai = st.checkbox("啟用 AI 實體節點", value=False, key="graph_ai")
        if use_ai and st.button("重新分析實體", use_container_width=True, key="graph_ai_run"):
            st.session_state.pop("graph_entity_map", None)

        st.divider()
        st.subheader("圖例 (Legend)")
        for kind, color in GRAPH_KIND_COLORS.items():
            if kind == "TIME":  # TIME shares amber with dynasty; skip the dup swatch.
                continue
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
                f"<span style='width:12px;height:12px;border-radius:50%;background:{color};"
                f"display:inline-block;'></span><span>{GRAPH_KIND_LABELS.get(kind, kind)}</span></div>",
                unsafe_allow_html=True,
            )

    entity_map: dict[str, list[dict[str, object]]] | None = None
    if use_ai:
        if "graph_entity_map" not in st.session_state:
            st.session_state.graph_entity_map = _collect_entity_map_for_graph(pages, llm_settings)
        entity_map = st.session_state.get("graph_entity_map")

    graph = build_relationship_graph(
        pages,
        include_dynasty=include_dynasty,
        include_tags=include_tags,
        include_mentions=include_mentions,
        entity_map=entity_map,
    )
    stats = graph["stats"]  # type: ignore[index]

    with col_detail:
        st.subheader("概況 (Overview)")
        st.metric("頁面 Pages", stats["pages"])  # type: ignore[index]
        st.metric("關聯物件 Objects", stats["objects"])  # type: ignore[index]
        st.metric("連結 Edges", stats["edges"])  # type: ignore[index]
        st.divider()
        # Most-connected pages (a quick "central nodes" list like the reference).
        page_nodes = [n for n in graph["nodes"] if n["kind"] == "page"]  # type: ignore[index]
        page_nodes.sort(key=lambda n: int(n["value"]), reverse=True)
        if page_nodes:
            st.subheader("核心頁面 (Central)")
            for node in page_nodes[:8]:
                title = str(node["label"])
                if st.button(
                    f"🔗 {title} · {int(node['value'])}",
                    key=f"graphjump_{title}",
                    use_container_width=True,
                ):
                    select_page(title)

    with col_graph:
        if not graph["nodes"]:  # type: ignore[index]
            st.info("目前的篩選條件下沒有可顯示的節點。請放寬左側篩選。")
        else:
            html_doc = _vis_network_html(graph, height_px=660)
            _embed_html(html_doc, height=680)
            st.caption("可拖曳節點、滾輪縮放、滑鼠懸停查看類型。虛線＝頁面互相提及。")


def render_main_page(pages: list[str], llm_settings: dict[str, object]) -> None:
    st.title("中國歷史知識庫 · Chinese History Wiki")
    wiki_tab, graph_tab, timeline_tab, prompt_tab = st.tabs(
        ["維基 Wiki", "關係圖譜 Graph", "時間線 Timeline", "提示詞設定 Prompt"]
    )

    with wiki_tab:
        render_wiki_page(pages, llm_settings)

    with graph_tab:
        render_relationship_graph(pages, llm_settings)

    with timeline_tab:
        render_timeline_view(pages)

    with prompt_tab:
        render_prompt_settings()


def apply_styles() -> None:
    st.set_page_config(page_title="中國歷史知識庫 Chinese History Wiki", page_icon=":scroll:", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 1280px; padding-top: 2rem; }
        [data-testid="stSidebar"] button { text-align: left; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        /* Dark-theme polish to match the knowledge-graph aesthetic */
        .stApp { background-color: #0E1117; }
        [data-testid="stSidebar"] { background-color: #14171F; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #1C1F26; border-radius: 6px 6px 0 0; padding: 6px 14px;
        }
        .stTabs [aria-selected="true"] { background-color: #2A2F3A; }
        [data-testid="stMetricValue"] { color: #4F9DFF; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    refresh_capability_flags()  # Detect packages installed during this session.
    ensure_data_dir()
    ensure_default_prompt()
    pages = scan_pages()
    init_state(pages)
    current_content = read_page(st.session_state.current_page) if st.session_state.current_page else ""
    llm_settings = render_sidebar(pages, current_content)
    pages = scan_pages()  # Refresh after possible page creation.
    render_main_page(pages, llm_settings)


if __name__ == "__main__":
    main()
