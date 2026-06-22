from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus, urlsplit, urlunsplit

import streamlit as st

try:
    from openai import OpenAI
except ImportError:  # Keep the app usable enough to show a clear setup error.
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:  # Claude support is optional unless that provider is selected.
    Anthropic = None

try:
    import fitz  # PyMuPDF
except ImportError:  # PDF support is optional until the Document OCR tab is used.
    fitz = None

try:
    import pytesseract
except ImportError:  # OCR support is optional for PDFs that already contain text.
    pytesseract = None

try:
    import numpy as np
except ImportError:  # RapidOCR uses numpy arrays; keep startup graceful if unavailable.
    np = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # Pure-Python OCR fallback when the Tesseract program is unavailable.
    RapidOCR = None

try:
    from PIL import Image
except ImportError:  # Pillow is required by pytesseract, but keep startup graceful.
    Image = None


DATA_DIR = Path("wiki_data")
KEYS_DIR = Path("keys")
PROMPT_DIR = Path("prmopt")
SOURCE_DOCS_DIR = Path("source_docs")
DEFAULT_OCR_LANGUAGE = "chi_sim+chi_tra+eng"
DEFAULT_OCR_DPI = 200
OCR_TEXT_MIN_CHARS = 80
PDF_ANALYSIS_MAX_CHARS = 24000
DEFAULT_PROMPT_NAME = "default_wiki_maintainer"
DEFAULT_WIKI_PROMPT = """You are an autonomous Markdown wiki maintainer.

Your job is to help maintain a persistent, interlinked local wiki made of Markdown pages.

Core rules:
- Return raw Markdown only unless the UI explicitly asks for something else.
- Preserve useful existing content and improve it instead of replacing it without reason.
- Use clear headings, short paragraphs, and practical examples.
- Prefer stable page titles that can become filenames.
- When mentioning an existing wiki page by title, write the exact title text so the app can auto-link it.
- Suggest new related pages when a concept deserves its own page.
- Avoid duplicate pages by reusing existing page titles when possible.
- Keep summaries concise and factual.
- Mark uncertainty clearly instead of inventing details.
- Do not include secrets, API keys, or private configuration in generated wiki content.

Markdown style:
- Start major pages with a short overview.
- Use `##` and `###` headings for structure.
- Use bullet lists for procedures, tradeoffs, and related topics.
- Use fenced code blocks with language tags when showing code.
- Keep links readable and avoid raw HTML unless necessary.

When maintaining the wiki autonomously:
- Think in terms of durable knowledge, not one-off chat answers.
- Connect concepts across pages by reusing exact page titles.
- Keep pages easy to scan, edit, and extend later.
"""
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "llama-3.2-1b-instruct"
DEFAULT_MAX_TOKENS = 4096
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


def ensure_source_docs_dir() -> None:
    SOURCE_DOCS_DIR.mkdir(exist_ok=True)


def prompt_path(name: str) -> Path:
    safe_name = key_file_stem(name) or DEFAULT_PROMPT_NAME
    return PROMPT_DIR / f"{safe_name}.md"


def ensure_default_prompt() -> None:
    ensure_prompt_dir()
    path = prompt_path(DEFAULT_PROMPT_NAME)
    if not path.exists():
        path.write_text(DEFAULT_WIKI_PROMPT, encoding="utf-8")


def sanitize_title(raw_title: str) -> str:
    """Convert user input into a safe flat-file page title."""
    title = raw_title.strip()
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"_+", "_", title).strip(" ._")
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


def scan_pdf_documents() -> list[Path]:
    ensure_prompt_dir()
    ensure_source_docs_dir()
    candidates: dict[str, Path] = {}
    for directory in (SOURCE_DOCS_DIR, PROMPT_DIR):
        for path in directory.glob("*.pdf"):
            candidates[str(path.resolve()).lower()] = path
    return sorted(candidates.values(), key=lambda path: path.name.casefold())


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


def read_page(title: str) -> str:
    path = page_path(title)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_page(title: str, content: str) -> None:
    ensure_data_dir()
    page_path(title).write_text(content, encoding="utf-8")


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
    path.write_text("", encoding="utf-8")
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
    st.experimental_rerun()


def init_state(pages: list[str]) -> None:
    st.session_state.setdefault("current_page", None)
    st.session_state.setdefault("edit_mode", False)
    st.session_state.setdefault("show_new_page", False)
    st.session_state.setdefault("related_topics", [])
    st.session_state.setdefault("active_prompt", DEFAULT_PROMPT_NAME)
    st.session_state.setdefault("show_new_prompt", False)

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


def autolink_markdown(markdown_text: str, page_titles: list[str], current_title: str | None) -> str:
    """Lightweight wiki linking while avoiding code blocks, code spans, and existing links."""
    titles = sorted(
        [title for title in page_titles if title and title != current_title],
        key=len,
        reverse=True,
    )
    if not markdown_text or not titles:
        return markdown_text

    escaped_titles = "|".join(re.escape(title) for title in titles)
    pattern = re.compile(rf"(?<![\w\]])({escaped_titles})(?![\w\(])")
    protected_pattern = re.compile(r"(```.*?```|`[^`\n]+`|\[[^\]]+\]\([^)]+\))", re.DOTALL)

    def link_segment(segment: str) -> str:
        def replace(match: re.Match[str]) -> str:
            title = match.group(1)
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


def find_tesseract_executable() -> str:
    candidates = [
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


@st.cache_resource(show_spinner=False)
def get_rapidocr_engine():
    if RapidOCR is None:
        return None
    return RapidOCR()


def rapidocr_available() -> bool:
    return RapidOCR is not None and np is not None


def pdf_support_status() -> dict[str, object]:
    tesseract_path = find_tesseract_executable()
    languages: list[str] = []
    if pytesseract is not None and tesseract_path:
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            languages = sorted(pytesseract.get_languages(config=""))
        except Exception:
            languages = []
    chinese_language_available = any(language in languages for language in ("chi_sim", "chi_tra"))
    return {
        "pymupdf": fitz is not None,
        "pytesseract": pytesseract is not None,
        "rapidocr": rapidocr_available(),
        "pillow": Image is not None,
        "tesseract_path": tesseract_path or "",
        "languages": languages,
        "chinese_language_available": chinese_language_available,
    }


def get_pdf_page_count(path: Path) -> int:
    if fitz is None:
        return 0
    try:
        with fitz.open(path) as document:
            return document.page_count
    except Exception:
        return 0


def clamp_page_range(start_page: int, end_page: int, page_count: int) -> tuple[int, int]:
    if page_count <= 0:
        return 1, 1
    start = max(1, min(start_page, page_count))
    end = max(start, min(end_page, page_count))
    return start, end


def render_pdf_page_image(page, dpi: int):
    scale = max(72, dpi) / 72
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image_bytes = pixmap.tobytes("png")
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def ocr_with_tesseract(image, language: str, tesseract_path: str) -> str:
    if pytesseract is None:
        raise RuntimeError("The `pytesseract` Python package is not installed.")
    if not tesseract_path:
        raise RuntimeError("The Tesseract program is not installed or is not on PATH.")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    try:
        return pytesseract.image_to_string(image, lang=language).strip()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR failed. Confirm that Chinese language data "
            f"for `{language}` is available. Details: {exc}"
        ) from exc


def ocr_with_rapidocr(image) -> str:
    if not rapidocr_available():
        raise RuntimeError(
            "RapidOCR is not installed. Install it with `python -m pip install rapidocr_onnxruntime`."
        )
    engine = get_rapidocr_engine()
    if engine is None:
        raise RuntimeError("RapidOCR could not be initialized.")
    image_array = np.array(image.convert("RGB"))
    result, _elapsed = engine(image_array)
    lines: list[str] = []
    for item in result or []:
        if len(item) >= 2 and isinstance(item[1], str):
            text = item[1].strip()
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


def ocr_page_image(image, language: str, engine: str) -> tuple[str, str]:
    selected_engine = engine
    status = pdf_support_status()
    if selected_engine == "auto":
        selected_engine = "tesseract" if status.get("tesseract_path") else "rapidocr"

    if selected_engine == "tesseract":
        return (
            ocr_with_tesseract(image, language, str(status.get("tesseract_path") or "")),
            "ocr-tesseract",
        )
    if selected_engine == "rapidocr":
        return ocr_with_rapidocr(image), "ocr-rapidocr"

    raise RuntimeError(f"Unknown OCR engine: {engine}")


def extract_pdf_text(
    path: Path,
    *,
    start_page: int,
    end_page: int,
    use_ocr: bool,
    ocr_language: str,
    ocr_engine: str,
    dpi: int,
    progress_callback=None,
) -> tuple[str, list[dict[str, object]]]:
    if fitz is None:
        raise RuntimeError("Install `PyMuPDF` to read PDF files: `python -m pip install PyMuPDF`.")
    if use_ocr and Image is None:
        raise RuntimeError("Install `Pillow` before using OCR: `python -m pip install Pillow`.")

    page_results: list[dict[str, object]] = []
    with fitz.open(path) as document:
        start, end = clamp_page_range(start_page, end_page, document.page_count)
        selected_pages = list(range(start - 1, end))
        total_pages = len(selected_pages)
        for index, page_index in enumerate(selected_pages, start=1):
            page = document.load_page(page_index)
            extracted = (page.get_text("text") or "").strip()
            method = "text-layer"
            page_text = extracted
            if use_ocr and len(re.sub(r"\s+", "", extracted)) < OCR_TEXT_MIN_CHARS:
                image = render_pdf_page_image(page, dpi)
                page_text, method = ocr_page_image(image, ocr_language, ocr_engine)

            page_number = page_index + 1
            page_results.append(
                {
                    "page": page_number,
                    "method": method,
                    "chars": len(page_text),
                    "text": page_text,
                }
            )
            if progress_callback is not None:
                progress_callback(index, total_pages, page_number, method)

    chunks = []
    for result in page_results:
        chunks.append(f"## Page {result['page']} ({result['method']})\n\n{result['text']}".strip())
    return "\n\n---\n\n".join(chunks).strip(), page_results


def truncate_for_llm(text: str, max_chars: int = PDF_ANALYSIS_MAX_CHARS) -> tuple[str, bool]:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned, False
    return cleaned[:max_chars].rstrip(), True


def build_document_source_note(
    *,
    pdf_path: Path,
    start_page: int,
    end_page: int,
    page_results: list[dict[str, object]],
    ocr_language: str,
    ocr_engine: str,
    used_ocr: bool,
) -> str:
    method_counts: dict[str, int] = {}
    for result in page_results:
        method = str(result.get("method") or "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
    method_summary = ", ".join(f"{method}: {count}" for method, count in sorted(method_counts.items()))
    total_chars = sum(int(result.get("chars") or 0) for result in page_results)
    return (
        f"Source PDF: `{pdf_path.name}`\n"
        f"Pages analyzed: {start_page}-{end_page}\n"
        f"Extraction methods: {method_summary or 'none'}\n"
        f"OCR engine: `{ocr_engine}`\n"
        f"OCR language: `{ocr_language}`\n"
        f"OCR fallback enabled: {'yes' if used_ocr else 'no'}\n"
        f"Extracted characters: {total_chars}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


def analyze_document_text(
    *,
    pdf_path: Path,
    extracted_text: str,
    page_range: tuple[int, int],
    analysis_prompt: str,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, bool]:
    text_for_llm, was_truncated = truncate_for_llm(extracted_text)
    system_prompt = build_system_prompt(
        "You are a careful Chinese historical-literature research assistant. Return raw Markdown only. "
        "Use the supplied OCR/PDF text as evidence, preserve Chinese titles and names, and mark uncertainty clearly."
    )
    user_prompt = (
        f"Analyze the Chinese historical/literary PDF `{pdf_path.name}` for wiki notes.\n"
        f"Page range: {page_range[0]}-{page_range[1]}.\n"
        f"Text was truncated before analysis: {'yes' if was_truncated else 'no'}.\n\n"
        "User analysis request:\n"
        f"{analysis_prompt.strip()}\n\n"
        "Extracted PDF/OCR text:\n"
        f"{text_for_llm}"
    )
    analysis = complete_chat(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    ).strip()
    return analysis, was_truncated


def save_uploaded_pdf(uploaded_file) -> Path:
    ensure_source_docs_dir()
    safe_name = sanitize_title(Path(uploaded_file.name).stem)
    target = SOURCE_DOCS_DIR / f"{safe_name}.pdf"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def format_pdf_option(path: Path) -> str:
    page_count = get_pdf_page_count(path)
    location = path.parent.name
    page_suffix = f", {page_count} pages" if page_count else ""
    return f"{path.name} ({location}{page_suffix})"


def render_pdf_status(status: dict[str, object]) -> None:
    with st.expander("OCR Setup Status", expanded=False):
        if status["pymupdf"]:
            st.success("PDF text extraction is available through PyMuPDF.")
        else:
            st.error("PyMuPDF is missing. Install it with `python -m pip install PyMuPDF`.")

        if status["pytesseract"] and status["pillow"]:
            st.success("Tesseract Python OCR package is installed.")
        else:
            st.warning("Install OCR Python packages with `python -m pip install pytesseract Pillow`.")

        if status.get("rapidocr"):
            st.success("RapidOCR fallback is available. This can OCR Chinese scanned pages without Tesseract.")
        else:
            st.warning("RapidOCR fallback is missing. Install it with `python -m pip install rapidocr_onnxruntime`.")

        tesseract_path = str(status.get("tesseract_path") or "")
        if tesseract_path:
            st.success(f"Tesseract found at `{tesseract_path}`.")
        else:
            if status.get("rapidocr"):
                st.info("Tesseract is not installed, so Auto OCR will use RapidOCR instead.")
            else:
                st.warning(
                    "Tesseract OCR is not on PATH. Text-layer extraction still works, but scanned pages need "
                    "Tesseract plus Chinese language data (`chi_sim` and/or `chi_tra`) or RapidOCR."
                )

        languages = status.get("languages") or []
        if languages:
            preview = ", ".join(str(language) for language in languages[:12])
            st.caption(f"Available OCR languages include: {preview}")
        if not status.get("chinese_language_available"):
            st.info(
                "For Chinese PDFs, install Tesseract language packs for Simplified Chinese (`chi_sim`) "
                "and Traditional Chinese (`chi_tra`) if OCR fallback is needed."
            )


def render_document_ocr(llm_settings: dict[str, object]) -> None:
    st.header("Document OCR")
    st.caption(
        "Extract Chinese PDF text into the wiki, with OCR fallback for scanned pages, then ask the selected LLM "
        "to turn the text into historical/literary notes."
    )

    status = pdf_support_status()
    render_pdf_status(status)

    uploaded_files = st.file_uploader(
        "Add PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Uploaded PDFs are saved in `{SOURCE_DOCS_DIR}/`. Existing PDFs in `{PROMPT_DIR}/` are also listed.",
    )
    if uploaded_files:
        saved_paths = []
        for uploaded_file in uploaded_files:
            saved_paths.append(save_uploaded_pdf(uploaded_file))
        st.success("Saved: " + ", ".join(path.name for path in saved_paths))

    pdf_paths = scan_pdf_documents()
    if not pdf_paths:
        st.info(f"Put PDFs in `{SOURCE_DOCS_DIR}/` or `{PROMPT_DIR}/`, then return to this tab.")
        return

    if fitz is None:
        st.error("PDF support is unavailable until PyMuPDF is installed.")
        return

    option_labels = [format_pdf_option(path) for path in pdf_paths]
    selected_label = st.selectbox("PDF to analyze", option_labels)
    pdf_path = pdf_paths[option_labels.index(selected_label)]
    page_count = get_pdf_page_count(pdf_path)
    if page_count <= 0:
        st.error("This PDF could not be opened. Try another file.")
        return

    st.caption(f"Selected file: `{pdf_path}`")

    col_start, col_end, col_dpi = st.columns(3)
    start_page = int(
        col_start.number_input("Start page", min_value=1, max_value=page_count, value=1, step=1)
    )
    default_end = min(page_count, start_page + 4)
    end_page = int(
        col_end.number_input("End page", min_value=start_page, max_value=page_count, value=default_end, step=1)
    )
    dpi = int(
        col_dpi.number_input(
            "OCR DPI",
            min_value=100,
            max_value=400,
            value=DEFAULT_OCR_DPI,
            step=25,
            help="Higher DPI can improve OCR accuracy but is slower.",
        )
    )

    use_ocr = st.checkbox(
        "Use OCR fallback when embedded PDF text is sparse",
        value=True,
        help="If a page already has readable text, the app uses it. OCR is only used for weak/scanned pages.",
    )
    engine_options = ["auto", "rapidocr", "tesseract"]
    ocr_engine = st.selectbox(
        "OCR engine",
        engine_options,
        index=0,
        help="Auto uses Tesseract when available; otherwise it uses RapidOCR, which works without a separate install.",
    )
    ocr_language = st.text_input(
        "Tesseract OCR language",
        value=DEFAULT_OCR_LANGUAGE,
        help="Used only by Tesseract. RapidOCR uses its built-in Chinese/English OCR model.",
    )
    if use_ocr and ocr_engine == "tesseract" and not status.get("tesseract_path"):
        st.warning("Tesseract engine selected, but Tesseract is not installed. Choose Auto or RapidOCR.")
    if use_ocr and ocr_engine in ("auto", "rapidocr") and not status.get("tesseract_path") and status.get("rapidocr"):
        st.info("Tesseract is not installed, so this run will use RapidOCR for scanned pages.")

    analysis_prompt = st.text_area(
        "Analysis prompt",
        value=(
            "Create concise Markdown wiki notes in English with Chinese terms preserved. "
            "Identify the work, author/context when visible, important people/places, themes, historical value, "
            "and possible related wiki pages. Mention uncertain OCR readings."
        ),
        height=120,
    )

    default_page_title = sanitize_title(f"{pdf_path.stem} p{start_page}-{end_page}")
    target_title = st.text_input("Save as wiki page", value=default_page_title)

    col_extract, col_save_raw, col_analyze = st.columns(3)
    extract_clicked = col_extract.button("Extract Preview", use_container_width=True)
    save_raw_clicked = col_save_raw.button("Save Raw Text", use_container_width=True)
    analyze_clicked = col_analyze.button("Analyze and Save", type="primary", use_container_width=True)

    if not (extract_clicked or save_raw_clicked or analyze_clicked):
        st.info("Start with a small page range, preview the extraction, then save or analyze it.")
        return

    start_page, end_page = clamp_page_range(start_page, end_page, page_count)
    progress = st.progress(0.0)
    status_line = st.empty()

    def update_progress(index: int, total: int, page_number: int, method: str) -> None:
        progress.progress(index / total)
        status_line.caption(f"Processed page {page_number} with {method} ({index}/{total}).")

    try:
        with st.spinner("Extracting PDF text..."):
            extracted_text, page_results = extract_pdf_text(
                pdf_path,
                start_page=start_page,
                end_page=end_page,
                use_ocr=use_ocr,
                ocr_language=ocr_language.strip() or DEFAULT_OCR_LANGUAGE,
                ocr_engine=ocr_engine,
                dpi=dpi,
                progress_callback=update_progress,
            )
    except Exception as exc:
        st.error(str(exc))
        return

    source_note = build_document_source_note(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        page_results=page_results,
        ocr_language=ocr_language.strip() or DEFAULT_OCR_LANGUAGE,
        ocr_engine=ocr_engine,
        used_ocr=use_ocr,
    )
    safe_target_title = sanitize_title(target_title)
    st.success("Extraction complete.")
    st.text_area("Extracted text preview", value=extracted_text[:12000], height=360)

    if save_raw_clicked:
        content = f"# {safe_target_title}\n\n## Source\n\n{source_note}\n\n## Extracted Text\n\n{extracted_text}\n"
        write_page(safe_target_title, content)
        st.success(f"Saved raw extracted text to `wiki_data/{safe_target_title}.md`.")
        return

    if analyze_clicked:
        try:
            with st.spinner("Asking the selected LLM to analyze the extracted text..."):
                analysis, was_truncated = analyze_document_text(
                    pdf_path=pdf_path,
                    extracted_text=extracted_text,
                    page_range=(start_page, end_page),
                    analysis_prompt=analysis_prompt,
                    **llm_settings,
                )
        except Exception as exc:
            st.error(format_llm_error(exc, str(llm_settings.get("provider") or ""), str(llm_settings.get("base_url") or "")))
            return

        truncation_note = (
            "\n\n> Note: The extracted text was longer than the analysis limit, so only the first "
            f"{PDF_ANALYSIS_MAX_CHARS} characters were sent to the LLM."
            if was_truncated
            else ""
        )
        content = (
            f"# {safe_target_title}\n\n"
            f"## Source\n\n{source_note}{truncation_note}\n\n"
            f"## AI Analysis\n\n{analysis}\n\n"
            f"## Extracted Text\n\n{extracted_text}\n"
        )
        write_page(safe_target_title, content)
        st.markdown(autolink_markdown(content, scan_pages(), safe_target_title))
        st.success(f"Saved analysis to `wiki_data/{safe_target_title}.md`.")


def render_sidebar(pages: list[str], current_content: str) -> dict[str, object]:
    st.sidebar.title("Raw LLM Wiki")

    with st.sidebar.expander("Model Configuration", expanded=True):
        source = st.radio(
            "Model source",
            [SOURCE_LOCAL, SOURCE_ONLINE],
            horizontal=True,
            help="Switch between a local OpenAI-compatible server and online API providers.",
        )
        provider = PROVIDER_LOCAL
        if source == SOURCE_ONLINE:
            provider = st.selectbox("API provider", ONLINE_PROVIDERS)

        config = get_provider_config(provider)
        st.caption(str(config.get("help") or ""))
        if provider in (PROVIDER_NEWAPI, PROVIDER_CUSTOM):
            st.info(
                "For custom OpenAI-compatible APIs, paste the provider URL in Base URL. "
                "You can also paste New API JSON into API Key and click Save Key."
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
                    "Use the provider root URL or OpenAI-compatible /v1 URL. "
                    "Do not include endpoint paths like /chat/completions."
                ),
            )
        else:
            st.caption("Claude uses Anthropic's native Messages API, so no Base URL is needed.")

        env_name = str(config.get("api_key_env") or "")
        key_help = "Local servers often ignore this, but the OpenAI SDK requires a non-empty value."
        if env_name:
            key_help = (
                f"Paste a key here, save one to {api_key_path(provider)}, "
                f"or set the {env_name} environment variable before starting Streamlit."
            )
        elif provider in (PROVIDER_NEWAPI, PROVIDER_CUSTOM):
            key_help = (
                "Paste a raw API key or a New API JSON object like "
                '{"_type":"newapi_channel_conn","key":"sk-...","url":"https://www.juaiapi.com"}.'
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
        key_status = f"Saved key file: `{api_key_path(provider)}`"
        if saved_api_key:
            st.caption(f"{key_status} is loaded for this provider.")
        else:
            st.caption(f"No saved key for this provider. {key_status}")

        col_save_key, col_delete_key = st.columns(2)
        if col_save_key.button("Save Key", key=f"save_key_{provider}", use_container_width=True):
            try:
                parsed_connection = parse_api_key_input(api_key)
                parsed_api_key = parsed_connection.get("api_key", "")
                parsed_base_url = parsed_connection.get("base_url", "")
                save_api_key(provider, parsed_api_key, parsed_base_url or base_url)
                if parsed_base_url and backend == BACKEND_OPENAI_COMPATIBLE:
                    st.session_state[f"base_url_{provider}"] = normalize_base_url(parsed_base_url)
                    st.session_state[api_key_widget_key] = parsed_api_key
                st.success(f"Saved API key for {provider}.")
                rerun()
            except ValueError as exc:
                st.warning(str(exc))
        if col_delete_key.button(
            "Delete Saved Key",
            key=f"delete_key_{provider}",
            disabled=not saved_api_key,
            use_container_width=True,
        ):
            if delete_saved_api_key(provider):
                st.session_state[api_key_widget_key] = str(config.get("api_key") or "")
                if backend == BACKEND_OPENAI_COMPATIBLE:
                    st.session_state[f"base_url_{provider}"] = str(config.get("base_url") or DEFAULT_BASE_URL)
                st.success(f"Deleted saved API key for {provider}.")
                rerun()
        if backend == BACKEND_OPENAI_COMPATIBLE and st.button(
            "Test Provider",
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
                    st.success(f"Connected. Available chat models include: {preview}")
                else:
                    st.warning("Connected, but no non-embedding models were returned by /v1/models.")
            except Exception as exc:
                st.error(format_llm_error(exc, provider, base_url))
        model = st.text_input(
            "Model",
            value=str(config.get("model") or ""),
            key=f"model_{provider}",
            help=(
                "Use a chat-capable model ID from the selected provider. "
                "For local/custom OpenAI-compatible providers, leave blank to try models returned by /v1/models."
            ),
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=32768,
            value=DEFAULT_MAX_TOKENS,
            step=256,
            help="Used directly by Claude and kept with settings for providers that need an output limit.",
        )

    with st.sidebar.expander("Prompt Setting", expanded=False):
        prompts = scan_prompts()
        active_prompt = get_active_prompt_name()
        selected_prompt = st.selectbox(
            "Active prompt",
            prompts,
            index=prompts.index(active_prompt),
            help="This prompt is read before every AI wiki task.",
        )
        if selected_prompt != active_prompt:
            st.session_state.active_prompt = selected_prompt
            rerun()
        st.caption(f"Using `prmopt/{selected_prompt}.md`")

    if st.sidebar.button("Create New Page", use_container_width=True):
        st.session_state.show_new_page = not st.session_state.show_new_page

    if st.session_state.show_new_page:
        with st.sidebar.form("new_page_form", clear_on_submit=True):
            new_title = st.text_input("New page title")
            submitted = st.form_submit_button("Create Page", use_container_width=True)
            if submitted:
                safe_title, created = create_page(new_title)
                st.session_state.show_new_page = False
                if not created:
                    st.sidebar.warning(f"`{safe_title}` already exists; opening it.")
                select_page(safe_title)

    st.sidebar.divider()
    st.sidebar.subheader("Pages")
    if not pages:
        st.sidebar.caption("No pages yet. Create one to begin.")
    for title in pages:
        marker = "* " if title == st.session_state.current_page else "- "
        if st.sidebar.button(f"{marker}{title}", key=f"page_{title}", use_container_width=True):
            select_page(title)

    st.sidebar.divider()
    with st.sidebar.expander("AI Agent", expanded=True):
        current_page = st.session_state.current_page
        if not current_page:
            st.caption("Create or select a page to use the AI tools.")
        else:
            parsed_current_connection = parse_api_key_input(api_key)
            request_api_key = parsed_current_connection.get("api_key", api_key)
            request_base_url = parsed_current_connection.get("base_url", base_url)
            draft_prompt = st.text_area(
                "Draft prompt",
                value=f"Write a useful, well-structured wiki page about {current_page}.",
                height=110,
            )
            is_empty = not current_content.strip()
            if not is_empty:
                st.caption("Draft generation is enabled only for empty pages.")
            if st.button("Generate Draft", disabled=not is_empty, use_container_width=True):
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
    st.subheader("Generating Draft")
    placeholder = st.empty()
    content = ""
    system_prompt = build_system_prompt(
        "You are a precise wiki-writing assistant. Return raw Markdown only. "
        "Do not wrap the full answer in a code fence."
    )
    user_prompt = (
        f"Create a Markdown wiki page titled `{title}`.\n\n"
        f"User request:\n{prompt}\n\n"
        "Use clear headings, concise explanations, and practical detail."
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
        write_page(title, content.strip() + "\n")
        st.session_state.edit_mode = False
        st.success("Draft saved.")
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
        st.warning("This page is empty. Generate or write content first.")
        return

    system_prompt = build_system_prompt("You are a concise wiki editor. Return Markdown only.")
    user_prompt = (
        f"Summarize this wiki page titled `{title}` in one short paragraph and 3-5 bullets.\n\n"
        f"{content}"
    )
    try:
        with st.spinner("Summarizing page..."):
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
        new_content = f"## AI Summary\n\n{summary}\n\n---\n\n{content.lstrip()}"
        write_page(title, new_content)
        st.success("Summary added to the top of the page.")
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
        st.warning("This page is empty. Generate or write content first.")
        return

    system_prompt = build_system_prompt(
        "You suggest concise wiki page titles. Return only a plain list, one title per line."
    )
    user_prompt = (
        "Suggest 3-5 related wiki pages to create next based on this page. "
        "Use short title-style phrases with no explanations.\n\n"
        f"Current page: {title}\n\n{content}"
    )
    try:
        with st.spinner("Finding related topics..."):
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

    st.subheader("Suggested Related Topics")
    for topic in topics:
        col_title, col_action = st.columns([0.72, 0.28])
        exists = topic in existing_pages
        col_title.markdown(f"- **{topic}**" + (" *(exists)*" if exists else ""))
        if col_action.button("Open" if exists else "Create", key=f"topic_{topic}"):
            if not exists:
                create_page(topic)
            select_page(topic)


def render_delete_page(pages: list[str], current_page: str) -> None:
    with st.expander("Delete Page"):
        st.warning("This permanently deletes the Markdown file for this page.")
        confirmation = st.text_input(
            "Type the page title to confirm deletion",
            key=f"delete_confirm_{current_page}",
            placeholder=current_page,
        )
        if st.button(
            "Delete This Page",
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
                st.success(f"Deleted `{current_page}`.")
            else:
                st.warning(f"`{current_page}` was already missing.")
            rerun()


def render_prompt_settings() -> None:
    st.header("Prompt Setting")
    st.caption("Prompts are saved as Markdown files in `prmopt/` and read before every AI wiki task.")

    prompts = scan_prompts()
    active_prompt = get_active_prompt_name()
    selected_prompt = st.selectbox(
        "Prompt to use",
        prompts,
        index=prompts.index(active_prompt),
        key="prompt_settings_selector",
    )
    if selected_prompt != active_prompt:
        st.session_state.active_prompt = selected_prompt
        rerun()

    prompt_content = read_prompt(selected_prompt)
    edited_prompt = st.text_area(
        "Prompt Markdown",
        value=prompt_content,
        height=520,
        key=f"prompt_editor_{selected_prompt}",
        help="Teach the AI how to maintain this wiki before it receives the specific draft/summary/topic task.",
    )

    col_save, col_reset, col_delete = st.columns(3)
    if col_save.button("Save Prompt", type="primary", use_container_width=True):
        saved_name = write_prompt(selected_prompt, edited_prompt)
        st.session_state.active_prompt = saved_name
        st.success(f"Saved `prmopt/{saved_name}.md`.")
        rerun()
    if col_reset.button(
        "Reset Default",
        disabled=selected_prompt != DEFAULT_PROMPT_NAME,
        use_container_width=True,
    ):
        write_prompt(DEFAULT_PROMPT_NAME, DEFAULT_WIKI_PROMPT)
        st.success("Default prompt restored.")
        rerun()
    if col_delete.button(
        "Delete Prompt",
        disabled=selected_prompt == DEFAULT_PROMPT_NAME,
        use_container_width=True,
    ):
        if delete_prompt(selected_prompt):
            st.session_state.active_prompt = DEFAULT_PROMPT_NAME
            st.success(f"Deleted `prmopt/{selected_prompt}.md`.")
            rerun()

    st.divider()
    st.subheader("Create Prompt")
    with st.form("new_prompt_form", clear_on_submit=True):
        new_prompt_name = st.text_input("New prompt name", placeholder="research_wiki_maintainer")
        new_prompt_seed = st.text_area(
            "Initial prompt",
            value=DEFAULT_WIKI_PROMPT,
            height=220,
        )
        submitted = st.form_submit_button("Create and Use Prompt", use_container_width=True)
        if submitted:
            safe_name, created = create_prompt(new_prompt_name, new_prompt_seed)
            st.session_state.active_prompt = safe_name
            if not created:
                st.warning(f"`prmopt/{safe_name}.md` already exists; switched to it.")
            else:
                st.success(f"Created `prmopt/{safe_name}.md`.")
            rerun()


def render_wiki_page(pages: list[str], llm_settings: dict[str, object]) -> None:
    current_page = st.session_state.current_page

    if not current_page:
        st.info("Create a page from the sidebar to start your local Markdown wiki.")
        return

    content = read_page(current_page)
    st.caption(f"`wiki_data/{current_page}.md`")
    st.header(current_page)

    st.session_state.edit_mode = st.toggle("Edit Page", value=st.session_state.edit_mode)

    if st.session_state.edit_mode:
        edited = st.text_area("Markdown source", value=content, height=520, key=f"editor_{current_page}")
        col_save, col_cancel = st.columns(2)
        if col_save.button("Save Changes", type="primary", use_container_width=True):
            write_page(current_page, edited)
            st.session_state.edit_mode = False
            st.success("Page saved.")
            rerun()
        if col_cancel.button("Cancel", use_container_width=True):
            st.session_state.edit_mode = False
            rerun()
        return

    col_summary, col_topics = st.columns(2)
    if col_summary.button("Summarize This Page", use_container_width=True):
        prepend_summary(title=current_page, content=content, **llm_settings)
    if col_topics.button("Suggest Related Topics", use_container_width=True):
        suggest_related_topics(title=current_page, content=content, **llm_settings)

    render_related_topics(pages)
    st.divider()

    if content.strip():
        st.markdown(autolink_markdown(content, pages, current_page))
    else:
        st.info("This page is empty. Use the AI Agent in the sidebar to generate a draft, or turn on Edit Page.")

    st.divider()
    render_delete_page(pages, current_page)


def render_main_page(pages: list[str], llm_settings: dict[str, object]) -> None:
    st.title("Raw LLM Wiki")
    wiki_tab, document_tab, prompt_tab = st.tabs(["Wiki", "Document OCR", "Prompt Setting"])

    with wiki_tab:
        render_wiki_page(pages, llm_settings)

    with document_tab:
        render_document_ocr(llm_settings)

    with prompt_tab:
        render_prompt_settings()


def apply_styles() -> None:
    st.set_page_config(page_title="Raw LLM Wiki", page_icon=":books:", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 980px; padding-top: 2rem; }
        [data-testid="stSidebar"] button { text-align: left; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    ensure_data_dir()
    ensure_default_prompt()
    ensure_source_docs_dir()
    pages = scan_pages()
    init_state(pages)
    current_content = read_page(st.session_state.current_page) if st.session_state.current_page else ""
    llm_settings = render_sidebar(pages, current_content)
    pages = scan_pages()  # Refresh after possible page creation.
    render_main_page(pages, llm_settings)


if __name__ == "__main__":
    main()
