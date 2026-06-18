from __future__ import annotations

import json
import os
import re
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


DATA_DIR = Path("wiki_data")
KEYS_DIR = Path("keys")
PROMPT_DIR = Path("prmopt")
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
PROVIDER_NEWAPI = "JuAI / NewAPI"
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
    wiki_tab, prompt_tab = st.tabs(["Wiki", "Prompt Setting"])

    with wiki_tab:
        render_wiki_page(pages, llm_settings)

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
    pages = scan_pages()
    init_state(pages)
    current_content = read_page(st.session_state.current_page) if st.session_state.current_page else ""
    llm_settings = render_sidebar(pages, current_content)
    pages = scan_pages()  # Refresh after possible page creation.
    render_main_page(pages, llm_settings)


if __name__ == "__main__":
    main()
