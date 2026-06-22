# Chinese History Wiki (中国历史知识库)

Chinese History Wiki is a Streamlit app for building a local, interlinked Markdown knowledge base of Chinese history, with optional AI help. Pages are stored as plain `.md` files (with YAML frontmatter) in `wiki_data/`, and provider API keys can be saved as local JSON files in `keys/`.

It is a specialized refactor of a generic "Raw LLM Wiki": the AI is prompted to act as an objective historian, mentions of existing pages auto-link even in unspaced Chinese text, page filenames stay valid despite Chinese punctuation, pages carry `dynasty`/`year`/`tags` frontmatter, and a Timeline view orders everything chronologically. A built-in NER tool extracts historical figures, places, and eras and turns them into new pages.

## Features

- Create, browse, edit, and delete local Markdown wiki pages.
- Store pages as flat files under `wiki_data/`, each with YAML frontmatter (`title`, `dynasty`, `year`, `tags`).
- **Chinese auto-linking**: mentions of existing page titles are linked even without spaces between words (e.g. inside `在唐朝时期`).
- **Historical NER**: one click extracts people `[PER]`, places `[LOC]`, and eras/dynasties `[TIME]` and auto-creates a page for each.
  - Uses a local BERT Chinese NER model when `transformers` is installed; otherwise falls back to your configured LLM, and finally to a built-in dynasty/era gazetteer — so the button always works.
- **Relationship graph (關係圖譜)**: a dedicated tab renders an interactive force-directed knowledge network of the whole wiki. Pages are connected through the objects they share — dynasty, tags, and (optionally) AI-extracted people/places/times — plus direct page→page mention edges. It is rebuilt from the Markdown files on every render, so adding or editing a page updates the graph automatically. Needs **no extra Python packages** (renders via vis-network from a CDN), so it works on any Python version.
- **OCR (圖片文字辨識)**: upload an image or PDF and extract Chinese text into an editable box, then append it to the current page or create a new page from it. Engines: PaddleOCR / Tesseract (offline) or any configured vision LLM. PDFs are rasterized page-by-page.
- **One-click dependency installer**: the sidebar `依賴套件 (Dependencies)` panel installs optional add-ons (PyMuPDF for PDF, Tesseract, PaddleOCR, BERT NER) into the running environment — no terminal needed. Features light up after the page reloads.
- **Timeline view**: a dedicated tab (and an optional sidebar sort) orders pages chronologically by their `year` frontmatter.
- **Traditional Chinese UI**: the interface and the default AI prompt are in Traditional Chinese (繁體中文), with a dark theme.
- **Objective-historian prompt**: the default system prompt directs the model to stay factual, cite differing historical viewpoints, and use clear headings (背景 / 經過 / 結果 / 歷史評價).
- **Chinese-safe filenames**: full-width punctuation (`：，。、《》「」“”（）…`) is stripped or replaced so file paths are valid on Windows, macOS, and Linux.
- Switch between a local model server and online API providers in the sidebar.
- Use OpenAI-compatible providers (LM Studio, OpenAI GPT, Google Gemini, custom endpoints) or Anthropic Claude.
- Save API keys per provider as JSON files under `keys/`; import New API connection JSON.
- Generate drafts for empty pages (with auto-filled frontmatter), add AI summaries, and suggest related topics.
- Manage reusable AI behavior prompts in `prmopt/`.

## Requirements

- Python 3.10 or newer (for the local BERT NER model, prefer 3.10–3.12; `torch`/`transformers` may lack wheels on the very newest releases such as 3.14)
- Streamlit
- OpenAI Python SDK
- PyYAML (recommended; the app has a minimal built-in fallback parser if missing)
- Optional: Anthropic Python SDK, required only when using Claude
- Optional: `transformers` + `torch`, only for the local BERT-based Chinese NER model
- Optional: a local OpenAI-compatible LLM server, such as LM Studio or an Ollama OpenAI-compatible server

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Optional add-ons:

```powershell
python -m pip install anthropic            # only for the Claude provider
python -m pip install transformers torch   # only for local BERT Chinese NER
```

Run the app:

```powershell
streamlit run app.py
```

Open the URL Streamlit prints in the terminal, usually `http://localhost:8501`.

## One-Click Start

On Windows, double-click `Start Raw LLM Wiki.bat`.

The launcher will:

- Open this project folder.
- Create `.venv` if it does not exist.
- Install dependencies from `requirements.txt` if they are missing.
- Open `http://localhost:8501` in your browser.
- Start the Streamlit app.

Keep the launcher window open while using the app. Close the window to stop the app.

## Basic Workflow

1. Use `Create New Page` in the sidebar to create a wiki page (e.g. `赤壁之战`).
2. Select a page from the sidebar page list. Toggle `按年代排序` to order it chronologically.
3. Turn on `Edit Page` to write Markdown manually (you can edit the YAML frontmatter at the top too).
4. For empty pages, use `Generate Draft` in the `AI Agent` panel. The draft is saved with auto-filled `dynasty`/`year`/`tags` frontmatter.
5. Use `Summarize This Page`, `Suggest Related Topics`, or `Extract & Create Entity Pages` on an existing page.
6. Open the `时间线 Timeline` tab to see all pages ordered by year.
7. Use `Delete Page` at the bottom of a page when you want to remove it.
8. Use the `Prompt Setting` tab to edit how the AI maintains the wiki.

## Historical Entity Extraction (NER)

On any non-empty page, click `Extract & Create Entity Pages`. The app runs Chinese Named Entity Recognition over the page body, then automatically creates a wiki page for each extracted entity, grouped into:

- `[PER]` 历史人物 — historical figures (e.g. 李世民, 曹操)
- `[LOC]` 地点 — places and historical sites (e.g. 长安, 赤壁)
- `[TIME]` 时间/朝代 — eras, reign titles, and dynasties (e.g. 唐朝, 贞观十九年)

The extractor chooses the best available backend automatically:

1. **Local BERT model** (best quality) — used when `transformers` is installed. Configure the model and score threshold in the sidebar `实体识别 NER Setting` panel. The default model `shibing624/bert4ner-base-chinese` (Apache-2.0) directly emits PER/LOC/TIME; `ckiplab/bert-base-chinese-ner` is also available (richer labels, but GPL-3.0). The first run downloads ~400 MB.
2. **LLM extraction** — if `transformers` is not installed, the app asks your configured chat model to return entities as JSON.
3. **Gazetteer** — with no ML library and no API, a built-in dynasty/era dictionary still extracts `[TIME]` entities offline.

The public helper `extract_historical_entities(content: str) -> list[str]` returns the deduplicated entity list and can be reused programmatically.

## Frontmatter and Timeline

Every page is stored with YAML frontmatter at the top, for example:

```yaml
---
title: 赤壁之战
dynasty: 东汉
year: 208
tags: [历史, 三国, 战役]
---
```

- `write_page` and `generate_draft` always write frontmatter. Missing fields are inferred from a dynasty gazetteer (and, for drafts, from the model).
- `year` is an integer; negative values mean BCE (e.g. `-202` = 公元前202年).
- The `时间线 Timeline` tab reads the `year` of every page and renders a chronologically sorted timeline grouped by dynasty. Undated pages are listed separately. A sidebar toggle can also sort the page list by year.
- PyYAML is used when available; otherwise a small built-in parser/dumper handles the simple frontmatter the app writes.

## Chinese Filenames

Page titles are sanitized into safe filenames. In addition to ASCII characters that are illegal on Windows/Unix (`< > : " / \ | ? *`), the app strips or replaces Chinese/full-width punctuation such as `：，。、；！？“”‘’「」『』《》（）【】…·` and the ideographic space, so a title like `汉武帝：雄才大略` becomes `汉武帝_雄才大略.md`. Chinese characters themselves are preserved, so filenames stay readable.

## Prompt Setting

The app stores AI behavior prompts as Markdown files in `prmopt/`. The active prompt is read before every AI task, including draft generation, summaries, related-topic suggestions, and entity extraction.

The default prompt is `prmopt/chinese_history_wiki_maintainer.md`. It instructs the AI to act as a professional, objective maintainer of a Chinese-history wiki: return raw Markdown, keep a rigorous academic tone, list differing viewpoints for disputed events instead of inventing facts, use clear headings (背景 / 经过 / 结果 / 历史评价), and write exact proper nouns so the app can auto-link them. (Installs upgraded from the older generic build have their stock `default_wiki_maintainer.md` migrated automatically; a customized one is preserved.)

You can manage prompts in two places:

- Sidebar `Prompt Setting`: quickly switch the active prompt.
- Main `Prompt Setting` tab: edit, save, create, reset, delete, and switch prompts.

Prompt files are plain Markdown. Create different prompts if you want different maintenance styles, such as dynasty-focused notes, biographies, or military-history timelines.

## Model Providers

The sidebar has a `Model source` switch:

- `Local model`: uses a local OpenAI-compatible server.
- `Online API`: lets you choose OpenAI GPT, Google Gemini, Anthropic Claude, JuAI/NewAPI, or a custom OpenAI-compatible provider.

Provider settings:

- `Base URL`: shown for OpenAI-compatible providers. Do not include endpoint paths such as `/chat/completions` or `/embeddings`.
- `API Key`: paste a provider key, save a key to `keys/`, or use an environment variable.
- `Model`: use a chat-capable model ID for the selected provider.
- `Temperature`: controls generation randomness.
- `Max output tokens`: used by Claude and available for providers that need an output cap.

Default models:

- Local OpenAI-compatible: `llama-3.2-1b-instruct`
- OpenAI GPT: `gpt-4o-mini`
- Google Gemini: `gemini-2.0-flash`
- Anthropic Claude: `claude-3-5-sonnet-latest`
- JuAI / NewAPI: blank by default; use `Test Provider` to list accessible models, then copy one into `Model` if needed.

## Local Model Setup

For LM Studio:

1. Open LM Studio and load a chat model.
2. Start the local OpenAI-compatible server.
3. In the app sidebar, choose `Local model`.
4. Use these default settings:

```text
Base URL: http://localhost:1234/v1
API Key: lm-studio
Model: llama-3.2-1b-instruct
```

If the server is on another machine, use that host instead:

```text
Base URL: http://192.168.114.1:1234/v1
```

The app also accepts a bare host URL such as `http://192.168.114.1:1234` and automatically treats it as `http://192.168.114.1:1234/v1`.

## Online API Setup

Choose `Online API`, then select a provider:

- `OpenAI GPT`: uses `https://api.openai.com/v1`.
- `Google Gemini`: uses Gemini's OpenAI-compatible endpoint.
- `Anthropic Claude`: uses Anthropic's native Messages API, so no Base URL is shown.
- `JuAI / NewAPI`: defaults to `https://www.juaiapi.com/v1` and supports New API connection JSON.
- `Custom OpenAI-compatible`: use any remote provider that exposes an OpenAI-compatible chat completions API.

Environment variable fallback names:

- OpenAI GPT: `OPENAI_API_KEY`
- Google Gemini: `GEMINI_API_KEY`
- Anthropic Claude: `ANTHROPIC_API_KEY`

## Saved API Keys

Use `Save Key` in the model configuration panel to store the current provider's API key. The app writes one JSON file per provider in `keys/`.

Example saved key file:

```json
{
  "provider": "OpenAI GPT",
  "api_key": "your-api-key"
}
```

The API key field also accepts New API connection JSON. Paste it into the API key field and press `Save Key`:

```json
{
  "_type": "newapi_channel_conn",
  "key": "sk-your-key",
  "url": "https://www.juaiapi.com"
}
```

The app saves `key` as the provider API key and uses `url` as the provider Base URL. A saved URL such as `https://www.juaiapi.com` is normalized to `https://www.juaiapi.com/v1` for OpenAI-compatible requests.

## Custom API Providers

For JuAI or another New API service:

1. Choose `Online API`.
2. Select `JuAI / NewAPI`.
3. Set `Base URL` to the provider root, for example `https://www.juaiapi.com`, or the `/v1` endpoint.
4. Paste either the raw API key or the New API JSON object into `API Key`.
5. Click `Save Key`.
6. Click `Test Provider` to check `/v1/models`.
7. If models are listed, copy an allowed chat model into `Model`, or leave `Model` blank to let the app try listed non-embedding models.

For any other OpenAI-compatible provider, choose `Custom OpenAI-compatible` and follow the same steps with that provider's Base URL.

Common saved key paths:

- `keys/local_openai-compatible.json`
- `keys/openai_gpt.json`
- `keys/google_gemini.json`
- `keys/anthropic_claude.json`
- `keys/juai_newapi.json`
- `keys/custom_openai-compatible.json`

When you switch providers, the app auto-loads the matching saved key if it exists. The API key lookup order is:

1. Key typed in the sidebar
2. Saved JSON key in `keys/`
3. Provider environment variable
4. Provider fallback, such as `lm-studio` for local servers

Saved `keys/*.json` files are ignored by `.gitignore`, but they are still plain text local secrets. Do not share them.

## Delete Pages

Open a page, expand `Delete Page` at the bottom, and type the exact page title to enable deletion. The app permanently removes the matching Markdown file from `wiki_data/`.

Back up `wiki_data/` if the page content matters.

## Troubleshooting

If AI actions fail:

- Confirm the selected provider is correct.
- Confirm the API key is entered, saved, or available through the provider environment variable.
- Confirm the model name is a chat-capable model for the selected provider.
- If a proxy says the token has no access to a model such as `gpt-5.4`, change the `Model` field to a model that token can access.
- For local servers, confirm the server is running.
- For OpenAI-compatible providers, confirm the Base URL ends at the API root, usually `/v1`.
- Do not use endpoint URLs such as `/v1/embeddings` or `/v1/chat/completions` as the Base URL.
- Do not use embedding models such as `text-embedding-nomic-embed-text-v1.5` for chat generation.
- If a local server is on another machine, confirm the IP address and firewall allow access to the configured port.

If you see a missing credentials error from the OpenAI SDK, enter any non-empty local key such as `lm-studio` for local servers, or save/set a real key for online providers.

## Project Structure

```text
.
+-- Start Raw LLM Wiki.bat  # Windows one-click launcher
+-- app.py          # Streamlit app
+-- requirements.txt # Python dependencies (core + optional add-ons)
+-- keys/          # Provider key storage; JSON secrets are ignored by git
+-- prmopt/        # Editable AI behavior prompts
+-- wiki_data/     # Markdown wiki pages (with YAML frontmatter)
+-- .gitignore
`-- README.md
```

## Notes

- Page titles are sanitized before being used as filenames, including Chinese/full-width punctuation.
- Pages are stored with YAML frontmatter (`title`, `dynasty`, `year`, `tags`); the body is what you see and edit below it.
- Draft generation is enabled only for empty pages to avoid overwriting existing content.
- The active prompt in `prmopt/` is prepended to AI system instructions.
- Local BERT NER is optional; without `transformers`/`torch` the app uses your LLM or a built-in gazetteer.
- The app writes and deletes files directly in `wiki_data/`.
- Saved API keys are local plaintext JSON files. Use environment variables instead if you do not want keys stored on disk.
