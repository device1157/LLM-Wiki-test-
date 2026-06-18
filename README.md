# Raw LLM Wiki

Raw LLM Wiki is a small Streamlit app for maintaining a local Markdown wiki with optional AI help. Pages are stored as plain `.md` files in `wiki_data/`, and provider API keys can be saved as local JSON files in `keys/`.

## Features

- Create, browse, edit, and delete local Markdown wiki pages.
- Store pages as flat files under `wiki_data/`.
- Switch between a local model server and online API providers in the sidebar.
- Use OpenAI-compatible providers, including LM Studio, OpenAI GPT, Google Gemini, and custom endpoints.
- Use Anthropic Claude through the Anthropic Messages API.
- Save API keys per provider as JSON files under `keys/`.
- Import New API connection JSON with `_type`, `key`, and `url`.
- Auto-load the saved API key when switching providers.
- Manage reusable AI behavior prompts in `prmopt/`.
- Generate drafts for empty pages.
- Add AI summaries to existing pages.
- Suggest related wiki topics and create or open them quickly.
- Automatically link mentions of existing page titles while rendering Markdown.

## Requirements

- Python 3.10 or newer
- Streamlit
- OpenAI Python SDK
- Anthropic Python SDK, required only when using Claude
- Optional: a local OpenAI-compatible LLM server, such as LM Studio or an Ollama OpenAI-compatible server

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install streamlit openai anthropic
```

Run the app:

```powershell
streamlit run app.py
```

Open the URL Streamlit prints in the terminal, usually `http://localhost:8501`.

## Basic Workflow

1. Use `Create New Page` in the sidebar to create a wiki page.
2. Select a page from the sidebar page list.
3. Turn on `Edit Page` to write Markdown manually.
4. For empty pages, use `Generate Draft` in the `AI Agent` panel.
5. Use `Summarize This Page` or `Suggest Related Topics` on an existing page.
6. Use `Delete Page` at the bottom of a page when you want to remove it.
7. Use the `Prompt Setting` tab to edit how the AI maintains the wiki.

## Prompt Setting

The app stores AI behavior prompts as Markdown files in `prmopt/`. The active prompt is read before every AI task, including draft generation, summaries, and related-topic suggestions.

The default prompt is `prmopt/default_wiki_maintainer.md`. It teaches the AI to maintain a persistent, interlinked Markdown wiki by preserving useful content, using clear headings, reusing exact page titles for links, suggesting related pages, and avoiding secrets in generated content.

You can manage prompts in two places:

- Sidebar `Prompt Setting`: quickly switch the active prompt.
- Main `Prompt Setting` tab: edit, save, create, reset, delete, and switch prompts.

Prompt files are plain Markdown. Create different prompts if you want different wiki-maintenance styles, such as research notes, course notes, project documentation, or personal knowledge management.

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
+-- app.py          # Streamlit app
+-- keys/          # Provider key storage; JSON secrets are ignored by git
+-- prmopt/        # Editable AI behavior prompts
+-- wiki_data/     # Markdown wiki pages
+-- .gitignore
`-- README.md
```

## Notes

- Page titles are sanitized before being used as filenames.
- Draft generation is enabled only for empty pages to avoid overwriting existing content.
- The active prompt in `prmopt/` is prepended to AI system instructions.
- The app writes and deletes files directly in `wiki_data/`.
- Saved API keys are local plaintext JSON files. Use environment variables instead if you do not want keys stored on disk.
