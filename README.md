# 📄 AI Resume ATS Analyzer

Upload a resume (PDF, DOCX or TXT) and get an **ATS score**, keyword gaps, prioritized
improvements and before/after bullet rewrites. Built with **Streamlit** and **Google Gemini (Flash)**.

## How the score works

| Part | Weight | What it measures |
|------|--------|------------------|
| Rule-based checks | 40% | Contact info, standard sections, length, bullets, numbers/metrics, action verbs (deterministic, no AI) |
| Gemini evaluation | 60% | Keyword relevance, structure, impact and clarity (tailored to a job description if you paste one) |

> No tool can reproduce a real company's ATS exactly. Treat the score as a guide for improving
> your resume, not as a guarantee.

## Features
- PDF / DOCX / TXT upload (max 5 MB)
- Optional job description for tailored keyword matching
- Prioritized improvements (High / Medium / Low)
- Matched vs missing keywords
- Rewrite suggestions for weak bullet points
- "Parsed text" view showing exactly what an ATS can read
- Downloadable Markdown report

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey, then either paste it in the
sidebar or save it as a secret (recommended).

### Store the key locally (optional)
Create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-3.6-flash"   # optional override
```

## Push to GitHub

Create a `.gitignore` containing:

```
venv/
__pycache__/
.streamlit/secrets.toml
.env
```

Then:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: AI Resume ATS Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repository on github.com first, without a README, then run the commands above.)

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Click **Deploy**. You get a public URL in a minute or two.

To update the app later, just `git push`; Streamlit redeploys automatically.

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| `GEMINI_API_KEY` | secrets, env var or sidebar | none (required) |
| `GEMINI_MODEL` | secrets, env var or sidebar | `gemini-flash-latest` |

The default `gemini-flash-latest` is Google's alias that always points to the newest Flash model.
If a model returns 404 / "no longer available" (Google retires models regularly), the app automatically
tries `gemini-3.6-flash`, `gemini-3.8-flash`, `gemini-3.5-flash-lite` and `gemini-2.5-flash` in turn and
shows which model was used. To change the list, edit `FALLBACK_MODELS` at the top of `app.py`.
Current model names: https://ai.google.dev/gemini-api/docs/models

## Troubleshooting

- **"Almost no text could be extracted"**: the PDF is a scanned image. Export a text-based PDF or DOCX.
- **"Invalid or unauthorized API key"**: check the key and that the Gemini API is enabled for it.
- **"None of these models are available"**: enter a current Flash model name from the docs link above in the sidebar.
- **Rate limit (429)**: the app retries automatically; if it persists, wait a minute.

## Privacy
Resume text is sent to the Gemini API for analysis. The app itself stores nothing.
Don't deploy it publicly with *your* API key if you don't want others to use your quota.

## Project structure
```
app.py            # the Streamlit app
requirements.txt  # dependencies
README.md
```
