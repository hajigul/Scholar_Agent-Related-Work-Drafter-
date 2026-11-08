# Scholar_Agent-Related-Work-Drafter-


A free, local tool that drafts the **related-work section** of a research paper. Paste a topic, title, or abstract, and it finds papers on arXiv, writes a short cited paragraph with a small local language model, and builds the reference list from real arXiv data.

No API keys, no accounts, no cost. Everything runs on your computer with [Ollama](https://ollama.com).

![Home screen](docs/rw-home.png)

![Draft with papers](docs/rw-draft.png)

## How it works

```
Your topic / abstract
        |
        v
 LLM proposes 3 search phrases (abbreviations spelled out)
        |
        v
 arXiv search for each phrase  -->  duplicates removed, weak matches ranked last
        |                           (keyword fallback if nothing is found)
        v
 LLM writes a ~150-200 word paragraph citing [1], [2], ...
        |
        v
 Python builds the reference list from the arXiv data
 and flags citation numbers that do not exist
```

The key design choice: **the model never writes the reference list.** Titles, authors, years and links come straight from arXiv, so they are always real. Only the paragraph is model-written.

## Features

- **Search planning.** The model turns your text into 3 short search phrases and expands abbreviations such as "KG" into "knowledge graph".
- **Fallback search.** If the planned searches find nothing, it searches with keywords taken from your own text.
- **Untick and rewrite.** Remove papers that don't fit, then rewrite the paragraph using only the ticked papers. Citations are renumbered.
- **Clickable citations.** Clicking `[3]` in the draft scrolls to and highlights paper 3.
- **Checks.** Warns if the draft cites a number that is not in the reference list, and lists papers that were found but never cited.
- **Search log.** A collapsible log shows what each query returned or which error it hit.
- **Export.** Copy the text, the references, or BibTeX, or download everything as Markdown.
- **Command line and web app**, dark and light themes, model picker, streaming output.

## Quick start

**Requirements:** Python 3.9+ and [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/hajigul/Related-Work-Drafter.git
cd Related-Work-Drafter

python3 -m venv agent-env
source agent-env/bin/activate        # Windows: agent-env\Scripts\activate
pip install -r requirements.txt

ollama pull llama3.2:3b
python rw_server.py
```

Open http://127.0.0.1:8001 (it opens automatically). Keep the Ollama app running.

### Command line

```bash
python related_work.py "noisy knowledge graph refinement with large language models"
```

This prints the draft and saves it as `related_work_<date>.md`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SCHOLAR_MODEL` | `llama3.2:3b` | Ollama model used for planning and writing |

Cache lifetime and model keep-alive are set in `config.py`. A larger model (for example `llama3.1:8b`) usually writes better paragraphs but is slower.

## Project structure

```
Related-Work-Drafter/
├── related_work.py   # engine: query planning, search, drafting, checks, CLI
├── rw_server.py      # local web server (streams events to the page)
├── rw.html           # front end
├── tools.py          # arXiv search with caching and 3-second request pacing
├── config.py         # settings
├── requirements.txt
└── docs/             # screenshots
```

## Limitations

- **The paragraph can be wrong.** A small local model can overstate what a paper did or write generic sentences. It only sees each paper's abstract snippet, never the full text. Check every claim against the linked papers.
- **This is a starting point, not a finished section.** Search is arXiv only, so journal papers and anything not on arXiv will be missing. Add those yourself.
- **Relevance is a keyword check.** The "on topic" badge means the topic words appear in the title or abstract, not that the paper truly fits. That is why the untick-and-rewrite control exists.
- **Search quality depends on the model's search phrases.** If results are poor, check the search log and try rewording your input.
- **No formal evaluation yet.** I have not measured retrieval or writing quality on a labeled test set.
- arXiv requests are paced 3 seconds apart, as the arXiv API terms request, so each draft takes a while.

## Security note

The server listens on `127.0.0.1` only. It has no login, so do not expose it to the public internet.

## Roadmap

- Evaluation set with hand-labeled relevance
- Search OpenAlex as well, for published venues
- Read full text from PDFs instead of abstract snippets only
- Export citations to a `.bib` file with stable keys

## Author

**Haji Gul**, PhD candidate (AI), Universiti Brunei Darussalam.
Research interests: knowledge graph completion, link prediction, LLM-augmented graph reasoning.

GitHub: [@hajigul](https://github.com/hajigul)

## License

MIT License. See `LICENSE`.

## Acknowledgments

[Ollama](https://ollama.com) and the [arXiv API](https://info.arxiv.org/help/api/index.html).
