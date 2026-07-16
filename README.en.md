# ml-coach

🇷🇺 **[Читать на русском →](README.md)** | 🇬🇧 English (current)

---

A personal ML trainer built on top of a machine learning course (HSE,
taught by Evgeny Patochenko and Zhanna Smelkova) and the public repository
[github.com/evgpat/ml_basics_course](https://github.com/evgpat/ml_basics_course).

The idea is simple: instead of rewatching lectures and doing exercises on
your own, you open this folder in [Claude Code](https://docs.claude.com/claude-code)
(or install the skill in Claude Cowork — see below) and say `/coach`,
`/case`, or `/interview` — and Claude pulls in the right course material,
asks questions and gives coding exercises based on your progress, actually
runs your code to check it, and keeps track of what needs review.

No server, no account, no cloud — all state (what you've already covered,
what's due for review, your case and interview history) lives in plain
files on your own disk, right in this folder.

This is the English-language edition of the project: the trainer speaks
entirely in English and reads fully translated course notes, while the
underlying course was originally taught in Russian at HSE.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Modes](#modes)
  - [`coach` — targeted practice](#coach--targeted-practice)
  - [`case` — a full business case](#case--a-full-business-case)
  - [`interview` — mock interview](#interview--mock-interview)
- [Skill level (Junior/Middle/Senior/Pro)](#skill-level-juniormiddlesniorpro)
- [Repository structure](#repository-structure)
- [About the materials, translation, and anonymization](#about-the-materials-translation-and-anonymization)
- [Using it in Claude Cowork](#using-it-in-claude-cowork)
- [Resetting progress / forking without your history](#resetting-progress--forking-without-your-history)

## How it works

This isn't a standalone app — it's a **Claude Code Skill**: a set of
markdown files with instructions (`.claude/skills/mlcoach-en/`) that Claude
Code automatically picks up when you open this folder. When you type
`/coach`, `/case`, or `/interview`, Claude Code has Claude read the
matching instruction file and follow it: read your topic map and progress,
pull in the right notes, ask questions, wait for your answer, actually run
your code (Claude Code has terminal access), grade the solution, and update
the progress file.

Schematically:

```
You type /coach
        │
        ▼
Claude reads _coach-en/curriculum.yaml (topic map) and _coach-en/progress.json (your progress)
        │
        ▼
Picks a topic (a new one in increasing order of difficulty, or one overdue for review)
        │
        ▼
Reads the ready-made notes for the topic from content/notes-en/<topic>.md
        │
        ▼
Gives a recap in its own words + 1-3 tasks (question / code / find the bug)
        │
        ▼
You answer → Claude actually runs your code and checks it against the reference
        │
        ▼
Updates _coach-en/progress.json: when to review this topic next
```

Key detail: the course material is pre-assembled into ready-made notes
(`content/notes-en/`) — one per topic, merging the lecture, the seminar,
and material from Evgeny Patochenko's repository into a single readable
text with formulas explained in plain language instead of raw LaTeX. This
was done once, up front, so the trainer session itself doesn't burn time
and tokens re-parsing PDFs and notebooks every single time — Claude just
reads the finished file.

## Requirements

Works two ways:

- **[Claude Code](https://docs.claude.com/claude-code)** (CLI, IDE
  extensions, desktop app, web app at `claude.ai/code`) — the primary,
  fully supported path. The skill is picked up automatically from
  `.claude/skills/`, nothing else needs to be installed.
- **Claude Cowork** (the desktop app's mode for working with a local
  folder) — via uploading a ready-made skill archive, see
  [Using it in Claude Cowork](#using-it-in-claude-cowork) below.

Either way, you need the ability to run Python code (`sklearn`, `pandas`,
etc.) — both Claude Code and Cowork can execute code in their own
environment.

**Won't work:**
- in the regular **Claude** chat app (claude.ai) without Cowork — that's a
  different product, it doesn't read `.claude/skills/` and doesn't support
  uploading custom skills;
- in the Claude mobile app.

## Installation

1. Make sure you have [Claude Code](https://docs.claude.com/claude-code)
   installed (`npm install -g @anthropic-ai/claude-code` for the CLI, or
   download the desktop app / install the IDE extension — see the official
   docs at the link above).

2. Clone this repository **with its submodule** (this matters — without
   the flag below, part of the material won't download):

   ```bash
   git clone --recurse-submodules https://github.com/Andryuha40/ml-coach.git
   cd ml-coach
   ```

   If you already cloned without the flag, fetch the submodule separately:

   ```bash
   git submodule update --init
   ```

   The submodule (`content/ml_basics_course/`, Evgeny Patochenko's original
   repository) is required only for the `case` mode — that's where the
   real datasets used to run code live. It's not required for `coach` and
   `interview`: all the material they need is already collected in
   `content/notes-en/`.

3. Open the `ml-coach` folder in Claude Code (`claude` in a terminal from
   this folder, or "Open Folder" in the desktop app/IDE extension).

4. Type `/mlcoach`, `/coach`, `/case`, or `/interview` — and get started.

No extra setup, API keys beyond your regular Claude Code subscription, or
sign-up is required.

## Modes

All three modes are implemented as branches of one skill, `mlcoach`
(`.claude/skills/mlcoach-en/`). You can call the dispatcher (`/mlcoach`)
and it will ask which mode you want, or you can name the command directly.

### `coach` — targeted practice

```
/coach
```

The trainer decides what to work on today:
- if there's a topic you covered recently and it's due for review (after 1,
  3, or 10-14 days, depending on how confidently you answered last time) —
  it'll suggest that one;
- otherwise — it picks the next new topic by difficulty (easy to hard:
  fundamentals first, then metric-based methods and trees, then ensembles
  and features, then advanced topics like interpretability and time
  series, with LLMs/agents/RAG last).

You can also name a topic yourself ("drill me on SVM").

Then Claude:
1. Briefly recaps the topic in its own words — and not a generic summary,
   specifically what you'll need for the tasks below.
2. Gives 1-3 tasks: a conceptual question ("why...", "what breaks if..."),
   a coding exercise (a real course code fragment with the implementation
   stripped out — write it yourself), or "find and explain the bug" (working
   code with a deliberate mutation — a bug).
3. Waits for your answer. If it's code, it actually runs it and compares
   the behavior/metric against the reference, showing the traceback if it
   fails.
4. Gives substantive feedback (not just "correct/incorrect") and updates
   your progress.

### `case` — a full business case

```
/case
```

A completely different format: instead of leading questions, you get a
problem phrased the way a real client would phrase it — with no
indication of what type of problem it is (regression? classification?),
which metric to use, or which model to pick. You have to work all of that
out yourself: understand the brief, choose an approach, write the code,
justify your metric choice.

Then comes a review in the spirit of a code review at a real job: does the
chosen metric match the cost asymmetry built into the case (e.g., missing
a sick patient is worse than one extra false alarm for a healthy person —
or the other way around), is the methodology clean (no data leakage,
train/test split done correctly, especially for time series data), did you
notice ethically sensitive aspects where the case has them (e.g., using a
feature that correlates with race or gender).

The repository currently has 11 cases across different domains: real
estate, credit scoring, quality control, marketing, commodity pricing,
social program allocation, medical screening, biotech (multiclass
classification), food safety, customer segmentation (clustering), spam
filtering (NLP).

### `interview` — mock interview

```
/interview
```

A simulated technical interview: question after question, with no
introductory explanations (Claude asks right away instead of giving a
mini-lecture), with clarifying follow-ups if an answer is shallow — a real
interviewer doesn't silently accept the first roughly-correct answer, they
dig deeper. It can span several topics in one session (you choose — one
topic, one area, or "you pick"). At the end — an honest, structured
verdict: what's strong, what's weak, an approximate level based on this
specific interview, and which topics are worth reviewing via `/coach`.

## Skill level (Junior/Middle/Senior/Pro)

You can optionally set a skill level — all three modes will then adjust
the difficulty of questions and tasks to match:

- **Junior** — basic definitions, more leading hints, easier tasks with a
  solution-structure template.
- **Middle** — the standard default level (if no level is set, everything
  works exactly as before, unchanged).
- **Senior** — questions on edge cases and trade-offs, tasks combine
  several topics at once, fewer hints.
- **Pro** — maximally open-ended, underspecified tasks, expecting you to
  raise non-obvious issues yourself (ethics, robustness, production risks)
  with no leading questions.

You can set or change the level at any time, just by saying so ("let's do
Senior level", "I'm more of a Junior right now") — even in the middle of a
session already in progress, not just at the start. The level persists
across sessions (in `_coach-en/progress.json`) until you explicitly ask to
change it.

## Repository structure

```
ml-coach/
├── .claude/skills/mlcoach-en/
│   ├── SKILL.md              # dispatcher: figures out which branch is needed
│   ├── coach.md                # full logic for the coach mode
│   ├── case.md                   # full logic for the case mode
│   └── interview.md               # full logic for the interview mode
├── content/
│   ├── lectures/               # lecture slide PDFs 01-10 + text transcripts 11-19 (source material, in Russian)
│   ├── seminars/                # seminar notebooks + text transcripts (source material, in Russian)
│   ├── notes-en/                 # ready-made consolidated notes per topic, in English
│   ├── extra_datasets/            # extra UCI datasets for cases outside Evgeny's repository's topics
│   └── ml_basics_course/          # git submodule → github.com/evgpat/ml_basics_course
└── _coach-en/
    ├── curriculum.yaml          # map of 20 topics: sources, notes, concepts, prerequisites
    ├── cases.yaml                 # catalog of 11 business cases for `case`
    ├── progress.json               # your progress per topic + skill level
    ├── case_log.json                # history of completed cases
    └── interview_log.json            # history of interviews
```

`_coach-en/curriculum.yaml` is a map of 20 topics, from fundamentals
(intro to ML, linear regression) through metric-based methods, trees and
ensembles, feature selection and interpretability, to advanced topics
(clustering, time series, transformers, LLMs/agents/RAG — this last block
is specific to this particular course; it isn't in Evgeny Patochenko's
repository). Plus an optional capstone module with two end-to-end business
cases on real data.

## About the materials, translation, and anonymization

- Lecture slide PDFs 01-10 and text transcripts of lectures 11-19/seminars
  10-18 are original course materials — in Russian, since that's the
  language the course was taught in. Live-session transcripts were
  anonymized before publication: mentions of specific students and their
  personal stories were removed or de-identified (rephrased into generic
  "food for thought" questions); the technical content was left unchanged.
- `content/notes-en/` — one note per topic, assembled from Evgeny
  Patochenko's repository (as the base) plus unique local course material
  (as an addition — e.g. the LLM/agents block, which isn't in the
  repository at all) into a single file, **and then translated into
  English in full** — this is the English-language edition's own layer on
  top of the Russian-language `content/notes/`. The goal, as with the
  Russian version, is for `coach` and `interview` to never re-parse a
  PDF/notebook from scratch during a session; the goal of the translation
  specifically is to let English speakers use the trainer without needing
  to read Russian source material.
- `content/ml_basics_course/` is not a copy but a git submodule pointing at
  Evgeny Patochenko's original repository (MIT license) — the underlying
  lesson PDFs and notebooks it contains are in Russian and were not
  translated; to update from upstream, just update the submodule
  (`git submodule update --remote`).
- `content/extra_datasets/` — datasets from the [UCI Machine Learning
  Repository](https://archive.ics.uci.edu/) for `case` scenarios not
  covered by datasets from Evgeny's repository (clustering, multiclass
  classification, NLP) — sources and descriptions in
  `content/extra_datasets/README.md`.
- `_coach-en/curriculum.yaml` and `_coach-en/cases.yaml` reference the
  original Russian-language lecture/seminar/lesson files as sources (their
  filenames are untranslated, since those are real files on disk) — only
  the topic titles, concepts, and case briefs shown to you during a
  session are in English.

## Using it in Claude Cowork

The Claude desktop app has a **Cowork** mode — working with a selected
local folder — and a separate **Settings → Skills** section where you can
upload a custom skill (a zip archive with `SKILL.md` inside). This is a
separate product from Claude Code, but it turned out to be compatible in
practice.

**1. Download the self-contained archive** — no need to rebuild anything
or clone the repository separately, the archive already has everything for
all three modes: the skill logic, `curriculum.yaml`/`cases.yaml`, all 24
consolidated English notes (`content/notes-en/`), and 12 datasets (8 from
Evgeny Patochenko's repository + 4 from UCI) that the `case` business cases
are built on (≈2.4 MB):

**[⬇ Download mlcoach-standalone-en.zip](https://raw.githubusercontent.com/Andryuha40/ml-coach/main/mlcoach-standalone-en.zip)**

**2. Unpack the archive** into any folder on disk — this becomes your
working folder, the equivalent of a cloned repository (inside, it has the
same structure: `_coach-en/`, `content/notes-en/`,
`content/ml_basics_course/datasets/`, `content/extra_datasets/`).

**3. Upload the skill.** In the Claude app: **Settings → Skills → Add →
Upload a skill** → pick the `SKILL.md` file from the unpacked folder (or
the archive itself — depends on what the uploader accepts; if that
doesn't work, try re-zipping just the unpacked folder as a whole).

**4. Open Cowork and pick the unpacked folder as your working folder.**
This matters: `_coach-en/curriculum.yaml` and the other files are read
relative to the working folder's path.

**5. Type** `/coach`, `/case`, or `/interview` in your Cowork session.

> To be honest, I haven't actually verified in practice which of steps 3-4
> Cowork really requires — maybe uploading in Settings is enough for the
> skill files (`SKILL.md` etc.), and it only sees the data (`_coach-en/`,
> `content/`) through the selected working folder; maybe it's the other
> way around, and all the archive's contents become available right after
> the Skills upload, with no need to pick a folder. Do both steps (3 and
> 4) — if one turns out to be unnecessary, nothing breaks.

Cowork can execute code, so checking coding exercises should work too.
This isn't an officially supported path (unlike Claude Code), so if you
run into problems, switch to Claude Code — everything is guaranteed to
work there.

If you only need the skill itself, without the data (e.g., you already
cloned the whole repository via `git clone` and just want to install the
skill separately in Cowork) — there's a lighter
**[mlcoach-en.zip](https://raw.githubusercontent.com/Andryuha40/ml-coach/main/.claude/skills/mlcoach-en.zip)**
(13 KB, only `SKILL.md`/`coach.md`/`case.md`/`interview.md`, no data — in that
case your Cowork working folder should be the cloned repository).

If you forked the repository and changed `.claude/skills/mlcoach-en/`,
`_coach-en/curriculum.yaml`, or the notes — you'll need to rebuild both
archives yourself and commit them again.

**Important — don't use Windows PowerShell's `Compress-Archive`.** The
Cowork skill uploader rejects an archive with the error "Zip file contains
path with invalid characters" if the paths inside are separated with a
backslash (`\`) instead of `/`, as the ZIP format requires — and that's
exactly how `Compress-Archive` in Windows PowerShell 5.1 writes paths
(this is a known bug specific to that command; `zip`/`Compress-Archive` in
PowerShell 7+ on other systems may behave differently, but it's best not
to risk it). Build the archive with one of the methods below — both are
guaranteed to write `/`:

macOS/Linux/WSL:

```bash
cd .claude/skills
zip -r mlcoach-en.zip mlcoach-en/
```

Python (works the same on any OS, including plain Windows PowerShell — it
doesn't rely on `Compress-Archive`):

```python
import zipfile, os

def zip_dir(src_root, arc_prefix, out_path):
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(src_root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, src_root).replace(os.sep, "/")
                zf.write(full, arc_prefix + "/" + rel)

zip_dir(".claude/skills/mlcoach-en", "mlcoach-en", ".claude/skills/mlcoach-en.zip")
```

For `mlcoach-standalone-en.zip` — assemble a folder with the same
structure as `ml-coach/`, but only with the files needed for
`coach`/`case`/`interview` (`.claude/skills/mlcoach-en/*.md` → into the
root; `_coach-en/curriculum.yaml`, `_coach-en/cases.yaml`, and empty
`progress.json`/`case_log.json`/`interview_log.json` templates;
`content/notes-en/`; the datasets from `cases.yaml` — every file listed in
a `dataset` field, at its own path
(`content/ml_basics_course/datasets/...` or
`content/extra_datasets/...`)), and zip it the same way (`zip_dir(...)`
from the example above).

Check the archive for backslashes before publishing it (should print
`entries with backslash: 0`):

```python
import struct
with open("archive.zip", "rb") as f:
    data = f.read()
pos = 0
while (idx := data.find(b"PK\x01\x02", pos)) != -1:
    n = struct.unpack("<H", data[idx+28:idx+30])[0]
    name = data[idx+46:idx+46+n]
    if b"\\" in name:
        print("BACKSLASH:", name)
    pos = idx + 4
```

## Resetting progress / forking without your history

`_coach-en/progress.json`, `_coach-en/case_log.json`, and
`_coach-en/interview_log.json` are your personal progress. If you want to
share your fork of the repository with someone else without your history —
reset them to the empty template (each file has a `_comment` explaining
its structure) before publishing.
