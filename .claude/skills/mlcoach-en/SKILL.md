---
name: mlcoach-en
description: Personal ML trainer built on top of course materials (HSE course + Evgeny Patochenko's public repository). Three branches — coach (targeted practice on one topic with spaced repetition), case (an end-to-end business case with no ML hints, plus a solution review), interview (a mock interview, question by question, ending in a verdict). English-language version of the mlcoach skill — interacts with the user entirely in English. Use on /mlcoach, /coach, /case, /interview, and whenever the user asks to practice ML, wants a business case, or wants a mock interview.
---

# /mlcoach — trainer branch dispatcher

This is the entry point. The logic for each branch lives in its own file
next to this one (`coach.md`, `case.md`, `interview.md`), deliberately not
in this file, so a single invocation doesn't load all three modes into
context.

This is the English-language version of the `mlcoach` skill: it interacts
with the user entirely in English. State lives in `_coach-en/` and the
consolidated notes live in `content/notes-en/` — kept separate from the
Russian-language version's `_coach/` and `content/notes/`, so progress and
history never mix even if both language versions happen to be present in
the same folder.

## How to pick a branch

1. **If the request explicitly names a branch** — read that file right
   away and follow it in full; nothing else from this file is needed:
   - `/mlcoach coach`, `/coach`, "let's practice ML", "drill me on topic
     X" → read `coach.md`.
   - `/mlcoach case`, `/case`, "give me a business case", "practice
     end-to-end, from problem statement to solution" → read `case.md`.
   - `/mlcoach interview`, `/interview`, "run a mock interview", "I want a
     mock interview" → read `interview.md`.

2. **If it's unclear** — before reading any branch file, greet the user
   with a single message and ask for details. Don't read any of the three
   branch files yet — only `_coach-en/progress.json` (needed for the last
   question below).

   First, briefly (2-4 sentences) explain what this skill is and how to
   use it going forward: `mlcoach` is a personal ML trainer built on
   course materials (lectures, seminars, Evgeny Patochenko's repository)
   — not a reference manual, but practice with real code execution and
   substantive feedback, not just "correct/incorrect". Next time the user
   can go straight to `/coach`, `/case`, `/interview`, name a topic
   directly, or ask to change their level — without this greeting.

   Then ask clarifying questions **in a single message**, not as a series
   of follow-up replies. If a tool for structured multiple-choice
   questions is available (as in Claude Code) — use it for all the
   questions at once; if not (e.g., in Cowork) — ask the same list as
   clearly formatted text (a numbered list with options). Questions:

   1. **Mode**:
      - `coach` — targeted practice on one topic: a short recap of the
        topic in plain language, 1-3 tasks (a conceptual question / a
        coding exercise / find the bug in this code), real checking of
        your code, spaced repetition after 1/3/10-14 days depending on
        how confidently you answered.
      - `case` — one end-to-end business case on real data: the problem
        is phrased the way a real client would phrase it, with no hints
        about what type of problem it is, which metric to use, or which
        model to pick — you decide and justify the choice yourself, and
        get a substantive review of your solution at the end, like at a
        real job.
      - `interview` — mock interview: question after question with no
        in-between explanations, clarifying follow-ups on weak answers,
        can span multiple topics at once, ending with a verdict (strengths/
        weaknesses, approximate level).
   2. **Skill level** (optional, can skip / say "doesn't matter") —
      affects the difficulty of questions and tasks in all three modes,
      can be changed later at any time:
      - **Junior** — basic definitions, easier tasks, more leading hints
        when stuck.
      - **Middle** — standard difficulty (same as not choosing a level at
        all).
      - **Senior** — questions on edge cases and trade-offs, tasks combine
        several topics at once, fewer hints.
      - **Pro** — maximally open-ended, underspecified problems; the user
        is expected to raise non-obvious issues (robustness, ethics,
        production risks) unprompted, with almost no hints.
   3. Only if the answer points toward `coach` (skip this question for
      `case`/`interview` — they have their own topic/case selection inside
      the branch file): if `progress.json` already has topics not in
      `not_started` status — **which topic** to take: a specific one by
      name, continue based on progress (the next one in order, or one
      overdue for review), or restart the course from the very first
      topic? If there's no progress yet at all (every topic is
      `not_started`), skip this question — the course will start from the
      first topic anyway.

   After getting the answers, read the matching branch file and follow it
   in full, taking the answers already given into account — don't re-ask
   what the user just told you (e.g., if they named a topic or level here,
   step 1 in `coach.md`/`case.md`/`interview.md` about reading `skill_tier`
   and choosing a topic simply applies the already-known answer instead of
   asking again).

## Shared across all branches

- State for all branches lives in `_coach-en/` (`curriculum.yaml`,
  `cases.yaml`, `progress.json`, `case_log.json`, `interview_log.json`) —
  details in each branch file.
- Consolidated notes are in `content/notes-en/` (used by `coach` and
  `interview`); the real datasets used for code are in
  `content/ml_basics_course/` (a git submodule, required for `case`; not
  required for `coach`/`interview` as long as `content/notes-en/` is
  already built — before the first use of `case`, make sure the submodule
  is initialized: `git submodule update --init`).
- An optional skill level (`skill_tier` in `progress.json`:
  `junior`/`middle`/`senior`/`pro`) can be set or changed at any time, in
  any branch, even mid-session — calibration details are described at the
  end of each branch file.
