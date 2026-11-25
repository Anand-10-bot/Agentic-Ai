# Environment-Anchored Monitoring (EAM): pilot

Code and results for the blog post
**[You Can Split an Attack Across Chats, but Not Across Databases](BLOG_URL)**
(CS G569 Agentic AI, BITS Pilani).

An attacker can split a harmful job into harmless-looking requests and send them
from different chats, accounts or AI services, so no single conversation shows
the attack. Every step still has to change the same company systems. EAM ignores
the requests and asks an LLM judge to read only a log of what changed. This
pilot compares it with monitors that read the requests, on 60 synthetic test
cases.

## What's here

| File | What it does |
|---|---|
| `schema.py` | The event log format: one line per change (`Event`), one task per `Trace` |
| `build_dataset.py` | Generates the 60 test cases: 30 attacks and 30 legitimate twins that touch the same systems, in six attack types with five pairs each (pairs 4 and 5 are deliberately hard) |
| `monitors.py` | The judge prompt, the judge back-ends (Groq, Gemini, Anthropic, mock) and the five monitors |
| `run_experiments.py` | Runs detection (E1) and live step-by-step monitoring (E3), and writes a blind-labelling sheet (E2) |
| `traces/dataset.json` | The 60 test cases used in the post (`build_dataset.py` recreates it exactly) |
| `results/` | The outputs behind every number in the post |

The five monitors:

| Name in code | Name in the post | Reads |
|---|---|---|
| M1 | Per-step filter | each request on its own |
| M2 | Full-chat monitor | all of a task's requests together |
| M3 | EAM | only the event log |
| M3raw | EAM, stripped | the event log minus seven opinion-like attributes |
| M4 | Rules | five hand-written if-then rules, no LLM |

## Reproduce the main result

```bash
pip install -r requirements.txt
python build_dataset.py                      # writes traces/dataset.json
export GROQ_API_KEY=...                      # PowerShell: $env:GROQ_API_KEY="..."
export GROQ_MODEL=openai/gpt-oss-120b        # PowerShell: $env:GROQ_MODEL="openai/gpt-oss-120b"
python run_experiments.py --judge groq
```

This makes a few hundred calls and fits in Groq's free tier, but takes about an
hour because it pauses between calls. Every reply is saved under `results/`, so
an interrupted run picks up where it stopped. The judge runs at temperature 0,
but the API does not guarantee identical replies, so a re-run can differ by a
case or two.

`python run_experiments.py --judge mock` checks the pipeline without an API key.
Its numbers are fake.

## Results

From `results/e1_detection_groq.csv` (judge: gpt-oss-120b on Groq):

| Monitor | Attacks caught | Legitimate jobs flagged | AUROC |
|---|---|---|---|
| Per-step filter | 17 / 30 | 5 / 30 | 0.79 |
| Full-chat monitor | 19 / 30 | 0 / 30 | 0.82 |
| **EAM** | **26 / 30** | **10 / 30** | **0.85** |
| EAM, stripped | 23 / 30 | 8 / 30 | 0.82 |
| Rules | 18 / 30 | 4 / 30 | n/a |

Run live after every step (`results/e3_timing_groq.csv`), EAM flagged 27 of 30
attacks at or before the step that does the damage and never caught the other
3. It also raised a false alarm on 13 of 30 legitimate jobs at some step.

## Limitations

- All 60 cases are synthetic and were written by us, knowing which ones were
  attacks. No real agent or real system was involved.
- One judge model.
- With 30 cases on each side, a gap of a few cases can be chance.
- The rules baseline (M4) has a known bug: its data-theft rule checks
  `approved=False`, but the upload events use `on_allowlist`, so that rule never
  fires. It is left as is so the numbers match the post.

## Team

Ojaswi Prakash, Anand Bajaj Gupta and Dhruv Goel, with Faculty Name (faculty)
and Mentor Name (mentor). CS G569 Agentic AI, BITS Pilani.

Code written with help from Claude (Anthropic) and checked by the authors.
