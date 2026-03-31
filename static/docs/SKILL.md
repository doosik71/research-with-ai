---
name: arxiv-paper-analysis-report
description: write a detailed paper analysis report in korean for arxiv papers when the user asks for a detailed paper analysis, section-by-section breakdown, method explanation, experiment summary, or critical reading of a paper given by title or url. use this when chatgpt should find the paper on arxiv, prefer ar5iv html over direct pdf text extraction, identify major sections, write a markdown report that is understandable without reading the original paper, tag equations with $ or $$, and finish with markdown lint.
---

# ArXiv Paper Analysis Report

## Overview

Produce a detailed Korean markdown report for an arXiv paper from a paper title or URL. Prefer the ar5iv HTML rendering of the paper body before falling back to direct PDF text extraction, then identify the main sections and explain the paper in enough detail that the user can understand it without reading the original paper.

## Input Handling

Accept either of the following user inputs:

- paper title
- URL for the paper page or PDF

Normalize the input first:

- if the user provides a PDF URL, recover the canonical arXiv abstract URL when possible
- if the user provides an arXiv URL, extract the arXiv identifier
- if the user provides a paper title, locate the matching arXiv paper first

## Retrieval Workflow

Follow this workflow in order.

1. Locate the target paper on arXiv.
2. Open the arXiv abstract page and verify the paper title, author list, year, and canonical arXiv URL.
3. Build the ar5iv HTML URL using the arXiv identifier in the form `https://ar5iv.labs.arxiv.org/html/<arxiv_id>`.
4. Prefer reading the paper from the ar5iv HTML page.
5. If ar5iv is unavailable, incomplete, or malformed, fall back to the arXiv PDF and extract text from the PDF.
6. Identify the major sections from the paper body, including at least abstract, introduction, method, experiments, and conclusion when present.
7. Write the report in markdown.
8. Run markdown lint and fix formatting issues that are easy to correct.

## Reading and Analysis Rules

When reading the paper:

- rely on the original paper content, not third-party summaries
- preserve technical terms in English when Korean translation would reduce precision
- infer the section structure from headings and content, even if the exact heading names differ
- pay special attention to the research problem, core idea, and detailed method explanation because these are required
- explain equations, architectures, losses, objectives, and training procedures in plain Korean when they are central to the method
- explain experiments with enough context to understand what was compared, how it was measured, and why the results matter
- if some detail is not clearly stated in the paper, say so explicitly instead of guessing

## Report Requirements

The report must be detailed enough that a reader can understand the paper without reading the original paper.

### Output language and tone

- write in korean
- keep specialized technical terms in English when helpful
- prefer precise, explanatory prose over short bullet fragments

### Required structure

The first line must be the paper title as an h1 heading:

```markdown
# <paper title>
```

Then include these sections in a sensible order.

## 1. Paper Overview

Include:

- one-paragraph overview of what the paper tries to achieve
- research problem
- why the problem matters

## 2. Core Idea

Include:

- the central intuition or design idea
- what is novel relative to prior approaches if the paper makes that clear

## 3. Detailed Method Explanation

Include:

- overall pipeline or system structure
- each important component and its role
- training objective, loss, inference procedure, or algorithmic flow when relevant
- explanation of important equations

## 4. Experiments and Findings

Include when present:

- datasets, tasks, baselines, metrics
- key quantitative or qualitative results
- what the experiments actually demonstrate

## 5. Strengths, Limitations, and Interpretation

Include:

- strengths supported by the paper
- limitations, assumptions, or open questions
- brief critical interpretation grounded in the paper

## 6. Conclusion

Include:

- concise wrap-up of what the paper contributes
- when this work is likely to matter in practice or future research

## Math Formatting

When writing mathematical expressions:

- tag inline math with `$...$`
- tag display math with `$$...$$`
- do not leave standalone equations untagged
- if the source uses symbols but the rendering is ambiguous, rewrite the math clearly before explaining it

## Metadata Block

At the very end of the report, append a JSON code block with this exact schema:

```json
{"title": "<Paper title>", "author": "<Author list>", "year": <Publish year>, "url": "<Arxiv URL>", "summary": "<summary file name>", "slide": ""}
```

Construct `summary` like this:
- start from the paper title
- convert to English lowercase characters when possible
- replace spaces with `_`
- append `.md`

Example:
- title: `Attention Is All You Need`
- summary: `attention_is_all_you_need.md`

If the title includes punctuation or symbols that would make the filename awkward, simplify them conservatively into a readable lowercase filename.

## Source Attribution

Before the JSON metadata block, add a final section that explicitly lists the paper source URL.

Use this heading:

```markdown
## Source
```

Then provide the canonical arXiv URL used for the report.

## Markdown Quality Check

Before finishing:

- ensure heading levels are consistent
- ensure code fences are balanced
- ensure lists are formatted correctly
- ensure math delimiters are balanced
- correct obvious markdown lint issues when possible

## Failure Handling

- if the title search returns multiple plausible papers, choose the best match using title similarity and arXiv metadata
- if the paper cannot be found on arXiv, say so clearly
- if ar5iv fails and PDF extraction also fails, report the failure transparently and summarize what was successfully retrieved
- never claim to have read sections that were not actually available
