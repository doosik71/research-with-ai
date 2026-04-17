/// <reference types="node" />

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

/**
 * @typedef {{ title: string, year: number, summary: string }} PaperRecord
 * @typedef {{ title?: string }} TopicMetadata
 * @typedef {{ title?: unknown, year?: unknown, summary?: unknown } | null | undefined} PaperRecordCandidate
 * @typedef {"report-01" | "report-04" | "report-07"} PerPaperStageLabel
 * @typedef {{ allowEdits?: boolean }} ClaudeRunOptions
 * @typedef {{
 *   report01: string,
 *   report02: string,
 *   report03: string,
 *   report04: string,
 *   report05: string,
 *   report06: string,
 *   report07: string,
 *   report08: string,
 *   report09: string,
 *   report10: string,
 *   report11: string,
 *   report12: string,
 * }} ReportDirs
 */

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");

function usage() {
	console.error("Usage: node scripts/generate-report.js [--1|--2|--3|--4|--5|--6|--7|--8|--9|--10|--11|--12] <topic_id>");
	console.error("");
	console.error("Stages:");
	console.error("  --1   Generate per-paper overall summaries into report-01");
	console.error("  --2   Merge report-01 summaries into report-02/summary-merged.md");
	console.error("  --3   Generate taxonomy.md from report-02/summary-merged.md");
	console.error("  --4   Generate per-paper methodology summaries into report-04");
	console.error("  --5   Merge report-04 summaries into report-05/method-merged.md");
	console.error("  --6   Generate method-summary.md from report-05/method-merged.md");
	console.error("  --7   Generate per-paper result summaries into report-07");
	console.error("  --8   Merge report-07 summaries into report-08/result-merged.md");
	console.error("  --9   Generate result-summary.md from report-08/result-merged.md");
	console.error("  --10  Merge taxonomy, method-summary, and result-summary into report-10/report.md");
	console.error("  --11  Generate introduction.md from report-10/report.md into report-11");
	console.error("  --12  Merge report-11/introduction.md and report-10/report.md into report-12/final-report.md");
	process.exit(1);
}

/**
 * @param {string[]} argv
 * @returns {{ topicId: string, stageNumber: number | null }}
 */
function parseArgs(argv) {
	const args = argv.slice(2);
	if (args.length < 1 || args.length > 2) {
		usage();
	}

	if (args.length === 1) {
		const [topicId] = args;
		if (!topicId || topicId.startsWith("--")) {
			usage();
		}
		return { topicId, stageNumber: null };
	}

	const [stageArg, topicId] = args;
	const match = /^--([1-9]|1[0-2])$/.exec(stageArg || "");
	if (!match || !topicId || topicId.startsWith("--")) {
		usage();
	}

	return { topicId, stageNumber: Number(match[1]) };
}

/**
 * @param {string} targetPath
 */
async function pathExists(targetPath) {
	try {
		await fs.access(targetPath);
		return true;
	} catch {
		return false;
	}
}

/**
 * @param {string} targetPath
 */
async function ensureDir(targetPath) {
	await fs.mkdir(targetPath, { recursive: true });
}

/**
 * @param {string} targetPath
 * @param {string} label
 */
async function ensureFile(targetPath, label) {
	if (!(await pathExists(targetPath))) {
		throw new Error(`${label} not found: ${path.relative(projectRoot, targetPath)}`);
	}
}

/**
 * @param {string} filePath
 */
async function readJson(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
	try {
		return JSON.parse(raw);
	} catch (error) {
		throw new Error(
			`Invalid JSON at ${path.relative(projectRoot, filePath)} (${error instanceof Error ? error.message : String(error)})`,
		);
	}
}

/**
 * @param {string} filePath
 */
async function readJsonl(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
	/** @type {unknown[]} */
	const records = [];

	for (const [index, rawLine] of raw.split(/\r?\n/).entries()) {
		const lineNumber = index + 1;
		const line = rawLine.trim();
		if (!line || line.startsWith("//")) {
			continue;
		}

		try {
			records.push(JSON.parse(line));
		} catch (error) {
			throw new Error(
				`Invalid JSONL at ${path.relative(projectRoot, filePath)}:${lineNumber} (${error instanceof Error ? error.message : String(error)})`,
			);
		}
	}

	return records;
}

/**
 * @param {ClaudeRunOptions} [options]
 */
function getClaudeCommand(options = {}) {
	const { allowEdits = true } = options;
	const args = ["launch", "claude", "--model", "qwen3.5", "--", "--print"];
	if (allowEdits) {
		args.push("--permission-mode", "acceptEdits");
	}

	return {
		command: "ollama",
		args,
	};
}

/**
 * @param {string} stageLabel
 * @param {string} message
 */
function logStageStart(stageLabel, message) {
	console.log(`[${stageLabel}] start ${message}`);
}

/**
 * @param {string} stageLabel
 * @param {number} current
 * @param {number} total
 * @param {string} filePath
 * @param {string} action
 */
function logFileProgress(stageLabel, current, total, filePath, action) {
	console.log(`[${stageLabel}] [${current}/${total}] ${action} ${path.relative(projectRoot, filePath)}`);
}

/**
 * @param {string} stageLabel
 * @param {number} current
 * @param {number} total
 * @param {PaperRecord} paper
 * @param {string} outputKind
 */
function logPaperProgress(stageLabel, current, total, paper, outputKind) {
	console.log(`[${stageLabel}] [${current}/${total}] ${outputKind}: ${paper.title} (${paper.year})`);
}

async function ensureClaudeAvailable() {
	const runner = { command: "ollama", args: ["--version"] };

	/** @type {Promise<void>} */
	await new Promise((resolve, reject) => {
		const child = spawn(runner.command, runner.args, {
			stdio: "ignore",
			cwd: projectRoot,
		});

		child.on("error", () => {
			reject(new Error("ollama command is not available."));
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve();
				return;
			}
			reject(new Error("ollama command is not available."));
		});
	});
}

/**
 * @param {string} prompt
 * @param {ClaudeRunOptions} [options]
 * @returns {Promise<string>}
 */
async function runClaude(prompt, options = {}) {
	const { command, args } = getClaudeCommand(options);

	return new Promise((resolve, reject) => {
		const child = spawn(command, args, {
			cwd: projectRoot,
			stdio: ["pipe", "pipe", "pipe"],
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", /** @param {Buffer|string} chunk */ (chunk) => {
			stdout += String(chunk);
		});

		child.stderr.on("data", /** @param {Buffer|string} chunk */ (chunk) => {
			stderr += String(chunk);
		});

		child.on("error", /** @param {Error} error */ (error) => {
			reject(new Error(`Failed to start ollama claude launcher: ${error.message}`));
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve(stdout.trim());
				return;
			}

			const detail = stderr.trim() || stdout.trim() || `exit code ${code}`;
			reject(new Error(`ollama claude launcher failed: ${detail}`));
		});

		child.stdin.end(prompt);
	});
}

/**
 * @param {string} targetPath
 */
async function runMarkdownlintFix(targetPath) {
	return new Promise((resolve, reject) => {
		const child = spawn("markdownlint", ["--fix", targetPath], {
			cwd: projectRoot,
			stdio: ["ignore", "pipe", "pipe"],
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", /** @param {Buffer|string} chunk */ (chunk) => {
			stdout += String(chunk);
		});

		child.stderr.on("data", /** @param {Buffer|string} chunk */ (chunk) => {
			stderr += String(chunk);
		});

		child.on("error", /** @param {Error} error */ (error) => {
			reject(new Error(`Failed to start markdownlint: ${error.message}`));
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve();
				return;
			}

			const detail = stderr.trim() || stdout.trim() || `exit code ${code}`;
			if (code === 1) {
				console.warn(
					`[markdownlint] remaining issues in ${path.relative(projectRoot, targetPath)} after --fix\n${detail}`,
				);
				resolve();
				return;
			}

			reject(new Error(`markdownlint --fix failed for ${path.relative(projectRoot, targetPath)}: ${detail}`));
		});
	});
}

/**
 * @param {string} line
 */
function countMarkdownTableColumns(line) {
	return line
		.trim()
		.replace(/^\|/, "")
		.replace(/\|$/, "")
		.split("|").length;
}

/**
 * @param {string[]} cells
 */
function buildMarkdownTableRow(cells) {
	return `|${cells.map((cell) => cell.trim()).join("|")}|`;
}

/**
 * @param {number} columns
 */
function buildMarkdownTableSeparator(columns) {
	return `|${Array.from({ length: columns }, () => "---").join("|")}|`;
}

/**
 * @param {string} markdown
 * @param {{ demoteTopLevelHeadings?: boolean }} [options]
 */
function normalizeMarkdown(markdown, options = {}) {
	const { demoteTopLevelHeadings = false } = options;
	const lines = markdown.replace(/\r\n/g, "\n").split("\n");
	const normalized = [];
	let inFence = false;
	let activeTableColumns = 0;

	for (let index = 0; index < lines.length; index += 1) {
		let line = lines[index];

		if (/^```$/.test(line.trim())) {
			inFence = !inFence;
			activeTableColumns = 0;
			normalized.push(line);
			continue;
		}

		if (!inFence && demoteTopLevelHeadings && /^#\s+/.test(line)) {
			line = line.replace(/^#\s+/, "## ");
		}

		if (!inFence) {
			line = line.replace(/\*\*([^*\n]*?)\s+\*\*/g, "**$1**");
		}

		const nextLine = lines[index + 1] ?? "";
		const isHeaderRow = !inFence && /\|/.test(line) && /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(nextLine);

		if (isHeaderRow) {
			activeTableColumns = countMarkdownTableColumns(line);
			const headerCells = line
				.trim()
				.replace(/^\|/, "")
				.replace(/\|$/, "")
				.split("|")
				.map((cell) => cell.trim());
			normalized.push(buildMarkdownTableRow(headerCells));
			continue;
		}

		if (!inFence && activeTableColumns > 0) {
			if (/^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line)) {
				normalized.push(buildMarkdownTableSeparator(activeTableColumns));
				continue;
			}

			if (/\|/.test(line)) {
				const cells = line
					.trim()
					.replace(/^\|/, "")
					.replace(/\|$/, "")
					.split("|")
					.map((cell) => cell.trim());

				if (cells.length < activeTableColumns) {
					while (cells.length < activeTableColumns) {
						cells.push("");
					}
					line = buildMarkdownTableRow(cells);
				} else if (cells.length > activeTableColumns) {
					const kept = cells.slice(0, activeTableColumns - 1);
					const tail = cells.slice(activeTableColumns - 1).join(" / ");
					line = buildMarkdownTableRow([...kept, tail]);
				} else {
					line = buildMarkdownTableRow(cells);
				}

				if (!line.startsWith("| ")) {
					line = buildMarkdownTableRow(
						line
							.trim()
							.replace(/^\|/, "")
							.replace(/\|$/, "")
							.split("|"),
					);
				}

				normalized.push(line);
				continue;
			}

			activeTableColumns = 0;
		}

		normalized.push(line);
	}

	let result = normalized.join("\n").replace(/\n{3,}/g, "\n\n").trim();
	if (demoteTopLevelHeadings) {
		result = result.replace(/^#\s+/gm, "## ");
	}
	return result;
}

/**
 * @param {PaperRecordCandidate} record
 * @param {number} index
 * @param {string} topicId
 * @returns {PaperRecord}
 */
function normalizePaperRecord(record, index, topicId) {
	const candidate = record ?? {};

	if (typeof candidate.title !== "string" || !candidate.title.trim()) {
		throw new Error(`Invalid paper record at entry ${index + 1} for topic "${topicId}": missing title`);
	}

	if (!Number.isInteger(candidate.year)) {
		throw new Error(`Invalid paper record at entry ${index + 1} for topic "${topicId}": missing or invalid year`);
	}

	const year = candidate.year;

	return {
		title: candidate.title.trim(),
		year,
		summary: typeof candidate.summary === "string" ? candidate.summary.trim() : "",
	};
}

/**
 * @param {PaperRecord} paper
 * @param {string} sourceText
 */
function buildStage1Prompt(paper, sourceText) {
  return `
다음은 논문 상세 분석 보고서다.

이 보고서를 바탕으로, 이후 "연구체계 분류" 분석에 직접 사용할 수 있는 한국어 마크다운 요약문을 작성하라.
이 요약문의 목적은 개별 논문을 잘 소개하는 것이 아니라, 여러 논문을 비교·분류할 때 필요한 핵심 정보를 일관된 형식으로 추출하는 것이다.

반드시 아래 규칙을 지켜라.

# 출력 목적
- 이 요약문은 후속 단계에서 여러 논문을 묶어 연구 분류 체계를 수립하는 입력 자료로 사용된다.
- 따라서 서술형 감상이나 일반적 소개보다, "분류 근거가 되는 정보"가 분명하게 드러나야 한다.
- 요약문에 없는 정보는 후속 분류 단계에서 사용할 수 없으므로, 분류에 필요한 핵심 요소를 빠뜨리지 말아라.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 첫 줄은 정확히 \`# ${paper.title} (${paper.year})\` 로 작성한다.
2. 전체는 첫 줄 포함 15~20줄 내외로 작성한다.
3. 마크다운 불릿 리스트 형식으로 작성한다.
4. 각 항목은 짧고 밀도 있게 1~2문장 이내로 작성한다.
5. 원문에 없는 내용은 추측, 일반화, 보완하여 쓰지 않는다.
6. 불필요한 서론, 결론, 메타 설명은 쓰지 않는다.
7. 수식적 표현이나 장황한 배경 설명보다, 분류 가능한 정보가 드러나도록 쓴다.
8. 애매하거나 원문 근거가 약한 경우 단정적으로 쓰지 않는다.

# 반드시 포함할 항목
아래 항목을 이 순서대로 반드시 포함하라.  
항목명은 그대로 사용하라.

- **연구 대상**: 이 논문이 다루는 시스템, 데이터, 환경, 문제영역, 적용대상
- **핵심 문제**: 해결하려는 중심 문제 또는 연구 질문
- **연구 목적**: 무엇을 개선, 규명, 제안, 비교하려는지
- **접근 관점**: 이 논문을 분류할 때 핵심이 되는 관점  
  (예: 시스템 구조, 운영 전략, 모델링 관점, 평가 관점, 적용 시나리오 등. 단, 반드시 원문 근거가 있을 때만 작성)
- **핵심 방법**: 사용한 주요 기법, 절차, 모델, 알고리즘, 실험 설계
- **입력/데이터/조건**: 사용 데이터, 입력 정보, 실험 조건, 비교 조건
- **주요 결과**: 성능, 관찰 결과, 비교 결과, 확인된 효과
- **한계 또는 조건**: 적용 범위, 제약, 전제조건, 실험상의 제한  
  (원문에 명시된 경우에만 작성)
- **분류 키워드**: 후속 분류체계 수립에 활용할 수 있도록 이 논문을 대표하는 키워드 3~6개
- **분류 근거 문장**: 이 논문이 어떤 연구 범주에 속할지를 판단하는 데 가장 중요한 근거를 1~2문장으로 정리

# 작성 원칙
- "방법론 분석"이나 "실험결과 분석"에만 유리한 세부사항보다, "연구체계 분류"에 필요한 상위 수준 정보를 우선 정리하라.
- 특히 아래 요소가 드러나도록 하라.
  - 무엇을 연구하는가
  - 어떤 문제를 다루는가
  - 어떤 관점에서 접근하는가
  - 어떤 유형의 방법에 속하는가
  - 어떤 맥락/대상에 적용되는가
- 단순 성능 수치 나열보다, 해당 논문의 위치를 분류할 수 있는 정보가 더 중요하다.
- 논문 제목만 반복 설명하지 말고, 실제 분류 근거가 되는 내용을 남겨라.

# 출력 형식 예시
형식만 참고하고, 실제 내용은 반드시 아래 보고서에 근거해 작성하라.

# ${paper.title} (${paper.year})
- **연구 대상**: ...
- **핵심 문제**: ...
- **연구 목적**: ...
- **접근 관점**: ...
- **핵심 방법**: ...
- **입력/데이터/조건**: ...
- **주요 결과**: ...
- **한계 또는 조건**: ...
- **분류 키워드**: 키워드1, 키워드2, 키워드3
- **분류 근거 문장**: ...

상세 분석 보고서:

${sourceText}
`.trim();
}

/**
 * @param {string} topicTitle
 * @param {string} mergedText
 */
function buildTaxonomyPrompt(topicTitle, mergedText) {
  return `
당신은 "${topicTitle}" 분야의 전문 연구자이다.
당신의 목표는 여러 편의 논문 요약문만을 근거로 연구 분석 보고서의 "1장. 연구체계 분류"를 작성하는 것이다.

이 보고서는 최종적으로 다음 3개 장으로 구성될 예정이지만, 지금 작성할 범위는 오직 1장만이다.
- 1장. 연구체계 분류
- 2장. 방법론 분석
- 3장. 실험결과 분석

중요:
- 지금은 "1장. 연구체계 분류"만 작성한다.
- 2장과 3장 내용은 절대 작성하지 않는다.
- 아래에 제공된 논문 요약문에 포함된 정보만 사용한다.
- 요약문에 없는 내용은 추정, 보완, 일반화하여 쓰지 않는다.
- 특정 논문이 어느 범주에 속하는지 불명확하면 임의 분류하지 말고 보수적으로 기술한다.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

다음 작성 규칙을 반드시 지켜라.

# 출력 규칙
1. 문서는 반드시 마크다운 형식으로 작성한다.
2. 첫 줄 제목은 반드시 다음과 같이 쓴다.
   ## 1장. 연구체계 분류
3. 제목 아래에는 먼저 "연구 분류 체계 수립 기준"이라는 소절을 두고,
   논문 요약들을 어떤 기준과 원칙으로 분류했는지 설명한다.
4. 그 다음 "연구 분류 체계" 소절에서 대분류와 하위 범주를 계층적으로 제시한다.
5. 분류 체계는 가능한 한 중복이 적고, 서로 구분 가능하며, 전체 논문을 포괄하도록 구성한다.
6. 각 대분류 또는 하위 범주별로 해당 범주에 속하는 논문 목록을 반드시 표로 제시한다.
7. 표에는 최소한 다음 열이 포함되어야 한다.
   - 분류
   - 논문명
   - 분류 근거
8. 논문명은 반드시 다음 형식으로 표기한다.
   논문 제목 (발표연도)
9. 분류 근거는 반드시 요약문에서 확인 가능한 내용만 바탕으로 1문장 이내로 간단히 쓴다.
10. 논문이 여러 범주에 동시에 걸칠 수 있더라도, 본문에서는 가장 대표적인 1개 범주에만 배치한다.
11. 범주별 설명은 장황하지 않게 작성하되, 왜 해당 범주가 필요한지는 드러나야 한다.
12. 마지막에는 "종합 정리" 소절을 두고 전체 연구 지형을 1개 단락으로 요약한다.
13. 출력에는 서론, 결론, 참고문헌, 2장/3장 예고 문구를 쓰지 않는다.
14. 요약문에 논문 발표연도가 없으면 연도를 추정하지 말고 다음처럼 쓴다.
   논문 제목 (연도 미상)
15. 표의 모든 데이터 행은 헤더와 정확히 같은 열 개수를 유지한다.
16. 코드 블록을 사용할 경우 반드시 시작 펜스를 \`\`\`text 처럼 언어와 함께 작성한다.

# 권장 작성 절차
아래 절차를 내부적으로 따른 뒤 결과만 출력하라.
- 먼저 논문들을 공통 주제, 문제 정의, 적용 대상, 접근 방식, 시스템 구성 관점에서 검토한다.
- 그 다음 대분류를 설계하고, 필요하면 각 대분류 아래 하위 범주를 둔다.
- 이후 모든 논문을 가장 적절한 범주에 1회씩만 배치한다.
- 마지막으로 분류 체계의 특징과 전체 연구 흐름을 요약한다.

# 출력 형식 예시
아래 형식을 참고하되, 실제 내용은 반드시 제공된 요약문에 근거해 작성한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준
(분류 기준과 원칙 설명)

### 2. 연구 분류 체계
#### 2.1 대분류 A
(필요 시 설명)

| 분류 | 논문명 | 분류 근거 |
|---|---|---|
| 대분류 A > 하위범주 A-1 | 논문 제목 (2023) | 요약문에 나타난 핵심 초점 |
| 대분류 A > 하위범주 A-2 | 논문 제목 (2021) | 요약문에 나타난 핵심 초점 |

#### 2.2 대분류 B
| 분류 | 논문명 | 분류 근거 |
|---|---|---|
| 대분류 B > 하위범주 B-1 | 논문 제목 (연도 미상) | 요약문에 나타난 핵심 초점 |

### 3. 종합 정리
(전체 연구 지형 요약 1단락)

다음은 분석 대상 논문 요약 모음이다.

---
${mergedText}
---
`.trim();
}

/**
 * @param {PaperRecord} paper
 * @param {string} sourceText
 */
function buildMethodPrompt(paper, sourceText) {
  return `
다음은 논문 상세 분석 보고서다.

이 보고서를 바탕으로, 후속 단계의 "방법론 분석"에 직접 사용할 수 있는 한국어 마크다운 문서를 작성하라.
목표는 논문의 방법론을 일관된 형식으로 구조화하여, 여러 논문의 방법론을 서로 비교·분석할 수 있도록 만드는 것이다.

반드시 아래 규칙을 지켜라.

# 출력 목적
- 이 문서는 개별 논문의 방법론만 추출하는 중간 산출물이다.
- 후속 단계에서는 여러 논문의 방법론을 비교하여 공통 구조, 차이점, 방법론 계열을 분석할 예정이다.
- 따라서 서술형 설명보다, 방법론의 구성 요소와 절차가 분명히 드러나도록 작성한다.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 첫 줄은 정확히 \`# ${paper.title} (${paper.year})\` 로 작성한다.
2. 문서는 한국어 마크다운으로 작성한다.
3. 전체는 첫 줄 포함 15~22줄 내외로 유지한다.
4. 반드시 불릿 리스트 형식으로 작성한다.
5. 각 항목은 1~2문장 이내로 짧고 밀도 있게 작성한다.
6. 방법론 중심으로만 작성하고, 배경 설명은 최소화한다.
7. 실험 결과, 성능 수치, 일반적 의의는 방법 이해에 꼭 필요한 경우가 아니면 쓰지 않는다.
8. 원문에 없는 세부 구현, 알고리즘 단계, 하이퍼파라미터, 수식 의미는 추측하지 않는다.
9. 논문에 해당 항목 정보가 없으면 없는 내용을 만들지 말고 생략하거나 "명시되지 않음"으로 처리한다.
10. 장황한 문장보다 비교 가능한 구조가 유지되도록 쓴다.

# 반드시 포함할 항목
아래 항목명을 그대로 사용하고, 가능한 한 이 순서를 유지하라.

- **문제 설정**: 방법론이 해결하려는 구체적 문제를 한두 문장으로 정리
- **입력과 출력**: 모델/시스템이 무엇을 입력받고 무엇을 산출하는지
- **전체 접근 방식**: 제안 방법의 전체 구조를 가장 압축적으로 설명
- **핵심 구성 요소**: 모듈, 단계, 서브시스템, 핵심 메커니즘
- **처리 절차**: 데이터 흐름, 연산 순서, 알고리즘 절차, 파이프라인
- **모델/구조적 특징**: 네트워크 구조, 수식 구조, 최적화 구조, 시스템 아키텍처 등
- **학습 또는 최적화 방식**: 학습 절차, 손실함수, 최적화 전략, 파라미터 추정 방식  
  (해당할 경우에만 작성)
- **추론 또는 적용 방식**: 실제 적용/실행/의사결정 절차  
  (해당할 경우에만 작성)
- **기존 방법과의 차별점**: 방법론 차원에서 무엇이 새롭거나 다른지
- **방법론의 전제/조건**: 적용 조건, 가정, 요구 데이터, 제약사항  
  (원문에 있을 경우에만 작성)
- **방법론 키워드**: 후속 비교 분석에 쓸 수 있는 핵심 키워드 3~6개
- **방법론 요약 문장**: 이 논문의 방법론을 대표하는 1~2문장 요약

# 작성 원칙
- "무엇을 제안했는가"보다 "어떻게 작동하는가"가 드러나야 한다.
- 가능하면 다음 정보를 분명히 남겨라.
  - 어떤 입력을 사용했는가
  - 어떤 구조/모듈로 이루어졌는가
  - 어떤 순서로 처리되는가
  - 어떤 학습/최적화 과정을 거치는가
  - 어떤 점에서 기존 방법과 구별되는가
- 수식이 등장하면 수식 전체를 길게 재현하기보다, 그 수식이 방법론에서 어떤 역할을 하는지 요약한다.
- 알고리즘이 등장하면 세부 코드 수준이 아니라 핵심 절차와 단계만 정리한다.
- 실험 파트의 내용은 원칙적으로 제외하되, 방법의 작동 방식을 이해하는 데 필수적인 설정만 제한적으로 포함한다.

# 출력 형식 예시
형식만 참고하고, 실제 내용은 반드시 아래 보고서에 근거해 작성하라.

# ${paper.title} (${paper.year})
- **문제 설정**: ...
- **입력과 출력**: ...
- **전체 접근 방식**: ...
- **핵심 구성 요소**: ...
- **처리 절차**: ...
- **모델/구조적 특징**: ...
- **학습 또는 최적화 방식**: ...
- **추론 또는 적용 방식**: ...
- **기존 방법과의 차별점**: ...
- **방법론의 전제/조건**: ...
- **방법론 키워드**: 키워드1, 키워드2, 키워드3
- **방법론 요약 문장**: ...

상세 분석 보고서:

${sourceText}
`.trim();
}

/**
 * @param {string} topicTitle
 * @param {string} mergedText
 */
function buildMethodSummaryPrompt(topicTitle, mergedText) {
  return `
다음은 "${topicTitle}" 주제 논문들의 방법론 구조화 문서 모음이다.

이 자료를 바탕으로 "2장. 방법론 분석"을 작성하라.
이 단계의 목표는 개별 논문 요약이 아니라, 전체 방법론을 비교·분류·추상화하는 것이다.

중요:
- 반드시 제공된 문서에 포함된 정보만 사용한다.
- 없는 내용을 일반화하거나 추정하지 않는다.
- 개별 논문 요약을 반복하지 말고, "방법론 구조"를 재구성하라.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 문서는 한국어 마크다운 형식으로 작성한다.
2. 제목은 반드시 다음과 같이 작성한다.
   ## 2장. 방법론 분석
3. 서술형 에세이가 아니라, 구조적 분석 문서 형태로 작성한다.
4. 표와 계층 구조를 적극적으로 사용한다.
5. 논문 제목은 반드시 다음 형식으로 표기한다.
   논문 제목 (연도)
6. 코드 블록을 사용할 경우 반드시 시작 펜스를 \`\`\`text 처럼 언어와 함께 작성한다.
7. 표의 모든 데이터 행은 헤더와 정확히 같은 열 개수를 유지한다.

# 반드시 포함할 구성

## 1. 공통 문제 설정 및 접근 구조
- 전체 논문들이 다루는 공통 문제를 정리
- 방법론 관점에서 공통적으로 나타나는 구조 요약
  (예: 입력 → 처리 → 출력, 모델 중심 vs 시스템 중심 등)

## 2. 방법론 계열 분류
- 논문들을 방법론 유형별로 그룹화
- 계열은 서로 구분 가능해야 하며, 중복 최소화

각 계열에 대해 다음을 포함하라:

### (계열명)
- 계열 정의
- 공통 특징 (구조, 절차, 접근 방식)
- 해당 논문 목록

| 방법론 계열 | 논문명 | 핵심 특징 |
|---|---|---|
| 계열 A | 논문 제목 (2023) | 핵심 구조 또는 접근 방식 |

## 3. 핵심 설계 패턴 분석
- 여러 논문에서 반복적으로 등장하는 방법론 패턴을 추출
- 예:
  - 특정 구조 (모듈 분리, 단계적 처리 등)
  - 특정 학습 방식 (지도/비지도/하이브리드)
  - 특정 데이터 활용 방식
- 패턴별로 어떤 논문들이 해당되는지 명시

## 4. 방법론 비교 분석
- 계열 간 차이점과 트레이드오프 분석
- 반드시 아래 관점을 포함:
  - 문제 접근 방식 차이
  - 구조/모델 차이
  - 적용 대상 차이
  - 복잡도 또는 확장성 차이 (자료에 있을 경우만)

## 5. 방법론 흐름 및 진화
- 시간 흐름 또는 접근 방식 변화 관점에서 정리
- 초기 접근 → 발전된 구조 → 최근 경향
- 단, 제공된 자료 범위 내에서만 기술

## 6. 종합 정리
- 전체 방법론 지형을 한 단락으로 요약
- 어떤 축으로 방법론이 나뉘는지 명확히 드러내라

# 작성 원칙
- 개별 논문 설명을 길게 반복하지 않는다
- 항상 "여러 논문을 묶어서 설명"한다
- 반드시 다음 질문에 답하는 형태로 작성하라:
  - 어떤 유형의 방법들이 존재하는가?
  - 어떻게 서로 다른가?
  - 무엇이 공통적인가?
  - 어떤 패턴이 반복되는가?
- 애매한 경우 억지로 계열을 만들지 말고 보수적으로 정리한다

방법론 문서 모음:

${mergedText}
`.trim();
}

/**
 * @param {PaperRecord} paper
 * @param {string} sourceText
 */
function buildResultPrompt(paper, sourceText) {
  return `
다음은 논문 상세 분석 보고서다.

이 보고서를 바탕으로, 후속 단계의 "3장. 실험결과 분석"에 직접 사용할 수 있는 한국어 마크다운 문서를 작성하라.
목표는 개별 논문의 실험 결과를 일관된 형식으로 구조화하여, 여러 논문의 실험 설정·비교 기준·결과·해석을 종합 비교할 수 있도록 만드는 것이다.

반드시 아래 규칙을 지켜라.

# 출력 목적
- 이 문서는 개별 논문의 실험 결과만 추출하는 중간 산출물이다.
- 후속 단계에서는 여러 논문의 실험 결과를 비교하여 공통 평가 방식, 성능 차이, 검증 패턴, 한계점을 분석할 예정이다.
- 따라서 단순 요약보다, 비교 가능한 실험 정보가 분명히 드러나도록 작성한다.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 첫 줄은 정확히 \`# ${paper.title} (${paper.year})\` 로 작성한다.
2. 문서는 한국어 마크다운으로 작성한다.
3. 전체는 첫 줄 포함 15~22줄 내외로 유지한다.
4. 반드시 불릿 리스트 형식으로 작성한다.
5. 각 항목은 1~2문장 이내로 짧고 밀도 있게 작성한다.
6. 실험 결과 중심으로만 작성하고, 방법론 설명은 결과를 이해하는 데 필요한 최소한만 포함한다.
7. 정량 결과가 있으면 가능한 한 수치, 비교 대상, 평가 기준을 함께 쓴다.
8. 표나 그림 번호를 그대로 옮기기보다, 그 안의 핵심 결과를 텍스트로 정리한다.
9. 제공된 보고서에 없는 결과, 누락된 수치, 비교 우위의 이유를 추측하지 않는다.
10. 항목 정보가 없으면 없는 내용을 만들지 말고 생략하거나 "명시되지 않음"으로 처리한다.
11. 결과가 여러 개인 경우, 가장 핵심적인 실험 결과부터 우선 정리한다.

# 반드시 포함할 항목
아래 항목명을 그대로 사용하고, 가능한 한 이 순서를 유지하라.

- **평가 목적**: 무엇을 검증하기 위한 실험인지
- **사용 데이터셋 또는 평가 환경**: 데이터셋, 벤치마크, 시뮬레이션/실환경, 테스트 조건
- **비교 대상**: baseline, 기존 방법, 비교 모델, 비교 조건
- **평가 지표**: accuracy, F1, latency, error, cost 등 사용된 지표
- **실험 설정의 핵심 조건**: 분할 방식, 파라미터 조건, 시나리오 차이, ablation 여부 등  
  (원문에 있을 경우에만 작성)
- **핵심 정량 결과**: 가장 중요한 수치 결과를 비교 맥락과 함께 정리
- **보조 결과 또는 추가 분석**: ablation, 민감도 분석, 조건별 비교, 사례 분석 등  
  (원문에 있을 경우에만 작성)
- **저자 해석**: 저자가 결과를 어떻게 해석하는지
- **한계 또는 주의사항**: 실험 범위, 일반화 한계, 조건 의존성 등  
  (원문에 있을 경우에만 작성)
- **결과 키워드**: 후속 종합 분석에 활용할 수 있는 키워드 3~6개
- **결과 요약 문장**: 이 논문의 실험 결과를 대표하는 1~2문장 요약

# 작성 원칙
- "수치가 높았다"처럼 맥락 없는 표현보다, 무엇과 비교해 어떤 지표에서 어떤 결과가 나왔는지 드러나야 한다.
- 정량 수치가 있으면 가능한 한 다음 요소를 함께 남겨라.
  - 어떤 데이터셋/환경에서
  - 누구와 비교해
  - 어떤 지표에서
  - 얼마나 개선/차이 나는지
- 수치가 없고 정성 평가만 있으면, 그 정성 결과가 무엇을 보여주는지 간결하게 정리하라.
- 방법론 설명은 최소화하되, 결과 해석에 꼭 필요한 실험 조건은 남긴다.
- 저자 해석과 관찰된 결과를 구분해서 써라.
- 실험이 여러 갈래라면, 대표 실험 / 비교 실험 / 추가 분석이 구분되도록 써라.

# 출력 형식 예시
형식만 참고하고, 실제 내용은 반드시 아래 보고서에 근거해 작성하라.

# ${paper.title} (${paper.year})
- **평가 목적**: ...
- **사용 데이터셋 또는 평가 환경**: ...
- **비교 대상**: ...
- **평가 지표**: ...
- **실험 설정의 핵심 조건**: ...
- **핵심 정량 결과**: ...
- **보조 결과 또는 추가 분석**: ...
- **저자 해석**: ...
- **한계 또는 주의사항**: ...
- **결과 키워드**: 키워드1, 키워드2, 키워드3
- **결과 요약 문장**: ...

상세 분석 보고서:

${sourceText}
`.trim();
}

/**
 * @param {string} topicTitle
 * @param {string} mergedText
 */
function buildResultSummaryPrompt(topicTitle, mergedText) {
  return `
다음은 "${topicTitle}" 주제 논문들의 실험 결과 구조화 문서 모음이다.

이 자료를 바탕으로 "3장. 실험결과 분석"을 작성하라.
이 단계의 목표는 개별 논문 결과를 요약하는 것이 아니라, 전체 실험 결과를 비교·정렬·해석하는 것이다.

중요:
- 반드시 제공된 문서에 포함된 정보만 사용한다.
- 없는 수치, 비교 결과, 해석을 추정하지 않는다.
- 개별 논문 요약을 반복하지 말고, "결과 구조"를 재구성하라.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 문서는 한국어 마크다운 형식으로 작성한다.
2. 제목은 반드시 다음과 같이 작성한다.
   ## 3장. 실험결과 분석
3. 서술형 에세이가 아니라 구조적 분석 문서 형태로 작성한다.
4. 표를 적극적으로 사용하여 논문 간 비교가 가능하도록 한다.
5. 논문 제목은 반드시 다음 형식으로 표기한다.
   논문 제목 (연도)
6. 코드 블록을 사용할 경우 반드시 시작 펜스를 \`\`\`text 처럼 언어와 함께 작성한다.
7. 표의 모든 데이터 행은 헤더와 정확히 같은 열 개수를 유지한다.

# 반드시 포함할 구성

## 1. 평가 구조 및 공통 실험 설정
- 전체 논문에서 공통적으로 사용된 평가 구조 정리
  - 주요 데이터셋 유형
  - 평가 환경 (실험/시뮬레이션/실환경)
  - 비교 방식 (baseline, SOTA 비교 등)
- 주요 평가 지표 정리

## 2. 주요 실험 결과 정렬
- 핵심 결과를 비교 가능하도록 정리

| 논문명 | 데이터셋/환경 | 비교 대상 | 평가 지표 | 핵심 결과 |
|---|---|---|---|---|
| 논문 제목 (2023) | ... | ... | ... | ... |

- 수치가 있는 경우 반드시 비교 맥락과 함께 정리

## 3. 성능 패턴 및 경향 분석
다음 관점에서 분석하라:

- 공통적으로 나타나는 성능 개선 패턴
- 특정 조건에서만 성능이 향상되는 경우
- 논문 간 상충되는 결과
- 데이터셋 또는 환경에 따른 성능 차이

## 4. 추가 실험 및 검증 패턴
- ablation study, 민감도 분석, 조건 변화 실험 등
- 어떤 방식으로 방법론을 검증하는지 공통 패턴 정리

## 5. 실험 설계의 한계 및 비교상의 주의점
- 비교 조건의 불일치
- 데이터셋 의존성
- 일반화 한계
- 평가 지표의 한계
(반드시 자료에 근거해서만 작성)

## 6. 결과 해석의 경향
- 저자들이 결과를 어떻게 해석하는지 공통 경향 정리
- 단, 해석과 실제 관찰 결과를 구분해서 기술

## 7. 종합 정리
- 전체 실험 결과가 시사하는 바를 1개 단락으로 정리
- 어떤 조건에서 어떤 방법이 유리한지 드러나야 한다

# 작성 원칙
- 개별 논문 결과를 길게 반복하지 않는다
- 반드시 "여러 논문을 묶어서" 설명한다
- 다음 질문에 답하는 형태로 작성하라:
  - 어떤 평가 방식이 주로 사용되는가?
  - 어떤 조건에서 성능이 좋아지는가?
  - 결과는 얼마나 일관적인가?
  - 어떤 경우 결과가 달라지는가?
- 수치가 있을 경우 반드시 비교 맥락과 함께 제시한다
- "우수하다", "좋다" 같은 표현은 지양하고, 구체적 결과 중심으로 작성한다

실험 결과 문서 모음:

${mergedText}
`.trim();
}

/**
 * @param {string} topicTitle
 * @param {string} reportText
 */
function buildIntroductionPrompt(topicTitle, reportText) {
	return `
다음은 "${topicTitle}" 주제에 대한 연구 분석 보고서 본문이다.

이 본문을 바탕으로 보고서의 "서론"만 작성하라.
이 서론은 최종 보고서의 맨 앞에 배치되며, 뒤이어 오는 1장~3장을 자연스럽게 읽을 수 있도록 연구 배경, 문제의식, 보고서 목적, 구성만 정리하는 역할을 한다.

중요:
- 반드시 아래에 제공된 보고서 본문에 포함된 정보만 사용한다.
- 본문에 없는 배경지식, 역사, 외부 사례, 일반론을 임의로 추가하지 않는다.
- 1장, 2장, 3장의 세부 내용을 다시 길게 요약하지 말고, 각 장이 어떤 관점의 분석을 담당하는지만 간결히 연결한다.
- 파일을 생성하거나 수정하지 말고, 최종 마크다운 본문만 stdout으로 출력하라.
- 출력 저장은 호출자가 처리하므로, 파일명 제안이나 저장 경로 설명도 쓰지 말아라.

# 출력 규칙
1. 문서는 한국어 마크다운 형식으로 작성한다.
2. 첫 줄 제목은 반드시 정확히 \`# 서론\` 으로 작성한다.
3. 전체 분량은 5~8개 단락 또는 소절 이내로 유지한다.
4. 서론은 최종 보고서의 도입부답게 간결하고 밀도 있게 작성한다.
5. 과장된 표현, 홍보성 표현, 메타 설명은 쓰지 않는다.
6. 장 구성 소개는 반드시 포함하되, 장별 세부 표나 긴 목록은 쓰지 않는다.

# 반드시 포함할 내용
- 연구 주제와 보고서가 다루는 범위
- 해당 주제를 비교·분석할 필요성 또는 문제의식
- 본 보고서가 어떤 관점으로 문헌을 정리하는지
- 보고서 구성 안내

# 권장 구성
아래 구조를 따르되, 실제 문장은 제공된 본문에 맞게 자연스럽게 작성하라.

# "${topicTitle}" 연구 분석 보고서

## 서론

### 1. 연구 배경
(주제와 연구 범위를 설명)

### 2. 문제의식 및 분석 필요성
(왜 이 주제를 체계적으로 정리해야 하는지 설명)

## 3. 보고서의 분석 관점
(연구체계 분류, 방법론 분석, 실험결과 분석이라는 세 축을 간결히 제시)

### 4. 보고서 구성
- 1장: ...
- 2장: ...
- 3장: ...

다음은 보고서 본문이다.

${reportText}
`.trim();
}

/**
 * @param {string} targetPath
 * @param {string} prompt
 * @param {ClaudeRunOptions} [options]
 */
async function writeClaudeOutput(targetPath, prompt, options = {}) {
	const maxAttempts = 3;
	let lastOutput = "";

	for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
		const output = await runClaude(prompt, options);
		if (output.trim()) {
			await fs.writeFile(targetPath, `${output.trim()}\n`, "utf8");
			return;
		}

		lastOutput = output;
		if (attempt < maxAttempts) {
			console.warn(
				`[claude] empty output for ${path.relative(projectRoot, targetPath)} (attempt ${attempt}/${maxAttempts}), retrying`,
			);
		}
	}

	throw new Error(
		`Claude returned empty output for ${path.relative(projectRoot, targetPath)} after ${maxAttempts} attempts${lastOutput ? `: ${lastOutput}` : ""}`,
	);
}

/**
 * @param {{
 *   papers: PaperRecord[],
 *   topicDir: string,
 *   outputDir: string,
 *   stageLabel: string,
 *   promptBuilder: (paper: PaperRecord, sourceText: string) => string,
 *   runOptions?: ClaudeRunOptions,
 * }} params
 */
async function createPerPaperOutputs({
	papers,
	topicDir,
	outputDir,
	stageLabel,
	promptBuilder,
	runOptions = {},
}) {
	let created = 0;
	let skippedMissingSource = 0;
	let skippedExisting = 0;
	let skippedMissingSummary = 0;
	const totalFiles = papers.length;
	/** @type {Record<PerPaperStageLabel, string>} */
	const outputKindByStage = {
		"report-01": "전체요약 생성",
		"report-04": "방법론 요약 생성",
		"report-07": "결과 요약 생성",
	};
	const outputKind =
		stageLabel in outputKindByStage
			? outputKindByStage[/** @type {PerPaperStageLabel} */ (stageLabel)]
			: "논문 처리";

	logStageStart(stageLabel, `per-paper outputs (${totalFiles} files)`);

	for (const [index, paper] of papers.entries()) {
		const current = index + 1;
		const targetPath = path.join(outputDir, paper.summary || `${paper.title}.md`);
		logPaperProgress(stageLabel, current, totalFiles, paper, outputKind);
		// logFileProgress(stageLabel, current, totalFiles, targetPath, "process");

		if (!paper.summary) {
			skippedMissingSummary += 1;
			continue;
		}

		const sourcePath = path.join(topicDir, paper.summary);
		if (!(await pathExists(sourcePath))) {
			skippedMissingSource += 1;
			continue;
		}

		// logFileProgress(stageLabel, current, totalFiles, sourcePath, "read");

		if (await pathExists(targetPath)) {
			skippedExisting += 1;
			continue;
		}

		const sourceText = await fs.readFile(sourcePath, "utf8");
		const prompt = promptBuilder(paper, sourceText);
		await writeClaudeOutput(targetPath, prompt, runOptions);
		created += 1;

		console.log(`[${stageLabel}] wrote ${path.relative(projectRoot, targetPath)}`);
	}

	console.log(
		`[${stageLabel}] created=${created} skipped_existing=${skippedExisting} skipped_missing_source=${skippedMissingSource} skipped_missing_summary=${skippedMissingSummary}`,
	);
}

/**
 * @param {{
 *   papers: PaperRecord[],
 *   sourceDir: string,
 *   targetPath: string,
 *   stageLabel: string,
 * }} params
 */
async function mergeStageFiles({
	papers,
	sourceDir,
	targetPath,
	stageLabel,
}) {
	logStageStart(stageLabel, `merge outputs into ${path.relative(projectRoot, targetPath)}`);

	if (await pathExists(targetPath)) {
		console.log(`[${stageLabel}] skip existing ${path.relative(projectRoot, targetPath)}`);
		return;
	}

	const chunks = [];
	const totalFiles = papers.length;

	for (const [index, paper] of papers.entries()) {
		if (!paper.summary) {
			continue;
		}

		const filePath = path.join(sourceDir, paper.summary);
		logFileProgress(stageLabel, index + 1, totalFiles, filePath, "merge");

		if (!(await pathExists(filePath))) {
			continue;
		}

		const content = (await fs.readFile(filePath, "utf8")).trim();
		if (!content) {
			continue;
		}

		chunks.push(content);
	}

	if (chunks.length === 0) {
		throw new Error(`No source files found to merge for ${path.relative(projectRoot, targetPath)}`);
	}

	await fs.writeFile(targetPath, `${chunks.join("\n\n---\n\n")}\n`, "utf8");
	console.log(`[${stageLabel}] wrote ${path.relative(projectRoot, targetPath)}`);
}

/**
 * @param {string} targetPath
 * @param {string} stageLabel
 * @param {string} prompt
 * @param {ClaudeRunOptions} [options]
 */
async function writeIfMissing(targetPath, stageLabel, prompt, options = {}) {
	logStageStart(stageLabel, `single output ${path.relative(projectRoot, targetPath)}`);
	logFileProgress(stageLabel, 1, 1, targetPath, "process");

	if (await pathExists(targetPath)) {
		console.log(`[${stageLabel}] skip existing ${path.relative(projectRoot, targetPath)}`);
		return;
	}

	await writeClaudeOutput(targetPath, prompt, options);
	console.log(`[${stageLabel}] wrote ${path.relative(projectRoot, targetPath)}`);
}

/**
 * @param {ReportDirs} reportDirs
 */
async function stage10(reportDirs) {
	const targetPath = path.join(reportDirs.report10, "report.md");
	const taxonomyPath = path.join(reportDirs.report03, "taxonomy.md");
	const methodSummaryPath = path.join(reportDirs.report06, "method-summary.md");
	const resultSummaryPath = path.join(reportDirs.report09, "result-summary.md");
	const sourcePaths = [taxonomyPath, methodSummaryPath, resultSummaryPath];

	logStageStart("report-10", `final merge into ${path.relative(projectRoot, targetPath)}`);

	await ensureFile(taxonomyPath, "taxonomy.md");
	await ensureFile(methodSummaryPath, "method-summary.md");
	await ensureFile(resultSummaryPath, "result-summary.md");

	for (const [index, filePath] of sourcePaths.entries()) {
		logFileProgress("report-10", index + 1, sourcePaths.length, filePath, "read");
	}

	const contents = await Promise.all([
		fs.readFile(taxonomyPath, "utf8"),
		fs.readFile(methodSummaryPath, "utf8"),
		fs.readFile(resultSummaryPath, "utf8"),
	]);

	const merged = normalizeMarkdown(
		contents.map((content) => content.trim()).join("\n\n---\n\n"),
		{ demoteTopLevelHeadings: true },
	);
	await fs.writeFile(targetPath, `${merged}\n`, "utf8");
	await runMarkdownlintFix(targetPath);
	console.log(`[report-10] wrote ${path.relative(projectRoot, targetPath)}`);
}

/**
 * @param {string} topicTitle
 * @param {ReportDirs} reportDirs
 */
async function stage11(topicTitle, reportDirs) {
	const targetPath = path.join(reportDirs.report11, "introduction.md");
	const reportPath = path.join(reportDirs.report10, "report.md");

	await ensureFile(reportPath, "report.md");

	await writeIfMissing(
		targetPath,
		"report-11",
		buildIntroductionPrompt(topicTitle, await fs.readFile(reportPath, "utf8")),
		{ allowEdits: false },
	);
}

/**
 * @param {ReportDirs} reportDirs
 */
async function stage12(reportDirs) {
	const targetPath = path.join(reportDirs.report12, "final-report.md");
	const introductionPath = path.join(reportDirs.report11, "introduction.md");
	const reportPath = path.join(reportDirs.report10, "report.md");
	const sourcePaths = [introductionPath, reportPath];

	logStageStart("report-12", `final merge into ${path.relative(projectRoot, targetPath)}`);

	await ensureFile(introductionPath, "introduction.md");
	await ensureFile(reportPath, "report.md");

	for (const [index, filePath] of sourcePaths.entries()) {
		logFileProgress("report-12", index + 1, sourcePaths.length, filePath, "read");
	}

	const contents = await Promise.all(sourcePaths.map((filePath) => fs.readFile(filePath, "utf8")));
	const merged = normalizeMarkdown(
		contents
			.map((content) => content.trim())
			.join("\n\n")
			.replace(/\n---\n/g, "\n"),
	);
	await fs.writeFile(targetPath, `${merged}\n`, "utf8");
	await runMarkdownlintFix(targetPath);
	console.log(`[report-12] wrote ${path.relative(projectRoot, targetPath)}`);
}

async function main() {
	const { topicId, stageNumber } = parseArgs(process.argv);

	const topicDir = path.join(docsRoot, topicId);
	const metadataPath = path.join(topicDir, "metadata.json");
	const paperListPath = path.join(topicDir, "paper_list.jsonl");

	const reportDirs = {
		report01: path.join(topicDir, "report-01"),
		report02: path.join(topicDir, "report-02"),
		report03: path.join(topicDir, "report-03"),
		report04: path.join(topicDir, "report-04"),
		report05: path.join(topicDir, "report-05"),
		report06: path.join(topicDir, "report-06"),
		report07: path.join(topicDir, "report-07"),
		report08: path.join(topicDir, "report-08"),
		report09: path.join(topicDir, "report-09"),
		report10: path.join(topicDir, "report-10"),
		report11: path.join(topicDir, "report-11"),
		report12: path.join(topicDir, "report-12"),
	};

	for (const dirPath of Object.values(reportDirs)) {
		await ensureDir(dirPath);
	}

	const summaryMergedPath = path.join(reportDirs.report02, "summary-merged.md");
	const methodMergedPath = path.join(reportDirs.report05, "method-merged.md");
	const resultMergedPath = path.join(reportDirs.report08, "result-merged.md");
	const needsMetadata = stageNumber === null || stageNumber === 3 || stageNumber === 6 || stageNumber === 9 || stageNumber === 11;
	const needsPapers =
		stageNumber === null || stageNumber === 1 || stageNumber === 2 || stageNumber === 4 || stageNumber === 5 || stageNumber === 7 || stageNumber === 8;
	const needsClaude =
		stageNumber === null || stageNumber === 1 || stageNumber === 3 || stageNumber === 4 || stageNumber === 6 || stageNumber === 7 || stageNumber === 9 || stageNumber === 11;

	/** @type {TopicMetadata} */
	let metadata = {};
	let topicTitle = topicId;
	/** @type {PaperRecord[]} */
	let papers = [];

	if (needsMetadata) {
		await ensureFile(metadataPath, "metadata.json");
		metadata = await readJson(metadataPath);
		topicTitle =
			typeof metadata?.title === "string" && metadata.title.trim()
				? metadata.title.trim()
				: topicId;
	}

	if (needsPapers) {
		await ensureFile(paperListPath, "paper_list.jsonl");
		papers = (await readJsonl(paperListPath)).map((record, index) =>
			normalizePaperRecord(record, index, topicId),
		);
	}

	if (needsClaude) {
		await ensureClaudeAvailable();
	}

	if (stageNumber === 1) {
		logStageStart("report-01", "generate per-paper summaries");
		await createPerPaperOutputs({
			papers,
			topicDir,
			outputDir: reportDirs.report01,
			stageLabel: "report-01",
			promptBuilder: buildStage1Prompt,
			runOptions: { allowEdits: false },
		});
		return;
	}

	if (stageNumber === 2) {
		logStageStart("report-02", "merge per-paper summaries");
		await mergeStageFiles({
			papers,
			sourceDir: reportDirs.report01,
			targetPath: summaryMergedPath,
			stageLabel: "report-02",
		});
		return;
	}

	if (stageNumber === 3) {
		logStageStart("report-03", "generate taxonomy");
		await writeIfMissing(
			path.join(reportDirs.report03, "taxonomy.md"),
			"report-03",
			buildTaxonomyPrompt(topicTitle, await fs.readFile(summaryMergedPath, "utf8")),
			{ allowEdits: false },
		);
		return;
	}

	if (stageNumber === 4) {
		logStageStart("report-04", "generate per-paper method summaries");
		await createPerPaperOutputs({
			papers,
			topicDir,
			outputDir: reportDirs.report04,
			stageLabel: "report-04",
			promptBuilder: buildMethodPrompt,
			runOptions: { allowEdits: false },
		});
		return;
	}

	if (stageNumber === 5) {
		logStageStart("report-05", "merge method summaries");
		await mergeStageFiles({
			papers,
			sourceDir: reportDirs.report04,
			targetPath: methodMergedPath,
			stageLabel: "report-05",
		});
		return;
	}

	if (stageNumber === 6) {
		logStageStart("report-06", "generate method summary");
		await writeIfMissing(
			path.join(reportDirs.report06, "method-summary.md"),
			"report-06",
			buildMethodSummaryPrompt(topicTitle, await fs.readFile(methodMergedPath, "utf8")),
			{ allowEdits: false },
		);
		return;
	}

	if (stageNumber === 7) {
		logStageStart("report-07", "generate per-paper result summaries");
		await createPerPaperOutputs({
			papers,
			topicDir,
			outputDir: reportDirs.report07,
			stageLabel: "report-07",
			promptBuilder: buildResultPrompt,
			runOptions: { allowEdits: false },
		});
		return;
	}

	if (stageNumber === 8) {
		logStageStart("report-08", "merge result summaries");
		await mergeStageFiles({
			papers,
			sourceDir: reportDirs.report07,
			targetPath: resultMergedPath,
			stageLabel: "report-08",
		});
		return;
	}

	if (stageNumber === 9) {
		logStageStart("report-09", "generate result summary");
		await writeIfMissing(
			path.join(reportDirs.report09, "result-summary.md"),
			"report-09",
			buildResultSummaryPrompt(topicTitle, await fs.readFile(resultMergedPath, "utf8")),
			{ allowEdits: false },
		);
		return;
	}

	if (stageNumber === 10) {
		await stage10(reportDirs);
		return;
	}

	if (stageNumber === 11) {
		await stage11(topicTitle, reportDirs);
		return;
	}

	if (stageNumber === 12) {
		await stage12(reportDirs);
		return;
	}

	logStageStart("report-01", "generate per-paper summaries");
	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report01,
		stageLabel: "report-01",
		promptBuilder: buildStage1Prompt,
		runOptions: { allowEdits: false },
	});

	logStageStart("report-02", "merge per-paper summaries");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report01,
		targetPath: summaryMergedPath,
		stageLabel: "report-02",
	});

	logStageStart("report-03", "generate taxonomy");
	await writeIfMissing(
		path.join(reportDirs.report03, "taxonomy.md"),
		"report-03",
		buildTaxonomyPrompt(topicTitle, await fs.readFile(summaryMergedPath, "utf8")),
		{ allowEdits: false },
	);

	logStageStart("report-04", "generate per-paper method summaries");
	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report04,
		stageLabel: "report-04",
		promptBuilder: buildMethodPrompt,
		runOptions: { allowEdits: false },
	});

	logStageStart("report-05", "merge method summaries");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report04,
		targetPath: methodMergedPath,
		stageLabel: "report-05",
	});

	logStageStart("report-06", "generate method summary");
	await writeIfMissing(
		path.join(reportDirs.report06, "method-summary.md"),
		"report-06",
		buildMethodSummaryPrompt(topicTitle, await fs.readFile(methodMergedPath, "utf8")),
		{ allowEdits: false },
	);

	logStageStart("report-07", "generate per-paper result summaries");
	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report07,
		stageLabel: "report-07",
		promptBuilder: buildResultPrompt,
		runOptions: { allowEdits: false },
	});

	logStageStart("report-08", "merge result summaries");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report07,
		targetPath: resultMergedPath,
		stageLabel: "report-08",
	});

	logStageStart("report-09", "generate result summary");
	await writeIfMissing(
		path.join(reportDirs.report09, "result-summary.md"),
		"report-09",
		buildResultSummaryPrompt(topicTitle, await fs.readFile(resultMergedPath, "utf8")),
		{ allowEdits: false },
	);

	await stage10(reportDirs);
	await stage11(topicTitle, reportDirs);
	await stage12(reportDirs);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
