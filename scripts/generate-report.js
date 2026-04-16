import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");

function usage() {
	console.error("Usage: node scripts/generate-report.js <topic_id>");
	process.exit(1);
}

async function pathExists(targetPath) {
	try {
		await fs.access(targetPath);
		return true;
	} catch {
		return false;
	}
}

async function ensureDir(targetPath) {
	await fs.mkdir(targetPath, { recursive: true });
}

async function ensureFile(targetPath, label) {
	if (!(await pathExists(targetPath))) {
		throw new Error(`${label} not found: ${path.relative(projectRoot, targetPath)}`);
	}
}

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

async function readJsonl(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
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

function getClaudeCommand() {
	if (process.platform === "win32") {
		return {
			command: "cmd.exe",
			args: ["/d", "/s", "/c", "claude -p --output-format text"],
		};
	}

	return {
		command: "claude",
		args: ["-p", "--output-format", "text"],
	};
}

async function ensureClaudeAvailable() {
	const runner =
		process.platform === "win32"
			? { command: "cmd.exe", args: ["/d", "/s", "/c", "claude --version"] }
			: { command: "claude", args: ["--version"] };

	await new Promise((resolve, reject) => {
		const child = spawn(runner.command, runner.args, {
			stdio: "ignore",
			cwd: projectRoot,
		});

		child.on("error", () => {
			reject(new Error("claude command is not available."));
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve();
				return;
			}
			reject(new Error("claude command is not available."));
		});
	});
}

async function runClaude(prompt) {
	const { command, args } = getClaudeCommand();

	return new Promise((resolve, reject) => {
		const child = spawn(command, args, {
			cwd: projectRoot,
			stdio: ["pipe", "pipe", "pipe"],
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", (chunk) => {
			stdout += String(chunk);
		});

		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});

		child.on("error", (error) => {
			reject(new Error(`Failed to start claude: ${error.message}`));
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve(stdout.trim());
				return;
			}

			const detail = stderr.trim() || stdout.trim() || `exit code ${code}`;
			reject(new Error(`claude failed: ${detail}`));
		});

		child.stdin.end(prompt);
	});
}

function normalizePaperRecord(record, index, topicId) {
	if (typeof record?.title !== "string" || !record.title.trim()) {
		throw new Error(`Invalid paper record at entry ${index + 1} for topic "${topicId}": missing title`);
	}

	if (!Number.isInteger(record?.year)) {
		throw new Error(`Invalid paper record at entry ${index + 1} for topic "${topicId}": missing or invalid year`);
	}

	return {
		title: record.title.trim(),
		year: record.year,
		summary: typeof record?.summary === "string" ? record.summary.trim() : "",
	};
}

function buildStage1Prompt(paper, sourceText) {
	return `다음은 논문 상세 분석 보고서다.

이 보고서를 바탕으로 한국어 마크다운 요약문을 작성하라.
반드시 아래 규칙을 지켜라.

1. 첫 줄은 정확히 \`# ${paper.title}(${paper.year})\` 로 작성한다.
2. 전체 분량은 첫 줄을 포함해 10줄 내외로 유지한다.
3. 각 줄은 짧고 밀도 있게 작성한다.
4. 논문의 연구 문제, 핵심 아이디어, 방법론, 주요 결과, 의의를 빠뜨리지 않는다.
5. 원문에 없는 내용을 추측해서 쓰지 않는다.
6. 불필요한 서론, 결론, 메타 설명은 쓰지 않는다.

상세 분석 보고서:

${sourceText}
`;
}

function buildTaxonomyPrompt(topicTitle, mergedText) {
	return `다음은 "${topicTitle}" 주제의 논문 요약 모음이다.

이 자료를 바탕으로 논문들을 체계적으로 분류한 taxonomy 문서를 한국어 마크다운으로 작성하라.
반드시 다음 규칙을 지켜라.

1. 제목은 \`# ${topicTitle} Taxonomy\` 로 시작한다.
2. 먼저 분류 원칙을 짧게 설명한다.
3. 그다음 대분류와 하위 범주를 체계적으로 제시한다.
4. 각 범주마다 해당 범주에 속하는 논문들을 bullet list로 나열한다.
5. 왜 그런 분류가 타당한지 간단히 설명한다.
6. 마지막에는 전체 연구 지형을 한 단락으로 정리한다.
7. 요약문에 없는 내용은 추정하지 않는다.

논문 요약 모음:

${mergedText}
`;
}

function buildMethodPrompt(paper, sourceText) {
	return `다음은 논문 상세 분석 보고서다.

이 보고서에서 방법론에 해당하는 내용만 정리하여 한국어 마크다운 문서를 작성하라.
반드시 다음 규칙을 지켜라.

1. 첫 줄은 정확히 \`# ${paper.title}(${paper.year})\` 로 작성한다.
2. 방법론 중심으로만 작성하고 배경 설명은 최소화한다.
3. 논문이 해결하려는 문제 설정, 제안 방법의 핵심 구성, 수식/구조/알고리즘/학습 절차가 있으면 포함한다.
4. 실험 결과나 일반적인 의의 설명은 핵심 방법 이해에 필요한 최소한만 언급한다.
5. 원문에 없는 세부 구현은 추측하지 않는다.

상세 분석 보고서:

${sourceText}
`;
}

function buildMethodSummaryPrompt(topicTitle, mergedText) {
	return `다음은 "${topicTitle}" 주제 논문들의 방법론 정리 문서 모음이다.

이 자료를 바탕으로 방법론 전반을 종합 분석한 한국어 마크다운 문서를 작성하라.
반드시 다음 규칙을 지켜라.

1. 제목은 \`# ${topicTitle} Method Summary\` 로 시작한다.
2. 공통 문제 설정, 대표적인 방법론 계열, 설계 패턴, 차별점, 장단점을 구조적으로 정리한다.
3. 유사한 방법끼리 묶고 서로 대비되는 방법은 비교해서 설명한다.
4. 마지막에는 방법론 흐름의 변화와 향후 연구 여지를 짧게 정리한다.
5. 제공된 자료에 없는 내용은 추정하지 않는다.

방법론 문서 모음:

${mergedText}
`;
}

function buildResultPrompt(paper, sourceText) {
	return `다음은 논문 상세 분석 보고서다.

이 보고서에서 실험 결과에 해당하는 내용만 정리하여 한국어 마크다운 문서를 작성하라.
반드시 다음 규칙을 지켜라.

1. 첫 줄은 정확히 \`# ${paper.title}(${paper.year})\` 로 작성한다.
2. 사용 데이터셋, 비교 기준, 핵심 수치 결과, 저자 해석을 분명히 정리한다.
3. 정량 결과가 있으면 가능한 한 수치 중심으로 쓴다.
4. 방법론 설명은 실험 결과를 이해하는 데 필요한 최소한만 포함한다.
5. 제공된 보고서에 없는 결과를 추측하지 않는다.

상세 분석 보고서:

${sourceText}
`;
}

function buildResultSummaryPrompt(topicTitle, mergedText) {
	return `다음은 "${topicTitle}" 주제 논문들의 실험 결과 정리 문서 모음이다.

이 자료를 바탕으로 실험 결과 전반을 종합 분석한 한국어 마크다운 문서를 작성하라.
반드시 다음 규칙을 지켜라.

1. 제목은 \`# ${topicTitle} Result Summary\` 로 시작한다.
2. 어떤 평가 축이 주로 사용되었는지 먼저 정리한다.
3. 논문들이 보고한 성능 개선 패턴, 일관된 경향, 상충되는 결과를 구조적으로 설명한다.
4. 실험 설계의 한계나 비교상의 주의점이 드러나면 함께 정리한다.
5. 마지막에는 전체 실험 결과가 시사하는 바를 짧게 정리한다.
6. 제공된 자료에 없는 내용은 추정하지 않는다.

실험 결과 문서 모음:

${mergedText}
`;
}

async function writeClaudeOutput(targetPath, prompt) {
	const output = await runClaude(prompt);
	if (!output.trim()) {
		throw new Error(`Claude returned empty output for ${path.relative(projectRoot, targetPath)}`);
	}
	await fs.writeFile(targetPath, `${output.trim()}\n`, "utf8");
}

async function createPerPaperOutputs({
	papers,
	topicDir,
	outputDir,
	stageLabel,
	promptBuilder,
}) {
	let created = 0;
	let skippedMissingSource = 0;
	let skippedExisting = 0;
	let skippedMissingSummary = 0;

	for (const paper of papers) {
		if (!paper.summary) {
			skippedMissingSummary += 1;
			continue;
		}

		const sourcePath = path.join(topicDir, paper.summary);
		if (!(await pathExists(sourcePath))) {
			skippedMissingSource += 1;
			continue;
		}

		const targetPath = path.join(outputDir, paper.summary);
		if (await pathExists(targetPath)) {
			skippedExisting += 1;
			continue;
		}

		const sourceText = await fs.readFile(sourcePath, "utf8");
		const prompt = promptBuilder(paper, sourceText);
		await writeClaudeOutput(targetPath, prompt);
		created += 1;
		console.log(`[${stageLabel}] wrote ${path.relative(projectRoot, targetPath)}`);
	}

	console.log(
		`[${stageLabel}] created=${created} skipped_existing=${skippedExisting} skipped_missing_source=${skippedMissingSource} skipped_missing_summary=${skippedMissingSummary}`,
	);
}

async function mergeStageFiles({
	papers,
	sourceDir,
	targetPath,
	stageLabel,
}) {
	if (await pathExists(targetPath)) {
		console.log(`[${stageLabel}] skip existing ${path.relative(projectRoot, targetPath)}`);
		return;
	}

	const chunks = [];
	for (const paper of papers) {
		if (!paper.summary) {
			continue;
		}

		const filePath = path.join(sourceDir, paper.summary);
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

async function writeIfMissing(targetPath, stageLabel, prompt) {
	if (await pathExists(targetPath)) {
		console.log(`[${stageLabel}] skip existing ${path.relative(projectRoot, targetPath)}`);
		return;
	}

	await writeClaudeOutput(targetPath, prompt);
	console.log(`[${stageLabel}] wrote ${path.relative(projectRoot, targetPath)}`);
}

async function stage10(reportDirs) {
	const targetPath = path.join(reportDirs.report10, "report.md");
	const taxonomyPath = path.join(reportDirs.report03, "taxonomy.md");
	const methodSummaryPath = path.join(reportDirs.report06, "method-summary.md");
	const resultSummaryPath = path.join(reportDirs.report09, "result-summary.md");

	await ensureFile(taxonomyPath, "taxonomy.md");
	await ensureFile(methodSummaryPath, "method-summary.md");
	await ensureFile(resultSummaryPath, "result-summary.md");

	const contents = await Promise.all([
		fs.readFile(taxonomyPath, "utf8"),
		fs.readFile(methodSummaryPath, "utf8"),
		fs.readFile(resultSummaryPath, "utf8"),
	]);

	const merged = contents.map((content) => content.trim()).join("\n\n---\n\n");
	await fs.writeFile(targetPath, `${merged}\n`, "utf8");
	console.log(`[report-10] wrote ${path.relative(projectRoot, targetPath)}`);
}

async function main() {
	const topicId = process.argv[2];
	if (!topicId || process.argv.length > 3) {
		usage();
	}

	const topicDir = path.join(docsRoot, topicId);
	const metadataPath = path.join(topicDir, "metadata.json");
	const paperListPath = path.join(topicDir, "paper_list.jsonl");

	await ensureFile(metadataPath, "metadata.json");
	await ensureFile(paperListPath, "paper_list.jsonl");
	await ensureClaudeAvailable();

	const metadata = await readJson(metadataPath);
	const topicTitle =
		typeof metadata?.title === "string" && metadata.title.trim()
			? metadata.title.trim()
			: topicId;
	const papers = (await readJsonl(paperListPath)).map((record, index) =>
		normalizePaperRecord(record, index, topicId),
	);

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
	};

	for (const dirPath of Object.values(reportDirs)) {
		await ensureDir(dirPath);
	}

	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report01,
		stageLabel: "report-01",
		promptBuilder: buildStage1Prompt,
	});

	const summaryMergedPath = path.join(reportDirs.report02, "summary-merged.md");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report01,
		targetPath: summaryMergedPath,
		stageLabel: "report-02",
	});

	await writeIfMissing(
		path.join(reportDirs.report03, "taxonomy.md"),
		"report-03",
		buildTaxonomyPrompt(topicTitle, await fs.readFile(summaryMergedPath, "utf8")),
	);

	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report04,
		stageLabel: "report-04",
		promptBuilder: buildMethodPrompt,
	});

	const methodMergedPath = path.join(reportDirs.report05, "method-merged.md");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report04,
		targetPath: methodMergedPath,
		stageLabel: "report-05",
	});

	await writeIfMissing(
		path.join(reportDirs.report06, "method-summary.md"),
		"report-06",
		buildMethodSummaryPrompt(topicTitle, await fs.readFile(methodMergedPath, "utf8")),
	);

	await createPerPaperOutputs({
		papers,
		topicDir,
		outputDir: reportDirs.report07,
		stageLabel: "report-07",
		promptBuilder: buildResultPrompt,
	});

	const resultMergedPath = path.join(reportDirs.report08, "result-merged.md");
	await mergeStageFiles({
		papers,
		sourceDir: reportDirs.report07,
		targetPath: resultMergedPath,
		stageLabel: "report-08",
	});

	await writeIfMissing(
		path.join(reportDirs.report09, "result-summary.md"),
		"report-09",
		buildResultSummaryPrompt(topicTitle, await fs.readFile(resultMergedPath, "utf8")),
	);

	await stage10(reportDirs);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
