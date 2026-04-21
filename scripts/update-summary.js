/// <reference types="node" />

import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
// pdfjs needs a fetch implementation that can read file:// URLs.
// We patch fetch before dynamically importing pdfjs.

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const useShellForCodex = process.platform === "win32";

const reportPrompt = `# 논문 상세 분석 보고서 작성 프롬프트

## 역할 정의

당신은 컴퓨터 비전 및 딥러닝 분야의 전문가 reviewer이다.
당신은 사용자가 제시한 논문에 대한 학술 수준의 상세 분석 보고서를 생성한다.

## 논문 읽기 및 분석 규칙

논문 읽기 시:

- 제3자 요약이 아닌 원문 내용을 직접 참고하라.
- 한국어 번역으로 인해 정확성이 떨어질 수 있는 경우, 전문 용어는 영어 원문을 그대로 사용하라.
- 제목과 내용이 다르더라도, 섹션 구조를 추론하라.
- 연구 문제, 핵심 아이디어, 그리고 상세한 방법 설명은 필수적이므로 특히 주의 깊게 살펴보라.
- 방법론의 핵심인 방정식, 아키텍처, 손실 함수, 목표 변수, 그리고 학습 절차는 쉬운 한국어로 설명하라.
- 비교 대상, 측정 방법, 그리고 결과의 중요성을 이해할 수 있도록 실험 내용을 충분한 맥락과 함께 설명하라.
- 논문에 명확하게 명시되지 않은 내용은 추측하지 말고 명시적으로 언급하라.

## 보고서 요구 사항

보고서는 원문을 읽지 않고도 내용을 이해할 수 있도록 충분히 상세해야 한다.

### 출력 언어 및 어조

- 한국어로 작성한다.
- 전문 용어는 필요한 경우에만 영어로 작성한다.
- 간결한 목록 형식보다는 정확하고 설명적인 서술형 문장을 사용하라.

### 필수 구조

생성된 보고서의 첫 줄은 반드시 \`h1\` 헤더 형식의 논문 제목이어야 한다.

\`\`\`markdown
# <영어 논문 제목>

- **저자**: {authors}
- **발표연도**: {published_year}
- **arXiv**: {arxiv_url}

다음 섹션들을 적절한 순서로 작성한다.

## 1. 논문 개요

포함해야 하는 내용:

- 논문의 목표에 대한 요약
- 연구 문제
- 문제의 중요성

## 2. 핵심 아이디어

포함해야 하는 내용:

- 중심적인 직관 또는 설계 아이디어
- 기존 접근 방식과의 차별점 (논문에서 명확히 제시된 경우)

## 3. 상세 방법 설명

포함해야 하는 내용 (있는 경우):

- 전체 파이프라인 또는 시스템 구조
- 각 주요 구성 요소 및 역할
- 관련성이 있는 경우 훈련 목표, 손실 함수, 추론 절차 또는 알고리즘 흐름
- 주요 방정식 설명

## 4. 실험 및 결과

포함해야 하는 내용 (있는 경우):

- 데이터셋, 작업, 기준선, 지표
- 주요 정량적 또는 정성적 결과
- 실험의 실제 결과

## 5. 강점, 한계

포함해야 하는 내용:

- 논문에서 뒷받침되는 강점
- 한계, 가정 또는 미해결 질문
- 논문에 근거한 간략한 비판적 해석

## 6. 결론

포함해야 하는 내용:

- 논문의 주요 기여 사항에 대한 요약
- 이 연구가 실제 적용이나 향후 연구에 중요한 역할을 할 가능성
\`\`\`

## 수학 공식의 서식 요건

수식을 작성할 때:

- 본문 내(inline) 수식은 \`$...$\` 태그로 표시한다.
- 블록(block) 수식은 \`$$...$$\` 태그로 표시한다.

## 분석 보고서 품질 검사

마무리 전:
- 제목 라벨이 일관적인지 확인한다.
- 코드 펜스가 균형 있게 배치되었는지 확인한다.
- 목록 형식이 올바른지 확인한다.
- 수학 구분 기호가 균형 있게 사용되었는지 확인한다.
- 마크다운 lint 오류가 명백한 경우 수정한다.
- 용어가 일관되게 사용되었는지 확인한다.
- 실제로 접근할 수 없는 부분을 읽었다고 주장하지 않는다.

## 논문에서 추출한 텍스트

- 논문에서 추출한 텍스트는 다음과 같다.

---`;


function ensureFileFetch() {
	const originalFetch = globalThis.fetch;
	if (!originalFetch) {
		throw new Error("Global fetch is not available in this Node.js runtime.");
	}
	globalThis.fetch = async (input, init) => {
		const url = typeof input === "string" ? input : input?.url;
		if (url && url.startsWith("file://")) {
			const filePath = fileURLToPath(url);
			const data = await fs.readFile(filePath);
			return new Response(data);
		}
		return originalFetch(input, init);
	};
}

function usage() {
	console.error("Usage: node update.js <folder/summary>");
	process.exit(1);
}

function help() {
	console.log("Usage: node scripts/update-summary.js <folder/summary>");
	console.log("");
	console.log("Generates a summary markdown file for one paper using the matching record in paper_list.jsonl.");
}

function normalizeArg(arg) {
	return arg.split(/[\\/]+/).filter(Boolean);
}

async function ensureCodexAvailable() {
	return new Promise((resolve, reject) => {
		const child = spawn("codex", ["--version"], {
			stdio: "ignore",
			shell: useShellForCodex,
		});

		child.on("error", () => {
			reject(
				new Error(
					"codex command is not available. Install it with: npm i -g @openai/codex",
				),
			);
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve();
				return;
			}

			reject(
				new Error(
					"codex command is not available. Install it with: npm i -g @openai/codex",
				),
			);
		});
	});
}

async function readJsonl(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
	const records = [];
	for (const rawLine of raw.split(/\r?\n/)) {
		const line = rawLine.trim();
		if (!line || line.startsWith("//")) {
			continue;
		}
		try {
			records.push(JSON.parse(line));
		} catch {
			throw new Error(`Invalid JSONL line in ${filePath}`);
		}
	}
	return records;
}

async function extractPdfText(pdfUrl) {
	ensureFileFetch();
	const { getDocument } = await import("pdfjs-dist/legacy/build/pdf.mjs");
	const res = await fetch(pdfUrl);
	if (!res.ok) {
		throw new Error(`Failed to fetch PDF: ${res.status} ${res.statusText}`);
	}
	const standardFontDir = path.join(
		projectRoot,
		"node_modules",
		"pdfjs-dist",
		"standard_fonts",
	);
	const standardFontDataUrl = standardFontDir.replace(/\\/g, "/") + "/";
	const arrayBuffer = await res.arrayBuffer();
	const loadingTask = getDocument({
		data: new Uint8Array(arrayBuffer),
		standardFontDataUrl,
	});
	let pdf;
	try {
		pdf = await loadingTask.promise;
	} catch (err) {
		throw new Error(`Invalid PDF: ${err instanceof Error ? err.message : err}`);
	}
	if (!pdf?.numPages || pdf.numPages < 1) {
		throw new Error("Invalid PDF: no pages found.");
	}
	let text = "";
	for (let pageNum = 1; pageNum <= pdf.numPages; pageNum += 1) {
		const page = await pdf.getPage(pageNum);
		const content = await page.getTextContent();
		const pageText = content.items.map((item) => item.str).join(" ");
		text += `${pageText}\n\n`;
	}
	return text.trim();
}

async function runCodex(promptPath, outputPath) {
	return new Promise((resolve, reject) => {
		const command = "codex";
		const args = ["exec", "-", "--model", "gpt-5.4", "--output-last-message", outputPath];
		const child = spawn(command, args, {
			stdio: ["pipe", "pipe", "pipe"],
			shell: useShellForCodex,
		});
		child.on("error", (err) => {
			reject(new Error(`Failed to start codex: ${err.message}`));
		});
		child.stdout.on("data", (chunk) => {
			process.stdout.write(chunk);
		});
		child.stderr.on("data", (chunk) => {
			process.stderr.write(chunk);
		});
		child.on("close", (code) => {
			if (code === 0) {
				resolve();
			} else {
				reject(new Error(`codex exited with code ${code}`));
			}
		});

		createReadStream(promptPath).pipe(child.stdin);
	});
}

async function main() {
	if (process.argv.includes("--help") || process.argv.includes("-h")) {
		help();
		return;
	}

	const arg = process.argv[2];
	if (!arg) usage();
	await ensureCodexAvailable();

	const parts = normalizeArg(arg);
	if (parts.length < 2) {
		console.error("Argument must be in the form <folder/summary>");
		process.exit(1);
	}

	let summary = parts.pop();
	if (!summary.endsWith(".md")) {
		summary = `${summary}.md`;
	}
	const folderRel = parts.join("/");
	const docsDir = path.join(projectRoot, "static", "docs", folderRel);
	const jsonlPath = path.join(docsDir, "paper_list.jsonl");

	const records = await readJsonl(jsonlPath);
	const record = records.find((r) => r?.summary === summary);
	if (!record) {
		throw new Error(`No record found with summary: ${summary}`);
	}
	if (!record.url) {
		throw new Error(`Record missing url for summary: ${summary}`);
	}

	const pdfUrl = String(record.url).replace("/abs/", "/pdf/");
	const extractedText = await extractPdfText(pdfUrl);
	if (extractedText.length < 1000) {
		throw new Error(
			`Invalid extracted text: length ${extractedText.length} is below 1000.`,
		);
	}

	const finalText = `${reportPrompt.trim()}\n\n${extractedText}\n`;
	const tempDir = path.join(projectRoot, "temp");
	await fs.mkdir(tempDir, { recursive: true });
	const outPath = path.join(tempDir, "prompt.txt");
	await fs.writeFile(outPath, finalText, "utf8");

	console.log(`Wrote ${outPath}`);

	const codexOutputPath = path.join(tempDir, "output.txt");
	await runCodex(outPath, codexOutputPath);
	console.log(`Wrote ${codexOutputPath}`);

	const summaryText = await fs.readFile(codexOutputPath, "utf8");
	const summaryLength = summaryText.trim().length;
	if (summaryLength < 1000) {
		throw new Error(`Invalid summary: length ${summaryLength} is below 1000.`);
	}

	const summaryPath = path.join(docsDir, summary);
	await fs.writeFile(summaryPath, summaryText + "\n", "utf8");
	// console.log(`Wrote ${summaryPath}`);
}

main().catch((err) => {
	console.error(err instanceof Error ? err.message : err);
	process.exit(1);
});
