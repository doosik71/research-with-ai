import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");

async function pathExists(targetPath) {
	try {
		await fs.access(targetPath);
		return true;
	} catch {
		return false;
	}
}

async function findTopicDirs(rootDir) {
	const entries = await fs.readdir(rootDir, { withFileTypes: true });
	return entries
		.filter((entry) => entry.isDirectory())
		.map((entry) => path.join(rootDir, entry.name));
}

async function ensureCodexAvailable() {
	return new Promise((resolve, reject) => {
		const child = spawn("codex", ["--version"], {
			stdio: "ignore",
			shell: true,
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

async function parsePaperList(jsonlPath) {
	const raw = await fs.readFile(jsonlPath, "utf8");
	const records = [];
	const errors = [];

	for (const [index, rawLine] of raw.split(/\r?\n/).entries()) {
		const lineNumber = index + 1;
		const line = rawLine.trim();
		if (!line || line.startsWith("//")) {
			continue;
		}

		try {
			records.push(JSON.parse(line));
		} catch (error) {
			errors.push({
				file: jsonlPath,
				lineNumber,
				message: error instanceof Error ? error.message : String(error),
			});
		}
	}

	return { records, errors };
}

async function collectMissingSummaries() {
	const topicDirs = await findTopicDirs(docsRoot);
	const missingTargets = [];
	let parseErrorCount = 0;

	for (const topicDir of topicDirs) {
		const metadataPath = path.join(topicDir, "metadata.json");
		if (!(await pathExists(metadataPath))) {
				continue;
		}

		const paperListPath = path.join(topicDir, "paper_list.jsonl");
		if (!(await pathExists(paperListPath))) {
			console.warn(
				`[skip] paper_list.jsonl not found: ${path.relative(projectRoot, topicDir)}`,
			);
			continue;
		}

		const { records, errors } = await parsePaperList(paperListPath);
		for (const error of errors) {
			parseErrorCount += 1;
			console.warn(
				`[invalid-jsonl] ${path.relative(projectRoot, error.file)}:${error.lineNumber} ${error.message}`,
			);
		}

		for (const record of records) {
			const summary = typeof record?.summary === "string" ? record.summary.trim() : "";
			if (!summary) {
				continue;
			}

			const summaryPath = path.join(topicDir, summary);
			if (await pathExists(summaryPath)) {
				continue;
			}

			const topic = path.basename(topicDir);
			const summaryBase = summary.endsWith(".md") ? summary.slice(0, -3) : summary;
			missingTargets.push({
				topic,
				title: typeof record?.title === "string" ? record.title : "",
				summary,
				targetArg: `${topic}/${summaryBase}`,
			});
		}
	}

	return { missingTargets, parseErrorCount };
}

async function runUpdateSummary(targetArg) {
	return new Promise((resolve, reject) => {
		const child = spawn(
			process.execPath,
			[path.join(__dirname, "update-summary.js"), targetArg],
			{
				stdio: "inherit",
			},
		);

		child.on("error", (error) => {
			reject(error);
		});

		child.on("close", (code) => {
			if (code === 0) {
				resolve();
				return;
			}
			reject(new Error(`update-summary.js exited with code ${code}`));
		});
	});
}

async function main() {
	await ensureCodexAvailable();
	const { missingTargets, parseErrorCount } = await collectMissingSummaries();

	console.log(
		`[scan-complete] missing=${missingTargets.length} invalid_jsonl=${parseErrorCount}`,
	);

	if (missingTargets.length === 0) {
		console.log("[done] No missing summary reports found.");
		return;
	}

	for (const target of missingTargets) {
		console.log(
			`[generate] ${target.targetArg}${target.title ? ` (${target.title})` : ""}`,
		);

		try {
			await runUpdateSummary(target.targetArg);
			console.log(`[generated] ${target.targetArg}`);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			console.error(
				`[failed] ${target.targetArg}${target.title ? ` (${target.title})` : ""}`,
			);
			console.error(`[stop] ${message}`);
			process.exitCode = 1;
			return;
		}
	}

	console.log(`[done] Generated ${missingTargets.length} missing summary report(s).`);
}

main().catch((error) => {
	const message = error instanceof Error ? error.stack ?? error.message : String(error);
	console.error(`[error] ${message}`);
	process.exit(1);
});
