/// <reference types="node" />

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");

function usage() {
	console.error("Usage: node scripts/update-index.js [topic]");
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

async function parsePaperList(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
	const records = [];

	for (const [index, rawLine] of raw.split(/\r?\n/).entries()) {
		const lineNumber = index + 1;
		const trimmed = rawLine.trim();
		if (!trimmed || trimmed.startsWith("//")) {
			continue;
		}

		let record;
		try {
			record = JSON.parse(trimmed);
		} catch (error) {
			throw new Error(
				`Invalid JSONL at ${path.relative(projectRoot, filePath)}:${lineNumber} (${error instanceof Error ? error.message : String(error)})`,
			);
		}

		records.push(record);
	}

	return records;
}

function comparePapers(left, right) {
	if (left.year !== right.year) {
		return right.year - left.year;
	}

	return left.title.localeCompare(right.title);
}

function renderIndex(topic, papers) {
	const lines = [`# ${topic}`, ""];
	let currentYear = null;

	for (const paper of papers) {
		if (paper.year !== currentYear) {
			if (currentYear !== null) {
				lines.push("");
			}

			currentYear = paper.year;
			lines.push(`## ${paper.year}`, "");
		}

		lines.push(`- [${paper.title}](${paper.summary}) ([arXiv](${paper.url}))`);
	}

	return `${lines.join("\n").trimEnd()}\n`;
}

function validatePaper(record, topic, index) {
	const prefix = `Invalid paper record for topic "${topic}" at entry ${index + 1}`;

	if (typeof record?.title !== "string" || !record.title.trim()) {
		throw new Error(`${prefix}: missing title`);
	}
	if (!Number.isInteger(record?.year)) {
		throw new Error(`${prefix}: missing or invalid year`);
	}
	if (typeof record?.summary !== "string" || !record.summary.trim()) {
		throw new Error(`${prefix}: missing summary`);
	}
	if (typeof record?.url !== "string" || !record.url.trim()) {
		throw new Error(`${prefix}: missing url`);
	}

	return {
		title: record.title.trim(),
		year: record.year,
		summary: record.summary.trim(),
		url: record.url.trim(),
	};
}

async function writeTopicIndex(topic) {
	const topicDir = path.join(docsRoot, topic);
	if (!(await pathExists(topicDir))) {
		throw new Error(`Topic folder not found: static/docs/${topic}`);
	}

	const metadataPath = path.join(topicDir, "metadata.json");
	const paperListPath = path.join(topicDir, "paper_list.jsonl");
	const indexPath = path.join(topicDir, "index.md");

	if (!(await pathExists(metadataPath))) {
		throw new Error(`metadata.json not found: static/docs/${topic}/metadata.json`);
	}
	if (!(await pathExists(paperListPath))) {
		throw new Error(`paper_list.jsonl not found: static/docs/${topic}/paper_list.jsonl`);
	}

	const metadata = await readJson(metadataPath);
	const topicTitle =
		typeof metadata?.title === "string" ? metadata.title.trim() : "";
	if (!topicTitle) {
		throw new Error(`Missing title in static/docs/${topic}/metadata.json`);
	}

	const papers = (await parsePaperList(paperListPath))
		.map((record, index) => validatePaper(record, topic, index))
		.sort(comparePapers);

	const content = renderIndex(topicTitle, papers);
	await fs.writeFile(indexPath, content, "utf8");

	console.log(`[written] ${path.relative(projectRoot, indexPath)}`);
}

async function findTopicsWithMetadata() {
	const entries = await fs.readdir(docsRoot, { withFileTypes: true });
	const topics = [];

	for (const entry of entries) {
		if (!entry.isDirectory()) {
			continue;
		}

		const metadataPath = path.join(docsRoot, entry.name, "metadata.json");
		if (await pathExists(metadataPath)) {
			topics.push(entry.name);
		}
	}

	return topics.sort((left, right) => left.localeCompare(right));
}

async function main() {
	const topic = process.argv[2];
	if (process.argv.length > 3) {
		usage();
	}

	if (topic) {
		await writeTopicIndex(topic);
		return;
	}

	const topics = await findTopicsWithMetadata();
	if (topics.length === 0) {
		console.log("[done] No topic directories with metadata.json were found.");
		return;
	}

	let hasError = false;

	for (const topicName of topics) {
		try {
			await writeTopicIndex(topicName);
		} catch (error) {
			hasError = true;
			console.error(
				`[failed] ${topicName}: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
	}

	if (hasError) {
		process.exit(1);
	}
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
