import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");

function usage() {
	console.error("Usage: node scripts/add-doc.js <topic> <arxiv-abs-url>");
	process.exit(1);
}

function help() {
	console.log("Usage: node scripts/add-doc.js <topic> <arxiv-abs-url>");
	console.log("");
	console.log("Fetches arXiv metadata and appends a paper record to static/docs/<topic>/paper_list.jsonl.");
}

async function pathExists(targetPath) {
	try {
		await fs.access(targetPath);
		return true;
	} catch {
		return false;
	}
}

function normalizeArxivUrl(url) {
	try {
		const parsed = new URL(url);
		if (!/arxiv\.org$/i.test(parsed.hostname)) {
			throw new Error("URL must point to arxiv.org");
		}
		if (!parsed.pathname.startsWith("/abs/")) {
			throw new Error("URL must be an arXiv abs page URL");
		}
		return `https://arxiv.org${parsed.pathname}`;
	} catch (error) {
		throw new Error(
			error instanceof Error ? error.message : "Invalid arXiv URL provided",
		);
	}
}

function decodeHtmlEntities(text) {
	return text
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")
		.replace(/&#x27;/gi, "'")
		.replace(/&#x2F;/gi, "/");
}

function escapeRegex(text) {
	return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function readMetaContent(html, name) {
	const pattern = new RegExp(
		`<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
		"i",
	);
	const match = html.match(pattern);
	return match ? decodeHtmlEntities(match[3].trim()) : "";
}

function readMetaContents(html, name) {
	const pattern = new RegExp(
		`<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
		"gi",
	);
	const values = [];
	let match;
	while ((match = pattern.exec(html)) !== null) {
		values.push(decodeHtmlEntities(match[3].trim()));
	}
	return values;
}

function createSummaryFilename(title) {
	const ascii = title.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
	const summaryBase = ascii
		.toLowerCase()
		.replace(/\s+/g, "_")
		.replace(/[^a-z0-9_]/g, "")
		.replace(/_+/g, "_")
		.replace(/^_+|_+$/g, "");

	if (!summaryBase) {
		throw new Error("Failed to derive a valid summary filename from the paper title.");
	}

	return `${summaryBase}.md`;
}

function normalizeAuthorName(name) {
	const parts = name
		.split(",")
		.map((part) => part.trim())
		.filter(Boolean);

	if (parts.length <= 1) {
		return name.trim();
	}

	return `${parts.slice(1).join(" ")} ${parts[0]}`.trim();
}

async function parsePaperList(filePath) {
	const raw = await fs.readFile(filePath, "utf8");
	const entries = [];
	const lines = raw.split(/\r?\n/);

	for (const [index, rawLine] of lines.entries()) {
		const lineNumber = index + 1;
		const trimmed = rawLine.trim();
		if (!trimmed) {
			continue;
		}

		const jsonText = trimmed.startsWith("//")
			? trimmed.slice(2).trim()
			: trimmed;

		if (!jsonText) {
			continue;
		}

		try {
			entries.push({
				record: JSON.parse(jsonText),
				lineNumber,
				commented: trimmed.startsWith("//"),
				rawLine,
			});
		} catch (error) {
			throw new Error(
				`Invalid JSONL at ${path.relative(projectRoot, filePath)}:${lineNumber} (${error instanceof Error ? error.message : String(error)})`,
			);
		}
	}

	return {
		lines,
		rawWithTrailingNewline: raw.endsWith("\n") ? raw : `${raw}\n`,
		entries,
	};
}

async function fetchPaperInfo(url) {
	const response = await fetch(url);
	if (!response.ok) {
		throw new Error(`Failed to fetch arXiv page: ${response.status} ${response.statusText}`);
	}

	const html = await response.text();
	const title = readMetaContent(html, "citation_title");
	const authors = readMetaContents(html, "citation_author");
	const publishedDate =
		readMetaContent(html, "citation_date") ||
		readMetaContent(html, "citation_online_date") ||
		readMetaContent(html, "citation_publication_date");

	const yearMatch = publishedDate.match(/\b(\d{4})\b/);
	const year = yearMatch ? Number(yearMatch[1]) : NaN;

	if (!title) {
		throw new Error("Failed to parse title from the arXiv page.");
	}
	if (authors.length === 0) {
		throw new Error("Failed to parse authors from the arXiv page.");
	}
	if (!Number.isFinite(year)) {
		throw new Error("Failed to parse publication year from the arXiv page.");
	}

	return {
		title,
		author: authors.map(normalizeAuthorName).join(", "),
		year,
	};
}

async function main() {
	if (process.argv.includes("--help") || process.argv.includes("-h")) {
		help();
		return;
	}

	const topic = process.argv[2];
	const inputUrl = process.argv[3];
	if (!topic || !inputUrl) {
		usage();
	}

	const topicDir = path.join(docsRoot, topic);
	if (!(await pathExists(topicDir))) {
		throw new Error(`Topic folder not found: static/docs/${topic}`);
	}

	const paperListPath = path.join(topicDir, "paper_list.jsonl");
	if (!(await pathExists(paperListPath))) {
		throw new Error(`paper_list.jsonl not found: static/docs/${topic}/paper_list.jsonl`);
	}

	const canonicalUrl = normalizeArxivUrl(inputUrl);
	const { lines, rawWithTrailingNewline, entries } = await parsePaperList(paperListPath);

	const existingByUrl = entries.find((entry) => {
		const url = typeof entry.record?.url === "string" ? entry.record.url.trim() : "";
		if (!url) {
			return false;
		}

		try {
			return normalizeArxivUrl(url) === canonicalUrl;
		} catch {
			return url === canonicalUrl;
		}
	});

	if (existingByUrl) {
		if (existingByUrl.commented) {
			const lineIndex = existingByUrl.lineNumber - 1;
			lines[lineIndex] = existingByUrl.rawLine.replace(/^(\s*)\/\/\s?/, "$1");
			await fs.writeFile(paperListPath, `${lines.join("\n")}\n`, "utf8");
			console.log(
				`[uncommented] ${path.relative(projectRoot, paperListPath)}:${existingByUrl.lineNumber}`,
			);
			return;
		}

		console.log(`[exists] ${canonicalUrl}`);
		process.exit(0);
	}

	const paperInfo = await fetchPaperInfo(canonicalUrl);
	const summary = createSummaryFilename(paperInfo.title);
	const duplicateSummary = entries.find((entry) => entry.record?.summary === summary);
	if (duplicateSummary) {
		throw new Error(
			`Duplicate summary filename already exists: ${summary} (${path.relative(projectRoot, paperListPath)}:${duplicateSummary.lineNumber})`,
		);
	}

	const newRecord = {
		title: paperInfo.title,
		author: paperInfo.author,
		year: paperInfo.year,
		url: canonicalUrl,
		summary,
		slide: "",
	};

	const nextContent = `${rawWithTrailingNewline}${JSON.stringify(newRecord)}\n`;
	await fs.writeFile(paperListPath, nextContent, "utf8");
	console.log(`[added] ${path.relative(projectRoot, paperListPath)}`);
	console.log(`[summary] ${summary}`);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
