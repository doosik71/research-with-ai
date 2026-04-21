import fs from 'node:fs/promises';
import path from 'node:path';

const DEFAULT_OUTPUT_DIR = '.\\temp';

function printUsage() {
	console.error('Usage: node import-arxivdata.js <old-data-path> [new-data-path]');
}

function printHelp() {
	console.log('Usage: node scripts/import-arxivdata.js <old-data-path> [new-data-path]');
	console.log('');
	console.log('Imports legacy arXiv JSON/Markdown pairs into a new paper_list.jsonl and summary markdown set.');
	console.log(`Default output directory: ${DEFAULT_OUTPUT_DIR}`);
}

function slugifyTitle(title) {
	const normalized = title
		.toLowerCase()
		.replace(/[\s]+/g, '_')
		.replace(/[^a-z0-9_]/g, '')
		.replace(/_+/g, '_')
		.replace(/^_+|_+$/g, '');

	return normalized || 'untitled_paper';
}

function buildUniqueFilename(baseSlug, usedNames) {
	let candidate = `${baseSlug}.md`;
	let suffix = 2;

	while (usedNames.has(candidate)) {
		candidate = `${baseSlug}_${suffix}.md`;
		suffix += 1;
	}

	usedNames.add(candidate);
	return candidate;
}

async function pathExists(targetPath) {
	try {
		await fs.access(targetPath);
		return true;
	} catch {
		return false;
	}
}

async function collectLegacyPapers(oldDataPath) {
	const entries = await fs.readdir(oldDataPath, { withFileTypes: true });
	const jsonFiles = entries
		.filter((entry) => entry.isFile() && path.extname(entry.name).toLowerCase() === '.json')
		.map((entry) => entry.name)
		.sort((a, b) => a.localeCompare(b));

	return jsonFiles;
}

async function main() {
	if (process.argv.includes('--help') || process.argv.includes('-h')) {
		printHelp();
		return;
	}

	const [, , oldDataArg, newDataArg] = process.argv;

	if (!oldDataArg) {
		printUsage();
		process.exitCode = 1;
		return;
	}

	const oldDataPath = path.resolve(oldDataArg);
	const newDataPath = path.resolve(newDataArg ?? DEFAULT_OUTPUT_DIR);

	if (!(await pathExists(oldDataPath))) {
		console.error(`Source path does not exist: ${oldDataPath}`);
		process.exitCode = 1;
		return;
	}

	await fs.mkdir(newDataPath, { recursive: true });

	const usedNames = new Set();
	const paperLines = [];
	const warnings = [];
	let processedCount = 0;
	let skippedCount = 0;

	const jsonFiles = await collectLegacyPapers(oldDataPath);

	for (const jsonFile of jsonFiles) {
		const baseName = path.basename(jsonFile, '.json');
		const jsonPath = path.join(oldDataPath, jsonFile);
		const mdPath = path.join(oldDataPath, `${baseName}.md`);

		if (!(await pathExists(mdPath))) {
			warnings.push(`Skipped ${baseName}: matching markdown file is missing.`);
			skippedCount += 1;
			continue;
		}

		try {
			const [jsonContent, markdownContent] = await Promise.all([
				fs.readFile(jsonPath, 'utf8'),
				fs.readFile(mdPath, 'utf8')
			]);

			const legacyPaper = JSON.parse(jsonContent);
			const { title, authors, year, url } = legacyPaper;

			if (!title || !authors || year === undefined || !url) {
				warnings.push(`Skipped ${baseName}: required fields are missing in JSON.`);
				skippedCount += 1;
				continue;
			}

			const baseSlug = slugifyTitle(title);
			const summaryFilename = buildUniqueFilename(baseSlug, usedNames);
			const summaryPath = path.join(newDataPath, summaryFilename);

			await fs.writeFile(summaryPath, markdownContent, 'utf8');

			paperLines.push(
				JSON.stringify({
					title,
					author: authors,
					year,
					url,
					summary: summaryFilename,
					slide: ''
				})
			);
			processedCount += 1;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			warnings.push(`Skipped ${baseName}: ${message}`);
			skippedCount += 1;
		}
	}

	const paperListPath = path.join(newDataPath, 'paper_list.jsonl');
	const paperListContent = paperLines.length > 0 ? `${paperLines.join('\n')}\n` : '';
	await fs.writeFile(paperListPath, paperListContent, 'utf8');

	console.log(`Source: ${oldDataPath}`);
	console.log(`Output: ${newDataPath}`);
	console.log(`Processed: ${processedCount}`);
	console.log(`Skipped: ${skippedCount}`);
	console.log(`paper_list.jsonl: ${paperListPath}`);

	if (warnings.length > 0) {
		console.warn('\nWarnings:');
		for (const warning of warnings) {
			console.warn(`- ${warning}`);
		}
	}
}

await main();
