/// <reference types="node" />

import fs from 'node:fs/promises';
import path from 'node:path';

function normalizeTitleToFilename(title) {
	return (
		title
			.toLowerCase()
			.replaceAll(/[^a-z0-9]+/g, '_')
			.replaceAll(/_+/g, '_')
			.replaceAll(/^_|_$/g, '') + '.md'
	);
}

function help() {
	console.log('Usage: node scripts/init-summary.js <topic_id>');
	console.log('');
	console.log('Fills missing summary filenames in static/docs/<topic_id>/paper_list.jsonl from each paper title.');
}

async function main() {
	if (process.argv.includes('--help') || process.argv.includes('-h')) {
		help();
		return;
	}

	const topicId = process.argv[2];

	if (!topicId) {
		console.error('Usage: node scripts/init-summary.js <topic_id>');
		process.exitCode = 1;
		return;
	}

	const paperListPath = path.join('static', 'docs', topicId, 'paper_list.jsonl');
	const fileContent = await fs.readFile(paperListPath, 'utf8');
	const lines = fileContent.split(/\r?\n/);

	let updatedCount = 0;

	for (const [index, line] of lines.entries()) {
		const trimmedLine = line.trim();

		if (!trimmedLine || trimmedLine.startsWith('//')) {
			continue;
		}

		const paper = JSON.parse(line);
		const title = typeof paper.title === 'string' ? paper.title.trim() : '';
		const summary = typeof paper.summary === 'string' ? paper.summary.trim() : '';

		if (!title || summary) {
			continue;
		}

		paper.summary = normalizeTitleToFilename(title);
		lines[index] = JSON.stringify(paper);
		updatedCount += 1;
	}

	await fs.writeFile(paperListPath, lines.join('\n'), 'utf8');
	console.log(`[DONE] Filled ${updatedCount} summary value(s).`);
}

main().catch((error) => {
	console.error(error.message);
	process.exitCode = 1;
});
