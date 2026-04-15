import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

function normalizeTitleToFilename(title) {
	return (
		title
			.toLowerCase()
			.replaceAll(/[^a-z0-9]+/g, '_')
			.replaceAll(/_+/g, '_')
			.replaceAll(/^_|_$/g, '') + '.md'
	);
}

async function main() {
	const topicId = process.argv[2];

	if (!topicId) {
		console.error('Usage: node scripts/init-summary.js <topic_id>');
		process.exitCode = 1;
		return;
	}

	const paperListPath = path.join('static', 'docs', topicId, 'paper_list.jsonl');
	const fileContent = await fs.readFile(paperListPath, 'utf8');
	const lines = fileContent.split(/\r?\n/);
	const rl = readline.createInterface({ input, output });

	try {
		while (true) {
			const title = (await rl.question('title> ')).trim();

			if (title === '/q' || title === '/quit' || title === 'quit' || title === '/exit' || title === 'exit') {
				break;
			}

			let matchedLineIndex = -1;
			let matchedPaper = null;

			for (const [index, line] of lines.entries()) {
				const trimmedLine = line.trim();

				if (!trimmedLine || trimmedLine.startsWith('//')) {
					continue;
				}

				const paper = JSON.parse(line);
				if (paper.title === title) {
					matchedLineIndex = index;
					matchedPaper = paper;
					break;
				}
			}

			if (matchedLineIndex === -1 || !matchedPaper) {
				console.log('[ERROR] No matching title found.\n');
				continue;
			}

			if (matchedPaper.summary) {
				console.log(matchedPaper.summary);
				console.log('[ERROR] A file name already exists.\n');
				continue;
			}

			matchedPaper.summary = normalizeTitleToFilename(title);
			lines[matchedLineIndex] = JSON.stringify(matchedPaper);
			await fs.writeFile(paperListPath, lines.join('\n'), 'utf8');

			console.log(matchedPaper.summary);
			console.log('[SUCCESS] Saved the file name.\n');
		}
	} finally {
		rl.close();
	}
}

main().catch((error) => {
	console.error(error.message);
	process.exitCode = 1;
});
