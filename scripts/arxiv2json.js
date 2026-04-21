function usage() {
	console.error('Usage: node scripts/arxiv2json.js <arxiv-abs-url>');
	process.exit(1);
}

function help() {
	console.log('Usage: node scripts/arxiv2json.js <arxiv-abs-url>');
	console.log('');
	console.log('Fetches metadata from an arXiv abs URL and prints a JSON record for paper_list.jsonl.');
}

function normalizeArxivUrl(url) {
	try {
		const parsed = new URL(url);

		if (!/arxiv\.org$/i.test(parsed.hostname)) {
			throw new Error('URL must point to arxiv.org');
		}

		if (!parsed.pathname.startsWith('/abs/')) {
			throw new Error('URL must be an arXiv abs page URL');
		}

		return `https://arxiv.org${parsed.pathname}`;
	} catch (error) {
		throw new Error(error instanceof Error ? error.message : 'Invalid arXiv URL provided');
	}
}

function decodeHtmlEntities(text) {
	return text
		.replace(/&amp;/g, '&')
		.replace(/&lt;/g, '<')
		.replace(/&gt;/g, '>')
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")
		.replace(/&#x27;/gi, "'")
		.replace(/&#x2F;/gi, '/');
}

function escapeRegex(text) {
	return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function readMetaContent(html, name) {
	const pattern = new RegExp(
		`<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
		'i'
	);
	const match = html.match(pattern);

	return match ? decodeHtmlEntities(match[3].trim()) : '';
}

function readMetaContents(html, name) {
	const pattern = new RegExp(
		`<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
		'gi'
	);
	const values = [];
	let match;

	while ((match = pattern.exec(html)) !== null) {
		values.push(decodeHtmlEntities(match[3].trim()));
	}

	return values;
}

function createSummaryFilename(title) {
	return (
		title
			.toLowerCase()
			.replaceAll(/[^a-z0-9]+/g, '_')
			.replaceAll(/_+/g, '_')
			.replaceAll(/^_|_$/g, '') + '.md'
	);
}

function normalizeAuthorName(name) {
	const parts = name
		.split(',')
		.map((part) => part.trim())
		.filter(Boolean);

	if (parts.length <= 1) {
		return name.trim();
	}

	return `${parts.slice(1).join(' ')} ${parts[0]}`.trim();
}

async function fetchPaperInfo(url) {
	const response = await fetch(url);

	if (!response.ok) {
		throw new Error(`Failed to fetch arXiv page: ${response.status} ${response.statusText}`);
	}

	const html = await response.text();
	const title = readMetaContent(html, 'citation_title');
	const authors = readMetaContents(html, 'citation_author');
	const publishedDate =
		readMetaContent(html, 'citation_date') ||
		readMetaContent(html, 'citation_online_date') ||
		readMetaContent(html, 'citation_publication_date');
	const yearMatch = publishedDate.match(/\b(\d{4})\b/);
	const year = yearMatch ? Number(yearMatch[1]) : NaN;

	if (!title) {
		throw new Error('Failed to parse title from the arXiv page.');
	}

	if (authors.length === 0) {
		throw new Error('Failed to parse authors from the arXiv page.');
	}

	if (!Number.isFinite(year)) {
		throw new Error('Failed to parse publication year from the arXiv page.');
	}

	return {
		title,
		author: authors.map(normalizeAuthorName).join(', '),
		year
	};
}

async function main() {
	if (process.argv.includes('--help') || process.argv.includes('-h')) {
		help();
		return;
	}

	const inputUrl = process.argv[2];

	if (!inputUrl) {
		usage();
	}

	const url = normalizeArxivUrl(inputUrl);
	const paperInfo = await fetchPaperInfo(url);
	const output = {
		title: paperInfo.title,
		author: paperInfo.author,
		year: paperInfo.year,
		url,
		summary: createSummaryFilename(paperInfo.title),
		slide: ''
	};

	console.log(JSON.stringify(output));
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
