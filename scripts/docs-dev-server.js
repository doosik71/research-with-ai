/// <reference types="node" />

import fs from 'node:fs/promises';
import http from 'node:http';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const docsRoot = path.join(projectRoot, 'static', 'docs');

const host = process.env.DOCS_DEV_HOST ?? '127.0.0.1';
const port = Number(process.env.DOCS_DEV_PORT ?? process.env.PORT ?? 8787);
let markdownItPromise = null;

function help() {
  console.log('Usage: node scripts/docs-dev-server.js');
  console.log('');
  console.log('Starts the local docs maintenance server.');
  console.log('Environment variables:');
  console.log('  DOCS_DEV_HOST  Server host (default: 127.0.0.1)');
  console.log('  DOCS_DEV_PORT  Server port (default: 8787)');
  console.log('  PORT           Fallback port when DOCS_DEV_PORT is not set');
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function toPosixPath(value) {
  return String(value).replaceAll('\\', '/');
}

function normalizeRelativePath(inputPath) {
  const value = toPosixPath(String(inputPath ?? '')).trim();
  return value.replace(/^\/+/, '').replace(/\/+/g, '/');
}

function resolveDocsPath(relativePath) {
  const normalized = normalizeRelativePath(relativePath);
  const absolutePath = path.resolve(docsRoot, normalized);
  const rootWithSep = `${docsRoot}${path.sep}`;

  if (absolutePath !== docsRoot && !absolutePath.startsWith(rootWithSep)) {
    throw new Error('Path escapes static/docs.');
  }

  return absolutePath;
}

function isMarkdownFile(fileName) {
  return fileName.toLowerCase().endsWith('.md');
}

function sortByName(a, b) {
  return a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' });
}

async function pathExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function readRequestBody(request) {
  const chunks = [];

  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  return Buffer.concat(chunks);
}

async function readJson(request) {
  const body = await readRequestBody(request);
  if (!body.length) return {};
  return JSON.parse(body.toString('utf8'));
}

function runNodeScript(scriptRelativePath, scriptArgs = []) {
  const scriptPath = path.join(projectRoot, scriptRelativePath);

  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [scriptPath, ...scriptArgs], {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8');
    });

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });

    child.on('error', (error) => {
      reject(new Error(`Failed to start ${path.basename(scriptRelativePath)}: ${error.message}`));
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve({
          ok: true,
          stdout: stdout.trim(),
          stderr: stderr.trim()
        });
        return;
      }

      reject(
        new Error(
          stderr.trim() ||
          stdout.trim() ||
          `${path.basename(scriptRelativePath)} exited with code ${code}`
        )
      );
    });
  });
}

function streamNodeScript(scriptRelativePath, scriptArgs = [], requestSignal) {
  const scriptPath = path.join(projectRoot, scriptRelativePath);
  const encoder = new TextEncoder();

  return new Response(
    new ReadableStream({
      start(controller) {
        const child = spawn(process.execPath, [scriptPath, ...scriptArgs], {
          cwd: projectRoot,
          stdio: ['ignore', 'pipe', 'pipe']
        });

        const push = (chunk) => {
          controller.enqueue(encoder.encode(chunk));
        };

        child.stdout.on('data', (chunk) => {
          push(chunk.toString('utf8'));
        });

        child.stderr.on('data', (chunk) => {
          push(chunk.toString('utf8'));
        });

        child.on('error', (error) => {
          controller.error(error);
        });

        child.on('close', (code) => {
          if (code === 0) {
            controller.close();
            return;
          }

          controller.error(new Error(`${path.basename(scriptRelativePath)} exited with code ${code}`));
        });

        if (requestSignal) {
          if (requestSignal.aborted) {
            child.kill('SIGTERM');
            controller.error(new Error('Aborted'));
            return;
          }

          requestSignal.addEventListener('abort', () => {
            child.kill('SIGTERM');
            controller.error(new Error('Aborted'));
          }, { once: true });
        }
      }
    }),
    {
      status: 200,
      headers: {
        'content-type': 'text/plain; charset=utf-8',
        'cache-control': 'no-store'
      }
    }
  );
}

function jsonResponse(statusCode, payload) {
  return new Response(JSON.stringify(payload, null, 2), {
    status: statusCode,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store'
    }
  });
}

function textResponse(statusCode, body, contentType = 'text/plain; charset=utf-8') {
  return new Response(body, {
    status: statusCode,
    headers: {
      'content-type': contentType,
      'cache-control': 'no-store'
    }
  });
}

async function getMarkdownIt() {
  if (!markdownItPromise) {
    markdownItPromise = import('markdown-it')
      .then((module) => new module.default({ html: false, linkify: true, typographer: true }))
      .catch(() => null);
  }

  return markdownItPromise;
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_\n]+)_/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return html;
}

function renderBasicMarkdown(markdown) {
  const lines = String(markdown ?? '').replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i] ?? '';
    if (!line.trim()) {
      i += 1;
      continue;
    }

    const fenceMatch = line.match(/^```(\w+)?\s*$/);
    if (fenceMatch) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i] ?? '')) {
        codeLines.push(lines[i] ?? '');
        i += 1;
      }
      if (i < lines.length) i += 1;
      blocks.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2] ?? '')}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) {
      blocks.push('<hr>');
      i += 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoteLines = [];
      while (i < lines.length && /^>\s?/.test(lines[i] ?? '')) {
        quoteLines.push((lines[i] ?? '').replace(/^>\s?/, ''));
        i += 1;
      }
      blocks.push(`<blockquote>${renderBasicMarkdown(quoteLines.join('\n')).replace(/^<p>|<\/p>$/g, '')}</blockquote>`);
      continue;
    }

    if (/^\s*(?:[-*+]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\.\s+/.test(line);
      const items = [];
      while (i < lines.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i] ?? '')) {
        items.push((lines[i] ?? '').replace(/^\s*(?:[-*+]|\d+\.)\s+/, ''));
        i += 1;
      }
      const tag = ordered ? 'ol' : 'ul';
      blocks.push(`<${tag}>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</${tag}>`);
      continue;
    }

    const paragraphLines = [];
    while (
      i < lines.length &&
      lines[i] &&
      lines[i].trim() &&
      !/^(#{1,6})\s+/.test(lines[i]) &&
      !/^```/.test(lines[i]) &&
      !/^>\s?/.test(lines[i]) &&
      !/^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i].trim())
    ) {
      paragraphLines.push(lines[i]);
      i += 1;
    }

    blocks.push(`<p>${renderInlineMarkdown(paragraphLines.join(' '))}</p>`);
  }

  return blocks.join('\n');
}

async function renderMarkdown(markdown) {
  const markdownIt = await getMarkdownIt();
  if (markdownIt) {
    return markdownIt.render(String(markdown ?? ''));
  }

  return renderBasicMarkdown(markdown);
}

async function listMarkdownFiles(baseDir) {
  if (!(await pathExists(baseDir))) return [];

  const entries = await fs.readdir(baseDir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    if (!entry.isFile() || !isMarkdownFile(entry.name)) continue;

    files.push({
      name: entry.name,
      relativePath: toPosixPath(path.relative(docsRoot, path.join(baseDir, entry.name))),
      size: (await fs.stat(path.join(baseDir, entry.name))).size
    });
  }

  files.sort((a, b) => sortByName(a.name, b.name));
  return files;
}

async function readPaperList(topicDir) {
  const paperListPath = path.join(topicDir, 'paper_list.jsonl');
  if (!(await pathExists(paperListPath))) return [];

  const raw = await fs.readFile(paperListPath, 'utf8');
  const papers = [];

  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('//')) continue;

    try {
      const parsed = JSON.parse(line);
      if (parsed && typeof parsed === 'object') {
        papers.push({
          title: typeof parsed.title === 'string' ? parsed.title : '',
          author: typeof parsed.author === 'string' ? parsed.author : '',
          year: parsed.year ?? '',
          url: typeof parsed.url === 'string' ? parsed.url : '',
          summary: typeof parsed.summary === 'string' ? parsed.summary : '',
          slide: typeof parsed.slide === 'string' ? parsed.slide : ''
        });
      }
    } catch {
      // Ignore malformed rows in the dev server.
    }
  }

  return papers;
}

async function listTopics() {
  const entries = await fs.readdir(docsRoot, { withFileTypes: true });
  const topics = [];

  // const rootFiles = await listMarkdownFiles(docsRoot);
  // if (rootFiles.length > 0) {
  // 	topics.push({
  // 		id: '',
  // 		label: 'ROOT',
  // 		files: rootFiles,
  // 		fileCount: rootFiles.length
  // 	});
  // }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;

    const topicDir = path.join(docsRoot, entry.name);
    const papers = await readPaperList(topicDir);
    const metadata = await readTopicMetadata(topicDir);

    topics.push({
      id: entry.name,
      label: metadata?.title ?? entry.name,
      papers,
      paperCount: papers.length,
      metadata
    });
  }

  topics.sort((a, b) => sortByName(a.label, b.label));
  return topics;
}

async function readTopicMetadata(topicDir) {
  const candidates = ['metadata.json', '--metadata.json'];

  for (const fileName of candidates) {
    const metadataPath = path.join(topicDir, fileName);
    if (!(await pathExists(metadataPath))) continue;

    try {
      const raw = await fs.readFile(metadataPath, 'utf8');
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  }

  return null;
}

function toTopicFolderName(title) {
  return String(title)
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '_')
    .replaceAll(/_+/g, '_')
    .replaceAll(/^_|_$/g, '');
}

function parseKeywordInput(keyword) {
  if (Array.isArray(keyword)) {
    return keyword.map((item) => String(item).trim()).filter(Boolean);
  }

  return String(keyword ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function isArxivId(value) {
  const text = String(value ?? '').trim();
  const base = text.includes('v') ? text.slice(0, text.lastIndexOf('v')) : text;
  const version = text.includes('v') ? text.slice(text.lastIndexOf('v') + 1) : '';

  if (version && !/^\d+$/.test(version)) {
    return false;
  }

  const dotIndex = base.indexOf('.');
  if (dotIndex !== 4) {
    return false;
  }

  const year = base.slice(0, 4);
  const number = base.slice(5);
  return /^\d{4}$/.test(year) && /^\d{4,5}$/.test(number);
}

function extractArxivIdFromText(value) {
  const candidate = String(value ?? '')
    .trim()
    .split(/[?#\s]/)[0]
    .replace(/\/+$/, '');

  if (!isArxivId(candidate)) {
    throw new Error('Invalid arXiv URL provided');
  }

  return candidate;
}

function normalizeArxivUrl(url) {
  const trimmed = String(url ?? '').trim();
  if (!trimmed) {
    throw new Error('URL is required.');
  }

  if (isArxivId(trimmed) && !trimmed.toLowerCase().includes('arxiv.org')) {
    return `https://arxiv.org/abs/${trimmed}`;
  }

  const absIndex = trimmed.toLowerCase().indexOf('arxiv.org/abs/');
  if (absIndex !== -1) {
    return `https://arxiv.org/abs/${extractArxivIdFromText(trimmed.slice(absIndex + 'arxiv.org/abs/'.length))}`;
  }

  const withScheme = /:\/\//.test(trimmed) ? trimmed : `https://${trimmed}`;
  let parsed;
  try {
    parsed = new URL(withScheme);
  } catch {
    if (isArxivId(trimmed)) {
      return `https://arxiv.org/abs/${trimmed}`;
    }
    throw new Error('Invalid arXiv URL provided');
  }

  if (!/arxiv\.org$/i.test(parsed.hostname)) {
    throw new Error('URL must point to arxiv.org');
  }

  if (!parsed.pathname.startsWith('/abs/')) {
    throw new Error('URL must be an arXiv abs page URL');
  }

  return `https://arxiv.org${parsed.pathname}`;
}

function decodeHtmlEntities(text) {
  return String(text)
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/gi, "'")
    .replace(/&#x2F;/gi, '/');
}

function escapeRegex(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function readMetaContent(html, name) {
  const pattern = new RegExp(
    `<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
    'i'
  );
  const match = String(html).match(pattern);
  return match ? decodeHtmlEntities(match[3].trim()) : '';
}

function readMetaContents(html, name) {
  const pattern = new RegExp(
    `<meta\\s+name=(["'])${escapeRegex(name)}\\1\\s+content=(["'])(.*?)\\2`,
    'gi'
  );
  const values = [];
  let match;

  while ((match = pattern.exec(String(html))) !== null) {
    values.push(decodeHtmlEntities(match[3].trim()));
  }

  return values;
}

function normalizeAuthorName(name) {
  const parts = String(name)
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

  if (parts.length <= 1) {
    return String(name).trim();
  }

  return `${parts.slice(1).join(' ')} ${parts[0]}`.trim();
}

function createSummaryFilename(title) {
  return `${String(title)
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '_')
    .replaceAll(/_+/g, '_')
    .replaceAll(/^_|_$/g, '')}.md`;
}

async function fetchArxivPaperInfo(url) {
  const canonicalUrl = normalizeArxivUrl(url);
  const response = await fetch(canonicalUrl);
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
    year,
    url: canonicalUrl
  };
}

function resolvePaperSummaryPath(topicId, summary) {
  const normalizedSummary = String(summary ?? '').trim();
  if (!normalizedSummary) return '';
  if (normalizedSummary.includes('/')) return normalizedSummary;
  if (!topicId) return normalizedSummary;
  return `${topicId}/${normalizedSummary}`;
}

async function createTopic({ title, keyword }) {
  const topicTitle = String(title ?? '').trim();
  if (!topicTitle) {
    throw new Error('Title is required.');
  }

  const folderName = toTopicFolderName(topicTitle);
  if (!folderName) {
    throw new Error('Title must contain at least one ASCII letter or number.');
  }

  const topicDir = path.join(docsRoot, folderName);
  if (await pathExists(topicDir)) {
    throw new Error('Topic folder already exists.');
  }

  await fs.mkdir(topicDir, { recursive: true });
  const metadata = {
    title: topicTitle,
    keyword: parseKeywordInput(keyword)
  };

  await fs.writeFile(path.join(topicDir, 'metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
  await fs.writeFile(path.join(topicDir, 'paper_list.jsonl'), '', 'utf8');

  return { id: folderName, metadata };
}

async function appendPaperToTopic(topicId, paper) {
  const topicDir = path.join(docsRoot, topicId);
  if (!(await pathExists(topicDir))) {
    throw new Error(`Topic folder not found: ${topicId}`);
  }

  const paperListPath = path.join(topicDir, 'paper_list.jsonl');
  if (!(await pathExists(paperListPath))) {
    throw new Error(`paper_list.jsonl not found: static/docs/${topicId}/paper_list.jsonl`);
  }

  const title = String(paper?.title ?? '').trim();
  const author = String(paper?.author ?? '').trim();
  const url = normalizeArxivUrl(paper?.url ?? '');
  const year = Number(paper?.year);

  if (!title) {
    throw new Error('Title is required.');
  }

  if (!author) {
    throw new Error('Author is required.');
  }

  if (!Number.isFinite(year)) {
    throw new Error('Year is required.');
  }

  const summary = createSummaryFilename(title);

  const raw = await fs.readFile(paperListPath, 'utf8');
  const nextRecord = {
    title,
    author,
    year,
    url,
    summary,
    slide: ''
  };

  const existingLines = raw.replace(/\r\n/g, '\n').split('\n').filter(Boolean);
  for (const rawLine of existingLines) {
    const trimmed = rawLine.trim();
    if (!trimmed || trimmed.startsWith('//')) continue;
    try {
      const parsed = JSON.parse(trimmed);
      const existingUrl = typeof parsed?.url === 'string' ? parsed.url.trim() : '';
      const existingSummary = typeof parsed?.summary === 'string' ? parsed.summary.trim() : '';
      if (existingUrl && normalizeArxivUrl(existingUrl) === url) {
        throw new Error('This arXiv paper already exists in the paper list.');
      }
      if (existingSummary && existingSummary === summary) {
        throw new Error('A paper with the same summary filename already exists.');
      }
    } catch (error) {
      if (error instanceof Error && error.message === 'This arXiv paper already exists in the paper list.') {
        throw error;
      }
      if (error instanceof Error && error.message === 'A paper with the same summary filename already exists.') {
        throw error;
      }
    }
  }

  const nextContent = `${raw}${raw.endsWith('\n') || raw.length === 0 ? '' : '\n'}${JSON.stringify(nextRecord)}\n`;
  await fs.writeFile(paperListPath, nextContent, 'utf8');

  return {
    topicId,
    paper: nextRecord
  };
}

async function runMaintenanceTask(task, payload = {}) {
  if (task === 'update-summary') {
    const summaryPath = String(payload?.path ?? '').trim();
    if (!summaryPath) {
      throw new Error('Summary path is required.');
    }
    return runNodeScript('scripts/update-summary.js', [summaryPath]);
  }

  if (task === 'update-index') {
    const topicId = String(payload?.topicId ?? '').trim();
    if (!topicId) {
      throw new Error('Topic id is required.');
    }
    return runNodeScript('scripts/update-index.js', [topicId]);
  }

  if (task === 'generate-manifest') {
    return runNodeScript('scripts/generate-manifest.js');
  }

  throw new Error(`Unknown maintenance task: ${task}`);
}

async function readMarkdownFile(relativePath) {
  const filePath = resolveDocsPath(relativePath);

  if (!filePath.endsWith('.md')) {
    throw new Error('Only markdown files can be managed by this server.');
  }

  const content = await fs.readFile(filePath, 'utf8');
  return { filePath, content };
}

async function writeMarkdownFile(relativePath, content) {
  const filePath = resolveDocsPath(relativePath);

  if (!filePath.endsWith('.md')) {
    throw new Error('Only markdown files can be managed by this server.');
  }

  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, 'utf8');
  return filePath;
}

async function deleteMarkdownFile(relativePath) {
  const filePath = resolveDocsPath(relativePath);

  if (!filePath.endsWith('.md')) {
    throw new Error('Only markdown files can be managed by this server.');
  }

  await fs.unlink(filePath);
  return filePath;
}

function renderHtmlPage() {
  return `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Docs Dev Server</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f1ea;
      --panel: #ffffff;
      --panel-2: #fbf7f2;
      --text: #1e1b18;
      --muted: #6d655f;
      --border: #d9cfc4;
      --accent: #7a4f2f;
      --accent-2: #3d6b7a;
      --danger: #9f3f2f;
      --shadow: 0 12px 32px rgba(41, 26, 13, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122, 79, 47, 0.12), transparent 30%),
        linear-gradient(180deg, #faf6f1 0%, #f0e7dc 100%);
      color: var(--text);
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 22px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.7);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0.02em;
    }
    header p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .menu-wrap {
      position: relative;
    }
    .action-menu {
      position: absolute;
      top: calc(100% + 8px);
      right: 0;
      min-width: 200px;
      padding: 8px;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 6px;
      z-index: 20;
    }
    .action-menu[hidden] {
      display: none;
    }
    .action-menu button {
      width: 100%;
      justify-content: flex-start;
      text-align: left;
      border-radius: 8px;
      padding: 9px 10px;
    }
    .action-menu button:disabled {
      opacity: 0.45;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 10px;
      padding: 9px 12px;
      cursor: pointer;
      box-shadow: 0 1px 0 rgba(0, 0, 0, 0.02);
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    button.danger {
      background: #fff5f4;
      color: var(--danger);
      border-color: #efc5be;
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    main {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(280px, 1.1fr) minmax(360px, 1.4fr) minmax(360px, 1.4fr);
      gap: 16px;
      padding: 16px;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }
    .status-bar {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      padding: 10px 22px 14px;
      border-top: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(12px);
    }
    .status-bar .status {
      min-height: 1.2em;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel-2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .panel-head h2 {
      margin: 0;
      font-size: 14px;
    }
    .panel-body {
      flex: 1;
      min-height: 0;
      overflow: auto;
    }
    .editor-panel .panel-body {
      padding: 0;
      overflow: hidden;
      display: flex;
    }
    .editor-panel textarea {
      flex: 1;
    }
    .preview-panel .panel-body {
      padding: 0;
    }
    .topics, .papers {
      list-style: none;
      margin: 0;
      padding: 0;
    }
    .topics li, .papers li {
      border-bottom: 1px solid rgba(0, 0, 0, 0.04);
    }
    .row-btn {
      display: flex;
      width: 100%;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      border: 0;
      background: #fff;
      color: var(--text);
      border-radius: 0;
      text-align: left;
    }
    .row-btn:hover {
      background: rgba(122, 79, 47, 0.06);
    }
    .row-btn.active {
      background: rgba(122, 79, 47, 0.12);
    }
    .paper-main {
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .paper-title {
      font-size: 14px;
      line-height: 1.35;
      white-space: normal;
      word-break: break-word;
    }
    .paper-meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.3;
      white-space: normal;
      word-break: break-word;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    textarea {
      width: 100%;
      min-height: 0;
      height: 100%;
      resize: none;
      border: 0;
      outline: none;
      padding: 16px;
      background: #fffdfb;
      color: var(--text);
      font-size: 14px;
      line-height: 1.7;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .preview {
      padding: 16px 18px;
      overflow: auto;
      background: linear-gradient(180deg, #fff 0%, #fdfaf6 100%);
    }
    .preview :is(h1, h2, h3, h4) { line-height: 1.2; }
    .preview img { max-width: 100%; }
    .preview pre {
      overflow: auto;
      padding: 14px;
      background: #f7f4ef;
      border-radius: 12px;
    }
    .status {
      font-size: 12px;
      color: var(--muted);
    }
    .status.error {
      color: var(--danger);
    }
    .empty {
      padding: 18px;
      color: var(--muted);
    }
    .popover-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(22, 17, 14, 0.42);
      display: grid;
      place-items: center;
      padding: 20px;
      z-index: 50;
    }
    .popover-backdrop[hidden] {
      display: none !important;
    }
    .popover {
      width: min(520px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18);
      overflow: hidden;
    }
    .popover-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 18px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--border);
    }
    .popover-head h3 {
      margin: 0;
      font-size: 16px;
    }
    .popover-body {
      display: grid;
      gap: 14px;
      padding: 18px;
    }
    .popover-body label {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .input-row {
      display: flex;
      gap: 8px;
      align-items: stretch;
    }
    .input-row input {
      flex: 1;
      min-width: 0;
    }
    .popover-body input {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 11px 12px;
      background: #fffdfb;
      color: var(--text);
    }
    .popover-body input[readonly] {
      background: #f5f1ea;
      color: var(--muted);
    }
    .popover-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding-top: 4px;
      align-items: center;
      flex-wrap: wrap;
    }
    .popover-actions .status {
      margin-right: auto;
      max-width: 280px;
    }
    .log-output {
      margin: 0;
      padding: 14px 16px;
      min-height: 220px;
      max-height: min(60vh, 520px);
      overflow: auto;
      background: #15110e;
      color: #f7ede3;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .log-empty {
      color: #cdbfb0;
    }
    @media (max-width: 1100px) {
      main {
        grid-template-columns: 1fr;
        height: auto;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Docs Dev Server</h1>
    </div>
    <div class="toolbar">
      <button id="refreshBtn">Refresh</button>
      <button id="addTopicBtn">Add Topic</button>
      <button id="addPaperBtn">Add Paper</button>
      <button class="primary" id="saveBtn" disabled>Save</button>
      <button class="danger" id="deleteBtn" disabled>Delete</button>
      <div class="menu-wrap">
        <button id="menuBtn" type="button" aria-haspopup="menu" aria-expanded="false" aria-label="Maintenance menu">☰</button>
        <div class="action-menu" id="actionMenu" hidden role="menu" aria-label="Maintenance actions">
          <button type="button" id="updateSummaryBtn" role="menuitem">Update Summary</button>
          <button type="button" id="updateIndexBtn" role="menuitem">Update Index</button>
          <button type="button" id="generateManifestBtn" role="menuitem">Generate Manifest</button>
        </div>
      </div>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-head">
        <h2>Topics</h2>
        <span class="meta" id="topicCount">0</span>
      </div>
      <div class="panel-body">
        <ul class="topics" id="topicList"></ul>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Papers</h2>
        <span class="meta" id="paperCount">0</span>
      </div>
      <div class="panel-body">
        <ul class="papers" id="paperList"></ul>
      </div>
    </section>

    <section class="panel editor-panel">
      <div class="panel-head">
        <h2>Editor</h2>
        <span class="meta" id="dirtyState">Clean</span>
      </div>
      <div class="panel-body">
        <textarea id="editor" spellcheck="false" placeholder="Select a paper to edit its summary markdown."></textarea>
      </div>
    </section>

    <section class="panel preview-panel">
      <div class="panel-head">
        <h2>Preview</h2>
        <span class="meta" id="currentPath">No paper selected</span>
      </div>
      <div class="panel-body">
        <div class="preview" id="preview">
          <div class="empty">Preview will appear here.</div>
        </div>
      </div>
    </section>
  </main>

  <footer class="status-bar">
    <span class="status" id="status">Ready</span>
  </footer>

  <div class="popover-backdrop" id="topicPopoverBackdrop" hidden>
    <div class="popover" role="dialog" aria-modal="true" aria-labelledby="topicPopoverTitle">
      <div class="popover-head">
        <h3 id="topicPopoverTitle">Add Topic</h3>
        <button id="closeTopicPopoverBtn" type="button" aria-label="Close">Close</button>
      </div>
      <form id="topicForm" class="popover-body">
        <label>
          <span>Title</span>
          <input id="topicTitleInput" name="title" type="text" placeholder="Activation Function" />
        </label>
        <label>
          <span>Keyword</span>
          <input id="topicKeywordInput" name="keyword" type="text" placeholder="activation function" />
        </label>
        <div class="popover-actions">
          <button type="button" id="cancelTopicBtn">Cancel</button>
          <button type="submit" class="primary">Create Topic</button>
        </div>
      </form>
    </div>
  </div>

  <div class="popover-backdrop" id="paperPopoverBackdrop" hidden>
    <div class="popover" role="dialog" aria-modal="true" aria-labelledby="paperPopoverTitle">
      <div class="popover-head">
        <h3 id="paperPopoverTitle">Add Paper</h3>
        <button id="closePaperPopoverBtn" type="button" aria-label="Close">Close</button>
      </div>
      <form id="paperForm" class="popover-body">
        <label>
          <span>URL</span>
          <div class="input-row">
            <input id="paperUrlInput" name="url" type="text" placeholder="1702.07790 or https://arxiv.org/abs/1702.07790" />
            <button type="button" id="fetchPaperBtn">Fetch</button>
          </div>
        </label>
        <label>
          <span>Title</span>
          <input id="paperTitleInput" name="title" type="text" placeholder="Activation Ensembles for Deep Neural Networks" />
        </label>
        <label>
          <span>Author</span>
          <input id="paperAuthorInput" name="author" type="text" placeholder="Mark Harmon, Diego Klabjan" />
        </label>
        <label>
          <span>Year</span>
          <input id="paperYearInput" name="year" type="text" inputmode="numeric" placeholder="2017" />
        </label>
        <label>
          <span>Summary</span>
          <input id="paperSummaryInput" name="summary" type="text" readonly />
        </label>
        <div class="popover-actions">
          <span class="status" id="paperPopoverStatus">Use Fetch to fill arXiv metadata, or enter it manually.</span>
          <button type="button" id="cancelPaperBtn">Cancel</button>
          <button type="submit" class="primary">Create Paper</button>
        </div>
      </form>
    </div>
  </div>

  <div class="popover-backdrop" id="summaryRunBackdrop" hidden>
    <div class="popover" role="dialog" aria-modal="true" aria-labelledby="summaryRunTitle">
      <div class="popover-head">
        <h3 id="summaryRunTitle">Update Summary Log</h3>
        <button id="closeSummaryRunBtn" type="button" aria-label="Close">Close</button>
      </div>
      <div class="popover-body">
        <div class="status" id="summaryRunStatus">Waiting to start...</div>
        <pre class="log-output" id="summaryRunLog"><span class="log-empty">No output yet.</span></pre>
        <div class="popover-actions">
          <button type="button" id="clearSummaryRunBtn">Clear</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = {
      topics: [],
      topicId: '',
      papers: [],
      paperPath: '',
      initialContent: '',
      dirty: false,
      previewTimer: null
    };

    const topicList = document.getElementById('topicList');
    const paperList = document.getElementById('paperList');
    const editor = document.getElementById('editor');
    const preview = document.getElementById('preview');
    const status = document.getElementById('status');
    const dirtyState = document.getElementById('dirtyState');
    const currentPath = document.getElementById('currentPath');
    const topicCount = document.getElementById('topicCount');
    const paperCount = document.getElementById('paperCount');
    const saveBtn = document.getElementById('saveBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const menuBtn = document.getElementById('menuBtn');
    const actionMenu = document.getElementById('actionMenu');
    const updateSummaryBtn = document.getElementById('updateSummaryBtn');
    const updateIndexBtn = document.getElementById('updateIndexBtn');
    const generateManifestBtn = document.getElementById('generateManifestBtn');
    const addTopicBtn = document.getElementById('addTopicBtn');
    const addPaperBtn = document.getElementById('addPaperBtn');
    const topicPopoverBackdrop = document.getElementById('topicPopoverBackdrop');
    const closeTopicPopoverBtn = document.getElementById('closeTopicPopoverBtn');
    const cancelTopicBtn = document.getElementById('cancelTopicBtn');
    const topicForm = document.getElementById('topicForm');
    const topicTitleInput = document.getElementById('topicTitleInput');
    const topicKeywordInput = document.getElementById('topicKeywordInput');
    const paperPopoverBackdrop = document.getElementById('paperPopoverBackdrop');
    const closePaperPopoverBtn = document.getElementById('closePaperPopoverBtn');
    const cancelPaperBtn = document.getElementById('cancelPaperBtn');
    const paperForm = document.getElementById('paperForm');
    const paperUrlInput = document.getElementById('paperUrlInput');
    const paperTitleInput = document.getElementById('paperTitleInput');
    const paperAuthorInput = document.getElementById('paperAuthorInput');
    const paperYearInput = document.getElementById('paperYearInput');
    const paperSummaryInput = document.getElementById('paperSummaryInput');
    const fetchPaperBtn = document.getElementById('fetchPaperBtn');
    const paperPopoverStatus = document.getElementById('paperPopoverStatus');
    const defaultEditorPlaceholder = editor.getAttribute('placeholder') || '';
    const summaryRunBackdrop = document.getElementById('summaryRunBackdrop');
    const closeSummaryRunBtn = document.getElementById('closeSummaryRunBtn');
    const clearSummaryRunBtn = document.getElementById('clearSummaryRunBtn');
    const summaryRunLog = document.getElementById('summaryRunLog');
    const summaryRunStatus = document.getElementById('summaryRunStatus');
    let summaryRunAbortController = null;

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function resolvePaperSummaryPath(topicId, summary) {
      const normalizedSummary = String(summary == null ? '' : summary).trim();
      if (!normalizedSummary) return '';
      if (normalizedSummary.includes('/')) return normalizedSummary;
      if (!topicId) return normalizedSummary;
      return topicId + '/' + normalizedSummary;
    }

    function createSummaryFilename(title) {
      return (
        String(title)
          .toLowerCase()
          .replaceAll(/[^a-z0-9]+/g, '_')
          .replaceAll(/_+/g, '_')
          .replaceAll(/^_|_$/g, '') + '.md'
      );
    }

    function isArxivId(value) {
      const text = String(value == null ? '' : value).trim();
      if (!text) {
        return false;
      }

      let base = text;
      const vIndex = text.lastIndexOf('v');
      if (vIndex > 0) {
        const version = text.slice(vIndex + 1);
        let isVersion = version.length > 0;
        for (let i = 0; i < version.length; i += 1) {
          const code = version.charCodeAt(i);
          if (code < 48 || code > 57) {
            isVersion = false;
            break;
          }
        }
        if (isVersion) {
          base = text.slice(0, vIndex);
        }
      }

      if (base.length < 10 || base[4] !== '.') {
        return false;
      }

      const year = base.slice(0, 4);
      const number = base.slice(5);
      if (number.length < 4 || number.length > 5) {
        return false;
      }

      for (let i = 0; i < year.length; i += 1) {
        const code = year.charCodeAt(i);
        if (code < 48 || code > 57) {
          return false;
        }
      }

      for (let i = 0; i < number.length; i += 1) {
        const code = number.charCodeAt(i);
        if (code < 48 || code > 57) {
          return false;
        }
      }

      return true;
    }

    function extractArxivIdFromText(value) {
      let candidate = String(value == null ? '' : value).trim();
      const separators = ['?', '#'];
      for (const separator of separators) {
        const index = candidate.indexOf(separator);
        if (index !== -1) {
          candidate = candidate.slice(0, index);
        }
      }

      for (let i = 0; i < candidate.length; i += 1) {
        if (candidate.charCodeAt(i) <= 32) {
          candidate = candidate.slice(0, i);
          break;
        }
      }

      while (candidate.endsWith('/')) {
        candidate = candidate.slice(0, -1);
      }

      if (!isArxivId(candidate)) {
        throw new Error('Invalid arXiv URL provided');
      }

      return candidate;
    }

    function normalizePaperUrlInput(rawInput) {
      const trimmed = String(rawInput == null ? '' : rawInput).trim();
      if (!trimmed) {
        throw new Error('URL is required.');
      }

      if (isArxivId(trimmed) && !trimmed.toLowerCase().includes('arxiv.org')) {
        return 'https://arxiv.org/abs/' + extractArxivIdFromText(trimmed);
      }

      const absMarker = 'arxiv.org/abs/';
      const absIndex = trimmed.toLowerCase().indexOf(absMarker);
      if (absIndex !== -1) {
        return 'https://arxiv.org/abs/' + extractArxivIdFromText(trimmed.slice(absIndex + absMarker.length));
      }

      const withScheme = trimmed.includes('://') ? trimmed : 'https://' + trimmed;
      let parsed;
      try {
        parsed = new URL(withScheme);
      } catch {
        if (isArxivId(trimmed)) {
          return 'https://arxiv.org/abs/' + extractArxivIdFromText(trimmed);
        }
        throw new Error('Invalid arXiv URL provided');
      }

      if (!parsed.hostname.toLowerCase().endsWith('arxiv.org')) {
        throw new Error('URL must point to arxiv.org');
      }
      if (!parsed.pathname.startsWith('/abs/')) {
        throw new Error('URL must be an arXiv abs page URL');
      }

      return 'https://arxiv.org/abs/' + extractArxivIdFromText(parsed.pathname.slice('/abs/'.length));
    }

    function updatePaperSummary() {
      paperSummaryInput.value = createSummaryFilename(paperTitleInput.value.trim());
    }

    function showMissingPaper(summaryPath) {
      state.paperPath = summaryPath;
      state.initialContent = '';
      editor.value = '';
      editor.placeholder = 'Markdown file not found: ' + summaryPath;
      updatePreview('<div class="empty">Markdown file not found: ' + escapeHtml(summaryPath) + '</div>');
      setDirty(false);
      updateCurrentSelection();
      updateMaintenanceMenuState();
      renderFiles();
      setStatus('Markdown file not found: ' + summaryPath);
    }

    function setStatus(message, kind = '') {
      status.textContent = message;
      status.className = kind ? 'status ' + kind : 'status';
    }

    function setDirty(dirty) {
      state.dirty = dirty;
      dirtyState.textContent = dirty ? 'Unsaved changes' : 'Clean';
      saveBtn.disabled = !state.paperPath;
      deleteBtn.disabled = !state.paperPath;
    }

    function openTopicPopover() {
      closeActionMenu();
      topicPopoverBackdrop.hidden = false;
      topicTitleInput.value = '';
      topicKeywordInput.value = '';
      topicTitleInput.focus();
    }

    function closeTopicPopover() {
      topicPopoverBackdrop.hidden = true;
    }

    function openPaperPopover() {
      closeActionMenu();
      paperPopoverBackdrop.hidden = false;
      paperUrlInput.value = '';
      paperTitleInput.value = '';
      paperAuthorInput.value = '';
      paperYearInput.value = '';
      paperSummaryInput.value = '';
      paperPopoverStatus.textContent = 'Use Fetch to fill arXiv metadata, or enter it manually.';
      paperUrlInput.focus();
    }

    function closePaperPopover() {
      paperPopoverBackdrop.hidden = true;
    }

    function clearSummaryRunLog() {
      summaryRunLog.textContent = '';
    }

    function appendSummaryRunLog(text) {
      if (!summaryRunLog.textContent || summaryRunLog.textContent === 'No output yet.') {
        summaryRunLog.textContent = '';
      }
      summaryRunLog.textContent += text;
      summaryRunLog.scrollTop = summaryRunLog.scrollHeight;
    }

    function openSummaryRunPopover(initialMessage) {
      summaryRunBackdrop.hidden = false;
      summaryRunStatus.textContent = initialMessage || 'Running update-summary...';
      clearSummaryRunLog();
      summaryRunLog.textContent = 'No output yet.';
    }

    function closeSummaryRunPopover() {
      summaryRunBackdrop.hidden = true;
      if (summaryRunAbortController) {
        summaryRunAbortController.abort();
        summaryRunAbortController = null;
      }
    }

    function updateCurrentSelection() {
      currentPath.textContent = state.paperPath || 'No paper selected';
      updateMaintenanceMenuState();
    }

    function updateMaintenanceMenuState() {
      updateSummaryBtn.disabled = !state.paperPath;
      updateIndexBtn.disabled = !state.topicId;
    }

    function closeActionMenu() {
      actionMenu.hidden = true;
      menuBtn.setAttribute('aria-expanded', 'false');
    }

    function toggleActionMenu() {
      const nextHidden = !actionMenu.hidden;
      actionMenu.hidden = nextHidden;
      menuBtn.setAttribute('aria-expanded', nextHidden ? 'false' : 'true');
      if (!nextHidden) {
        updateMaintenanceMenuState();
      }
    }

    async function runMaintenanceTask(task, payload = {}) {
      const response = await fetch('/api/maintenance', {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ task, ...payload })
      });

      const text = await response.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = null;
      }

      if (!response.ok) {
        throw new Error((data && data.error) || text || response.statusText);
      }

      return data || { message: text };
    }

    async function runSummaryUpdateStream(summaryPath) {
      if (summaryRunAbortController) {
        summaryRunAbortController.abort();
      }
      summaryRunAbortController = new AbortController();
      const response = await fetch('/api/maintenance/stream', {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ task: 'update-summary', path: summaryPath }),
        signal: summaryRunAbortController.signal
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      if (!response.body) {
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (chunk) {
          appendSummaryRunLog(chunk);
        }
      }

      appendSummaryRunLog('\\n[done]\\n');
    }

    function renderTopics() {
      topicList.innerHTML = '';
      topicCount.textContent = String(state.topics.length);

      for (const topic of state.topics) {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.className = 'row-btn' + (state.topicId === topic.id ? ' active' : '');
        const label = document.createElement('span');
        label.textContent = topic.label;
        const meta = document.createElement('span');
        meta.className = 'meta';
        meta.textContent = topic.paperCount + ' papers';
        button.append(label, meta);
        button.addEventListener('click', () => {
          state.topicId = topic.id;
          renderTopics();
          renderFiles();
        });
        item.appendChild(button);
        topicList.appendChild(item);
      }

      updateMaintenanceMenuState();
    }

    function renderFiles() {
      paperList.innerHTML = '';
      const topic = state.topics.find((entry) => entry.id === state.topicId) || state.topics[0] || null;
      const papers = topic ? topic.papers : [];
      state.papers = papers;
      paperCount.textContent = String(papers.length);
      const activePath = topic ? state.paperPath : '';

      if (!topic) {
        paperList.innerHTML = '<li class="empty">No papers found.</li>';
        return;
      }

      if (papers.length === 0) {
        paperList.innerHTML = '<li class="empty">This topic has no papers.</li>';
        return;
      }

      for (const paper of papers) {
        const item = document.createElement('li');
        const button = document.createElement('button');
        const paperPath = resolvePaperSummaryPath(topic.id, paper.summary);
        button.className = 'row-btn' + (activePath === paperPath ? ' active' : '');
        const main = document.createElement('span');
        main.className = 'paper-main';
        const label = document.createElement('span');
        label.className = 'paper-title';
        label.textContent = paper.title || '(untitled)';
        const meta = document.createElement('span');
        meta.className = 'paper-meta';
        const details = [paper.year, paper.author].filter(Boolean).join(' · ');
        meta.textContent = details || (paper.summary ? paper.summary : 'No summary file');
        main.append(label, meta);
        button.append(main);
        button.addEventListener('click', () => openPaper(paper));
        item.appendChild(button);
        paperList.appendChild(item);
      }

      updateMaintenanceMenuState();
    }

    function updatePreview(html) {
      preview.innerHTML = html || '<div class="empty">Preview is empty.</div>';
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || response.statusText);
      }
      return response.json();
    }

    async function loadTopics(selectPath = '') {
      const data = await fetchJson('/api/topics');
      state.topics = data.topics || [];

      if (!state.topics.length) {
        topicList.innerHTML = '<li class="empty">No topics found.</li>';
        paperList.innerHTML = '<li class="empty">No papers available.</li>';
        updateCurrentSelection();
        topicCount.textContent = '0';
        paperCount.textContent = '0';
        return;
      }

      if (!state.topics.some((topic) => topic.id === state.topicId)) {
        state.topicId = state.topics[0].id;
      }

      renderTopics();
      renderFiles();

      const targetPath = selectPath || state.paperPath;
      if (targetPath) {
        const exists = state.topics.some((topic) =>
          topic.papers.some((paper) => resolvePaperSummaryPath(topic.id, paper.summary) === targetPath)
        );
        if (exists) {
          const topic = state.topics.find((entry) => entry.id === state.topicId) || state.topics[0] || null;
          const paper = topic && topic.papers
            ? topic.papers.find((entry) => resolvePaperSummaryPath(topic.id, entry.summary) === targetPath) || null
            : null;
          if (paper) {
            await openPaper(paper, { silent: true });
          }
        }
      }
    }

    async function openPaper(paper, options = {}) {
      if (state.dirty && !options.silent) {
        const shouldContinue = confirm('You have unsaved changes. Open another paper anyway?');
        if (!shouldContinue) return;
      }

      const topic = state.topics.find((entry) => entry.id === state.topicId) || state.topics[0] || null;
      const summaryPath = resolvePaperSummaryPath(topic && topic.id ? topic.id : '', paper ? paper.summary : '');
      if (!summaryPath) {
        setStatus('This paper has no summary markdown.');
        return;
      }

      setStatus('Loading paper...');
      const response = await fetch('/api/file?path=' + encodeURIComponent(summaryPath));
      if (!response.ok) {
        const message = await response.text();
        if (response.status === 404 || /ENOENT/i.test(message)) {
          showMissingPaper(summaryPath);
          return;
        }
        throw new Error(message);
      }

      const content = await response.text();
      const topicId = summaryPath.includes('/') ? summaryPath.split('/')[0] : '';
      state.topicId = topicId;
      state.paperPath = summaryPath;
      state.initialContent = content;
      editor.value = content;
      editor.placeholder = defaultEditorPlaceholder;
      setDirty(false);
      updateCurrentSelection();
      renderFiles();
      await refreshPreview();
      setStatus('Loaded ' + summaryPath);
    }

    async function refreshPreview() {
      const content = editor.value;
      if (state.previewTimer) clearTimeout(state.previewTimer);

      state.previewTimer = setTimeout(async () => {
        try {
          const response = await fetch('/api/render', {
            method: 'POST',
            headers: { 'content-type': 'text/plain; charset=utf-8' },
            body: content
          });
          if (!response.ok) throw new Error(await response.text());
          updatePreview(await response.text());
        } catch (error) {
          updatePreview('<pre>' + escapeHtml(error.message || String(error)) + '</pre>');
        }
      }, 150);
    }

    async function saveCurrentFile() {
      if (!state.paperPath) return;

      setStatus('Saving...');
      const response = await fetch('/api/file?path=' + encodeURIComponent(state.paperPath), {
        method: 'PUT',
        headers: { 'content-type': 'text/plain; charset=utf-8' },
        body: editor.value
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      state.initialContent = editor.value;
      setDirty(false);
      setStatus('Saved ' + state.paperPath);
      await loadTopics(state.paperPath);
    }

    async function createNewFile() {
      const defaultPath = state.topicId
        ? state.topicId + '/new_document.md'
        : 'new_document.md';
      const nextPath = prompt('New markdown file path', defaultPath);
      if (!nextPath) return;

      let normalized = String(nextPath).trim().replaceAll(String.fromCharCode(92), '/');
      while (normalized.startsWith('/')) {
        normalized = normalized.slice(1);
      }
      if (!normalized.toLowerCase().endsWith('.md')) {
        alert('Markdown files must end with .md');
        return;
      }

      const response = await fetch('/api/file', {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ path: normalized, content: '' })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      await loadTopics(normalized);
      const topic = state.topics.find((entry) => entry.id === state.topicId) || state.topics[0] || null;
      const paper = topic && topic.papers
        ? topic.papers.find((entry) => resolvePaperSummaryPath(topic.id, entry.summary) === normalized) || null
        : null;
      if (paper) {
        await openPaper(paper);
      }
    }

    async function deleteCurrentFile() {
      if (!state.paperPath) return;
      const confirmed = confirm('Delete ' + state.paperPath + '? This cannot be undone.');
      if (!confirmed) return;

      setStatus('Deleting...');
      const response = await fetch('/api/file?path=' + encodeURIComponent(state.paperPath), {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const deletedPath = state.paperPath;
      state.paperPath = '';
      state.initialContent = '';
      editor.value = '';
      editor.placeholder = defaultEditorPlaceholder;
      updatePreview('');
      setDirty(false);
      updateCurrentSelection();
      setStatus('Deleted ' + deletedPath);
      await loadTopics();
    }

    function onEditorInput() {
      setDirty(editor.value !== state.initialContent);
      refreshPreview();
    }

    editor.addEventListener('input', onEditorInput);
    saveBtn.addEventListener('click', () => saveCurrentFile().catch(reportError));
    deleteBtn.addEventListener('click', () => deleteCurrentFile().catch(reportError));
    refreshBtn.addEventListener('click', () => loadTopics().catch(reportError));
    menuBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      toggleActionMenu();
    });
    addTopicBtn.addEventListener('click', openTopicPopover);
    addPaperBtn.addEventListener('click', openPaperPopover);
    actionMenu.addEventListener('click', (event) => {
      event.stopPropagation();
    });
    updateSummaryBtn.addEventListener('click', () => updateSummaryFromMenu().catch(reportError));
    updateIndexBtn.addEventListener('click', () => updateIndexFromMenu().catch(reportError));
    generateManifestBtn.addEventListener('click', () => generateManifestFromMenu().catch(reportError));
    closeTopicPopoverBtn.addEventListener('click', closeTopicPopover);
    cancelTopicBtn.addEventListener('click', closeTopicPopover);
    closePaperPopoverBtn.addEventListener('click', closePaperPopover);
    cancelPaperBtn.addEventListener('click', closePaperPopover);
    closeSummaryRunBtn.addEventListener('click', closeSummaryRunPopover);
    clearSummaryRunBtn.addEventListener('click', clearSummaryRunLog);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !actionMenu.hidden) {
        closeActionMenu();
      }
      if (event.key === 'Escape' && !summaryRunBackdrop.hidden) {
        closeSummaryRunPopover();
      }
      if (!topicPopoverBackdrop.hidden && event.key === 'Escape') {
        closeTopicPopover();
      }
      if (!paperPopoverBackdrop.hidden && event.key === 'Escape') {
        closePaperPopover();
      }
    });
    document.addEventListener('click', (event) => {
      if (actionMenu.hidden) {
        return;
      }
      if (menuBtn.contains(event.target) || actionMenu.contains(event.target)) {
        return;
      }
      closeActionMenu();
    });
    topicForm.addEventListener('submit', (event) => {
      event.preventDefault();
      createTopicFromForm().catch(reportError);
    });
    paperTitleInput.addEventListener('input', updatePaperSummary);
    fetchPaperBtn.addEventListener('click', () => fetchPaperFromForm().catch(reportError));
    paperForm.addEventListener('submit', (event) => {
      event.preventDefault();
      createPaperFromForm().catch(reportError);
    });

    async function boot() {
      try {
        await loadTopics();
        setStatus('Ready');
      } catch (error) {
        reportError(error);
      }
    }

    function reportError(error) {
      setStatus(error.message || String(error), 'error');
      console.error(error);
    }

    async function createTopicFromForm() {
      const title = topicTitleInput.value.trim();
      const keyword = topicKeywordInput.value.trim();

      setStatus('Creating topic...');
      const response = await fetch('/api/topic', {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ title, keyword })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = await response.json();
      closeTopicPopover();
      await loadTopics();
      state.topicId = payload.id;
      renderTopics();
      renderFiles();
      setStatus('Created ' + payload.id);
    }

    async function fetchPaperFromForm() {
      const url = normalizePaperUrlInput(paperUrlInput.value);
      paperUrlInput.value = url;
      paperPopoverStatus.textContent = 'Fetching arXiv metadata...';

      const response = await fetch('/api/arxiv-paper?url=' + encodeURIComponent(url));
      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = await response.json();
      paperTitleInput.value = payload.title || '';
      paperAuthorInput.value = payload.author || '';
      paperYearInput.value = payload.year ? String(payload.year) : '';
      paperUrlInput.value = payload.url || url;
      updatePaperSummary();
      paperPopoverStatus.textContent = 'Metadata loaded from arXiv.';
    }

    async function createPaperFromForm() {
      const topic = state.topics.find((entry) => entry.id === state.topicId) || state.topics[0] || null;
      if (!topic) {
        throw new Error('No topic selected.');
      }

      const url = normalizePaperUrlInput(paperUrlInput.value);
      const title = paperTitleInput.value.trim();
      const author = paperAuthorInput.value.trim();
      const year = paperYearInput.value.trim();
      const summary = paperSummaryInput.value.trim() || createSummaryFilename(title);

      if (!title) {
        throw new Error('Title is required.');
      }
      if (!author) {
        throw new Error('Author is required.');
      }
      if (!year) {
        throw new Error('Year is required.');
      }

      paperPopoverStatus.textContent = 'Creating paper...';
      const response = await fetch('/api/paper', {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8' },
        body: JSON.stringify({
          topicId: topic.id,
          url,
          title,
          author,
          year,
          summary
        })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      await response.json();
      closePaperPopover();
      await loadTopics();
      setStatus('Created ' + title);
    }

    async function updateSummaryFromMenu() {
      if (!state.paperPath) {
        throw new Error('No summary file selected.');
      }

      closeActionMenu();
      openSummaryRunPopover('Updating summary...');
      setStatus('Updating summary...');

      try {
        await runSummaryUpdateStream(state.paperPath);
        summaryRunStatus.textContent = 'Completed: ' + state.paperPath;
        setStatus('Updated summary: ' + state.paperPath);
        await loadTopics(state.paperPath);
      } catch (error) {
        if (summaryRunAbortController && summaryRunAbortController.signal && summaryRunAbortController.signal.aborted) {
          summaryRunStatus.textContent = 'Cancelled.';
          setStatus('Summary update cancelled.');
          return;
        }

        summaryRunStatus.textContent = 'Failed.';
        appendSummaryRunLog('\\n[error] ' + (error.message || String(error)) + '\\n');
        throw error;
      } finally {
        summaryRunAbortController = null;
      }
    }

    async function updateIndexFromMenu() {
      if (!state.topicId) {
        throw new Error('No topic selected.');
      }

      closeActionMenu();
      setStatus('Updating index...');
      const result = await runMaintenanceTask('update-index', { topicId: state.topicId });
      if (result.stdout) console.log(result.stdout);
      if (result.stderr) console.error(result.stderr);
      setStatus('Updated index for ' + state.topicId);
    }

    async function generateManifestFromMenu() {
      closeActionMenu();
      setStatus('Generating manifest...');
      const result = await runMaintenanceTask('generate-manifest');
      if (result.stdout) console.log(result.stdout);
      if (result.stderr) console.error(result.stderr);
      setStatus('Generated manifest');
    }

    boot();
  </script>
</body>
</html>`;
}

async function handleRequest(request) {
  try {
    const hostHeader = request.headers.host ?? `${host}:${port}`;
    const requestUrl = new URL(request.url ?? '/', `http://${hostHeader}`);
    const { pathname } = requestUrl;

    if (pathname === '/' || pathname === '/index.html') {
      return textResponse(200, renderHtmlPage(), 'text/html; charset=utf-8');
    }

    if (pathname === '/api/topics' && request.method === 'GET') {
      return jsonResponse(200, { topics: await listTopics() });
    }

    if (pathname === '/api/topic' && request.method === 'POST') {
      const payload = await readJson(request);
      const created = await createTopic({
        title: payload.title,
        keyword: payload.keyword
      });

      return jsonResponse(200, created);
    }

    if (pathname === '/api/arxiv-paper' && request.method === 'GET') {
      const url = requestUrl.searchParams.get('url') ?? '';
      const paper = await fetchArxivPaperInfo(url);
      return jsonResponse(200, paper);
    }

    if (pathname === '/api/paper' && request.method === 'POST') {
      const payload = await readJson(request);
      const created = await appendPaperToTopic(String(payload.topicId ?? ''), payload);
      return jsonResponse(200, created);
    }

    if (pathname === '/api/maintenance' && request.method === 'POST') {
      const payload = await readJson(request);
      const result = await runMaintenanceTask(String(payload.task ?? ''), payload);
      return jsonResponse(200, {
        ok: true,
        task: String(payload.task ?? ''),
        stdout: result.stdout ?? '',
        stderr: result.stderr ?? ''
      });
    }

    if (pathname === '/api/maintenance/stream' && request.method === 'POST') {
      const payload = await readJson(request);
      const task = String(payload.task ?? '');
      if (task !== 'update-summary') {
        return textResponse(400, 'Only update-summary supports streaming.');
      }

      const summaryPath = String(payload.path ?? '').trim();
      if (!summaryPath) {
        return textResponse(400, 'Summary path is required.');
      }

      return streamNodeScript('scripts/update-summary.js', [summaryPath], request.signal);
    }

    if (pathname === '/api/file' && request.method === 'GET') {
      const relativePath = requestUrl.searchParams.get('path') ?? '';
      const { content } = await readMarkdownFile(relativePath);
      return textResponse(200, content, 'text/plain; charset=utf-8');
    }

    if (pathname === '/api/file' && request.method === 'POST') {
      const payload = await readJson(request);
      const relativePath = String(payload.path ?? '');
      const content = String(payload.content ?? '');
      await writeMarkdownFile(relativePath, content);
      return jsonResponse(200, { ok: true, path: normalizeRelativePath(relativePath) });
    }

    if (pathname === '/api/file' && request.method === 'PUT') {
      const relativePath = requestUrl.searchParams.get('path') ?? '';
      const body = await readRequestBody(request);
      await writeMarkdownFile(relativePath, body.toString('utf8'));
      return jsonResponse(200, { ok: true, path: normalizeRelativePath(relativePath) });
    }

    if (pathname === '/api/file' && request.method === 'DELETE') {
      const relativePath = requestUrl.searchParams.get('path') ?? '';
      await deleteMarkdownFile(relativePath);
      return jsonResponse(200, { ok: true, path: normalizeRelativePath(relativePath) });
    }

    if (pathname === '/api/render' && request.method === 'POST') {
      const body = await readRequestBody(request);
      return textResponse(200, await renderMarkdown(body.toString('utf8')), 'text/html; charset=utf-8');
    }

    return textResponse(404, 'Not found');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const status = /escape/i.test(message) ? 400 : 500;
    return textResponse(status, message);
  }
}

async function main() {
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    help();
    return;
  }

  const server = http.createServer((request, response) => {
    handleRequest(request)
      .then(async (result) => {
        response.statusCode = result.status;
        result.headers.forEach((value, key) => {
          response.setHeader(key, value);
        });
        const body = Buffer.from(await result.arrayBuffer());
        response.end(body);
      })
      .catch((error) => {
        response.statusCode = 500;
        response.setHeader('content-type', 'text/plain; charset=utf-8');
        response.end(error instanceof Error ? error.stack ?? error.message : String(error));
      });
  });

  server.listen(port, host, () => {
    console.log(`Docs dev server running at http://${host}:${port}`);
    console.log(`Managing markdown files under ${path.relative(projectRoot, docsRoot)}`);
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(1);
});
