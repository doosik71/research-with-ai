/**
 * marp-lite.ts
 *
 * A lightweight Marp-compatible Markdown-to-HTML renderer.
 * Designed for Cloudflare Pages / Edge environments (no Node.js dependencies).
 *
 * Supported Marp syntax:
 *  - Slide splitting by `---`
 *  - Front matter (YAML 완전 파싱: 스칼라, boolean, 숫자, 인라인 배열, 블록 스칼라 |/>)
 *  - HTML comment directives (global & local, scoped `_` prefix)
 *    · theme, paginate, headingDivider, size, style
 *    · backgroundColor, color
 *    · backgroundImage, backgroundSize, backgroundPosition, backgroundRepeat
 *    · header, footer, class
 *  - Directive inheritance + scoped (`_`) override
 *  - Built-in themes: default, gaia, uncover
 *  - headingDivider (auto slide split at headings)
 *  - Pagination (`paginate: true`)
 *  - Header / Footer (with inline Markdown)
 *  - Extended image syntax
 *    · Inline resize: `![w:200px h:100px](img.jpg)`
 *    · Background: `![bg](img.jpg)`, `![bg cover](img.jpg)`, `![bg contain](img.jpg)`
 *    · Background position: `![bg left](img.jpg)`, `![bg right](img.jpg)`
 *    · Background percentage: `![bg 50%](img.jpg)`
 *    · CSS filters: `![blur:4px](img.jpg)`, `![grayscale:1](img.jpg)`, etc.
 *  - Fragmented lists (`*` bullets, `1)` ordered)
 *  - `<!--fit-->` in headings (auto-scale)
 *  - `<style>` and `<style scoped>` blocks
 *  - MathJax math: `$inline$`, `$$block$$` (원문 보존 → MathJax 브라우저 렌더링)
 *  - Standard Markdown: headings, bold, italic, strikethrough, code, blockquote,
 *    tables, links, horizontal rule, ordered/unordered lists
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface MarpLiteOptions {
	/** Allow raw HTML in Markdown (default: false) */
	html?: boolean;
	/** Enable KaTeX math rendering (default: true) */
	math?: boolean;
}

export interface MarpLiteResult {
	/** Rendered HTML — each slide is wrapped in <svg viewBox><foreignObject><section> for auto-scaling */
	html: string;
	/** Generated CSS for all themes + directives */
	css: string;
	/** Number of slides */
	slideCount: number;
}

interface SlideDirectives {
	theme?: string;
	paginate?: boolean;
	headingDivider?: number | number[];
	size?: string;
	style?: string;
	backgroundColor?: string;
	color?: string;
	backgroundImage?: string;
	backgroundSize?: string;
	backgroundPosition?: string;
	backgroundRepeat?: string;
	header?: string;
	footer?: string;
	class?: string;
}

interface ParsedSlide {
	content: string;
	directives: SlideDirectives;
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in Themes
// ─────────────────────────────────────────────────────────────────────────────

const THEMES: Record<string, string> = {
	default: `
/* Marp-lite: default theme */
.marp-slide-wrapper { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; }
section.marp-slide {
  width: 100%; height: 100%;
  background: #fff; color: #333;
  padding: 60px 80px;
  box-sizing: border-box;
  display: flex; flex-direction: column; justify-content: center;
  position: relative; overflow: hidden;
  font-size: 28px; line-height: 1.5;
}
section.marp-slide h1 { font-size: 56px; font-weight: 700; margin: 0 0 24px; color: #1a1a1a; border-bottom: 3px solid #e0e0e0; padding-bottom: 16px; }
section.marp-slide h2 { font-size: 44px; font-weight: 600; margin: 0 0 20px; color: #222; }
section.marp-slide h3 { font-size: 36px; font-weight: 600; margin: 0 0 16px; color: #333; }
section.marp-slide h4 { font-size: 30px; font-weight: 600; margin: 0 0 12px; }
section.marp-slide p  { margin: 0 0 16px; }
section.marp-slide ul, section.marp-slide ol { margin: 0 0 16px 0; }
section.marp-slide li { margin-bottom: 8px; }
section.marp-slide code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-family: 'Courier New', monospace; }
section.marp-slide pre { background: #f0f0f0; padding: 20px; border-radius: 8px; overflow: auto; margin: 0 0 16px; }
section.marp-slide pre code { background: none; padding: 0; font-size: 0.8em; }
section.marp-slide blockquote { border-left: 4px solid #ccc; margin: 0 0 16px 0; padding: 8px 0 8px 20px; color: #666; font-style: italic; }
section.marp-slide table { border-collapse: collapse; width: 100%; margin: 0 0 16px; }
section.marp-slide th, section.marp-slide td { border: 1px solid #ddd; padding: 10px 14px; text-align: left; }
section.marp-slide th { background: #f0f0f0; font-weight: 600; }
section.marp-slide a { color: #0366d6; text-decoration: none; }
section.marp-slide.invert { background: #1a1a1a !important; color: #eee !important; }
section.marp-slide.invert h1, section.marp-slide.invert h2, section.marp-slide.invert h3 { color: #fff !important; }
section.marp-slide.invert code { background: #333; color: #eee; }
section.marp-slide.invert blockquote { border-color: #555; color: #bbb; }
`,

	gaia: `
/* Marp-lite: gaia theme */
.marp-slide-wrapper { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f0e8; }
section.marp-slide {
  width: 100%; height: 100%;
  background: #fff7ed; color: #433;
  padding: 60px 80px;
  box-sizing: border-box;
  display: flex; flex-direction: column; justify-content: center;
  position: relative; overflow: hidden;
  font-size: 28px; line-height: 1.5;
}
section.marp-slide h1 { font-size: 56px; font-weight: 700; margin: 0 0 24px; color: #c0392b; }
section.marp-slide h2 { font-size: 44px; font-weight: 600; margin: 0 0 20px; color: #e74c3c; }
section.marp-slide h3 { font-size: 36px; font-weight: 600; margin: 0 0 16px; color: #c0392b; }
section.marp-slide h4 { font-size: 30px; font-weight: 600; margin: 0 0 12px; }
section.marp-slide p  { margin: 0 0 16px; }
section.marp-slide ul, section.marp-slide ol { margin: 0 0 16px 0; }
section.marp-slide li { margin-bottom: 8px; }
section.marp-slide code { background: #fdecea; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-family: 'Courier New', monospace; color: #c0392b; }
section.marp-slide pre { background: #fdecea; padding: 20px; border-radius: 8px; overflow: auto; margin: 0 0 16px; }
section.marp-slide pre code { background: none; padding: 0; font-size: 0.8em; color: inherit; }
section.marp-slide blockquote { border-left: 4px solid #e74c3c; margin: 0 0 16px 0; padding: 8px 0 8px 20px; color: #888; font-style: italic; }
section.marp-slide table { border-collapse: collapse; width: 100%; margin: 0 0 16px; }
section.marp-slide th, section.marp-slide td { border: 1px solid #ecc; padding: 10px 14px; }
section.marp-slide th { background: #fdecea; font-weight: 600; }
section.marp-slide a { color: #c0392b; }
section.marp-slide.lead { align-items: center; text-align: center; }
section.marp-slide.lead h1 { font-size: 64px; }
section.marp-slide.invert { background: #433 !important; color: #fff7ed !important; }
section.marp-slide.invert h1, section.marp-slide.invert h2 { color: #f96 !important; }
`,

	uncover: `
/* Marp-lite: uncover theme */
.marp-slide-wrapper { font-family: 'Helvetica Neue', Arial, sans-serif; background: #e8e8e8; }
section.marp-slide {
  width: 100%; height: 100%;
  background: #fff; color: #222;
  padding: 60px 100px;
  box-sizing: border-box;
  display: flex; flex-direction: column; justify-content: center; align-items: flex-start;
  position: relative; overflow: hidden;
  font-size: 28px; line-height: 1.6;
  border-top: 6px solid #09c;
}
section.marp-slide h1 { font-size: 60px; font-weight: 300; margin: 0 0 24px; color: #09c; letter-spacing: -1px; }
section.marp-slide h2 { font-size: 46px; font-weight: 300; margin: 0 0 20px; color: #09c; }
section.marp-slide h3 { font-size: 36px; font-weight: 400; margin: 0 0 16px; color: #333; }
section.marp-slide h4 { font-size: 30px; font-weight: 400; margin: 0 0 12px; }
section.marp-slide p  { margin: 0 0 16px; }
section.marp-slide ul, section.marp-slide ol { margin: 0 0 16px 0; }
section.marp-slide li { margin-bottom: 8px; }
section.marp-slide code { background: #f0f8ff; border: 1px solid #cde; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; font-family: 'Courier New', monospace; }
section.marp-slide pre { background: #f0f8ff; border: 1px solid #cde; padding: 20px; border-radius: 6px; overflow: auto; margin: 0 0 16px; }
section.marp-slide pre code { background: none; border: none; padding: 0; font-size: 0.8em; }
section.marp-slide blockquote { border-left: 4px solid #09c; margin: 0 0 16px 0; padding: 8px 0 8px 20px; color: #666; }
section.marp-slide table { border-collapse: collapse; width: 100%; margin: 0 0 16px; }
section.marp-slide th, section.marp-slide td { border: 1px solid #cde; padding: 10px 14px; }
section.marp-slide th { background: #f0f8ff; font-weight: 600; color: #09c; }
section.marp-slide a { color: #09c; }
section.marp-slide.invert { background: #09c !important; color: #fff !important; border-top-color: #fff; }
section.marp-slide.invert h1, section.marp-slide.invert h2 { color: #fff !important; }
section.marp-slide.invert code { background: rgba(255,255,255,0.2); border-color: rgba(255,255,255,0.4); color: #fff; }
`
};

const PAGINATION_CSS = `
section.marp-slide .marp-pagination {
  position: absolute; bottom: 20px; right: 30px;
  font-size: 16px; color: #aaa; opacity: 0.7;
}
`;

const HEADER_FOOTER_CSS = `
section.marp-slide .marp-header {
  position: absolute; top: 16px; left: 80px; right: 80px;
  font-size: 18px; color: #999; border-bottom: 1px solid #eee;
  padding-bottom: 6px;
}
section.marp-slide .marp-footer {
  position: absolute; bottom: 16px; left: 80px; right: 80px;
  font-size: 18px; color: #999; border-top: 1px solid #eee;
  padding-top: 6px;
}
`;

const FIT_HEADING_CSS = `
section.marp-slide .marp-fit-heading {
  display: block; width: 100%;
  white-space: nowrap; overflow: hidden;
  font-size: clamp(16px, 5vw, 80px);
}
`;

const FRAGMENT_CSS = `
section.marp-slide ul.marp-fragment > li,
section.marp-slide ol.marp-fragment > li {
  opacity: 0; transition: opacity 0.3s;
}
section.marp-slide ul.marp-fragment > li.visible,
section.marp-slide ol.marp-fragment > li.visible {
  opacity: 1;
}
`;

const SPLIT_BG_CSS = `
section.marp-slide.marp-split-left {
  flex-direction: row; padding: 0;
}
section.marp-slide.marp-split-left .marp-split-bg {
  width: 50%; height: 100%; flex-shrink: 0;
  background-size: cover; background-position: center;
}
section.marp-slide.marp-split-left .marp-split-content {
  flex: 1; padding: 60px 60px; display: flex; flex-direction: column; justify-content: center;
}
section.marp-slide.marp-split-right {
  flex-direction: row-reverse; padding: 0;
}
section.marp-slide.marp-split-right .marp-split-bg {
  width: 50%; height: 100%; flex-shrink: 0;
  background-size: cover; background-position: center;
}
section.marp-slide.marp-split-right .marp-split-content {
  flex: 1; padding: 60px 60px; display: flex; flex-direction: column; justify-content: center;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Inline Markdown renderer (no external deps)
// ─────────────────────────────────────────────────────────────────────────────

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

function renderInline(text: string, allowHtml: boolean): string {
	// Protect code spans first
	const codeSpans: string[] = [];
	text = text.replace(/`([^`]+)`/g, (_, code) => {
		codeSpans.push(`<code>${escapeHtml(code)}</code>`);
		return `\x00CODE${codeSpans.length - 1}\x00`;
	});

	// Math $...$ / $$...$$ — MathJax가 직접 처리하도록 원문 보호
	// 볼드/이탤릭 치환에서 수식 내부의 _ * 가 오염되지 않도록 플레이스홀더로 보호 후 복원
	const mathSpans: string[] = [];
	text = text.replace(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g, (match) => {
		mathSpans.push(match);
		return `\x00MATH${mathSpans.length - 1}\x00`;
	});

	// Extended image syntax: ![alt keywords](url)
	text = text.replace(/!\[([^\]]*)\]\(([^)]*)\)/g, (_, alt, url) => {
		return renderImage(alt, url);
	});

	// Links: [text](url)
	text = text.replace(/\[([^\]]*)\]\(([^)]*)\)/g, (_, linkText, url) => {
		return `<a href="${escapeHtml(url)}">${linkText}</a>`;
	});

	// Bold + Italic
	text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
	text = text.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');
	// Bold
	text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
	text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');
	// Italic
	text = text.replace(/\*([^*\n]+?)\*/g, '<em>$1</em>');
	text = text.replace(/_([^_\n]+?)_/g, '<em>$1</em>');
	// Strikethrough
	text = text.replace(/~~(.+?)~~/g, '<del>$1</del>');

	// Restore code spans
	text = text.replace(/\x00CODE(\d+)\x00/g, (_, i) => codeSpans[parseInt(i)]);

	// Restore math spans (원문 $...$ 그대로 복원 — MathJax가 브라우저에서 처리)
	text = text.replace(/\x00MATH(\d+)\x00/g, (_, i) => mathSpans[parseInt(i)]);

	return text;
}

function renderImage(alt: string, url: string): string {
	const keywords = alt.split(/\s+/);
	const isBg = keywords.includes('bg');

	if (isBg) {
		// Background images are handled at slide level, return placeholder
		return `<!-- bg:${url}:${alt} -->`;
	}

	// Inline image with optional size / filter
	let style = '';
	const width = alt.match(/(?:width:|w:)(\S+)/);
	const height = alt.match(/(?:height:|h:)(\S+)/);
	if (width) style += `width:${width[1]};`;
	if (height) style += `height:${height[1]};`;

	// CSS filter keywords
	const filterMap: Record<string, string> = {
		blur: 'blur',
		brightness: 'brightness',
		contrast: 'contrast',
		grayscale: 'grayscale',
		'hue-rotate': 'hue-rotate',
		invert: 'invert',
		opacity: 'opacity',
		saturate: 'saturate',
		sepia: 'sepia',
		'drop-shadow': 'drop-shadow'
	};
	const filters: string[] = [];
	for (const [kw, fn] of Object.entries(filterMap)) {
		const m = alt.match(new RegExp(`${kw}(?::([\\S]+))?`));
		if (m) {
			filters.push(m[1] ? `${fn}(${m[1]})` : `${fn}(1)`);
		}
	}
	if (filters.length > 0) style += `filter:${filters.join(' ')};`;

	const cleanAlt = alt
		.replace(/(?:width:|w:|height:|h:)\S+/g, '')
		.replace(
			/(?:blur|brightness|contrast|grayscale|hue-rotate|invert|opacity|saturate|sepia|drop-shadow)(?::\S+)?/g,
			''
		)
		.trim();

	return `<img src="${escapeHtml(url)}" alt="${escapeHtml(cleanAlt)}"${style ? ` style="${style}"` : ''}>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-level Markdown renderer
// ─────────────────────────────────────────────────────────────────────────────

function renderBlocks(markdown: string, allowHtml: boolean): string {
	const lines = markdown.split('\n');
	const output: string[] = [];
	let i = 0;

	while (i < lines.length) {
		const line = lines[i];

		// ── Fenced code block
		if (/^```/.test(line)) {
			const lang = line.slice(3).trim();
			const codeLines: string[] = [];
			i++;
			while (i < lines.length && !/^```/.test(lines[i])) {
				codeLines.push(lines[i]);
				i++;
			}
			i++; // closing ```
			const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : '';
			output.push(`<pre><code${langClass}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
			continue;
		}

		// ── Math block $$...$$
		if (/^\$\$/.test(line)) {
			const mathLines: string[] = [];
			i++;
			while (i < lines.length && !/^\$\$/.test(lines[i])) {
				mathLines.push(lines[i]);
				i++;
			}
			i++;
			// $$ 구분자를 포함해서 원문 그대로 출력 → MathJax가 브라우저에서 처리
			output.push(`<div class="marp-math-block">$$\n${mathLines.join('\n')}\n$$</div>`);
			continue;
		}

		// ── <style scoped> or <style>
		if (/^<style(\s+scoped)?\s*>/i.test(line) && allowHtml) {
			const isScoped = /scoped/i.test(line);
			const styleLines: string[] = [];
			i++;
			while (i < lines.length && !/^<\/style>/i.test(lines[i])) {
				styleLines.push(lines[i]);
				i++;
			}
			i++;
			const tag = isScoped ? 'style data-scoped' : 'style';
			output.push(`<${tag}>${styleLines.join('\n')}</style>`);
			continue;
		}

		// ── HTML passthrough
		if (allowHtml && /^<[a-zA-Z]/.test(line)) {
			const htmlLines: string[] = [line];
			i++;
			while (i < lines.length && lines[i].trim() !== '') {
				htmlLines.push(lines[i]);
				i++;
			}
			output.push(htmlLines.join('\n'));
			continue;
		}

		// ── Table
		if (/\|/.test(line) && i + 1 < lines.length && /^\|?[\s\-:|]+\|/.test(lines[i + 1])) {
			const headerCells = parseTableRow(line);
			i += 2; // skip header + separator
			const rows: string[][] = [];
			while (i < lines.length && /\|/.test(lines[i])) {
				rows.push(parseTableRow(lines[i]));
				i++;
			}
			let tableHtml = '<table><thead><tr>';
			headerCells.forEach((c) => {
				tableHtml += `<th>${renderInline(c.trim(), allowHtml)}</th>`;
			});
			tableHtml += '</tr></thead><tbody>';
			rows.forEach((row) => {
				tableHtml += '<tr>';
				row.forEach((c) => {
					tableHtml += `<td>${renderInline(c.trim(), allowHtml)}</td>`;
				});
				tableHtml += '</tr>';
			});
			tableHtml += '</tbody></table>';
			output.push(tableHtml);
			continue;
		}

		// ── Blockquote
		if (/^>/.test(line)) {
			const bqLines: string[] = [];
			while (i < lines.length && /^>/.test(lines[i])) {
				bqLines.push(lines[i].replace(/^>\s?/, ''));
				i++;
			}
			output.push(`<blockquote>${renderBlocks(bqLines.join('\n'), allowHtml)}</blockquote>`);
			continue;
		}

		// ── Headings
		const headingMatch = line.match(/^(#{1,6})\s+(.*)/);
		if (headingMatch) {
			const level = headingMatch[1].length;
			let content = headingMatch[2];
			let isFit = false;
			if (/<!--\s*fit\s*-->/.test(content)) {
				isFit = true;
				content = content.replace(/<!--\s*fit\s*-->\s*/g, '').trim();
			}
			const inner = renderInline(content, allowHtml);
			if (isFit) {
				output.push(`<h${level}><span class="marp-fit-heading">${inner}</span></h${level}>`);
			} else {
				output.push(`<h${level}>${inner}</h${level}>`);
			}
			i++;
			continue;
		}

		// ── Horizontal rule (not slide separator — those are stripped before)
		if (/^(\*{3,}|-{3,}|_{3,})\s*$/.test(line)) {
			output.push('<hr>');
			i++;
			continue;
		}

		// ── Unordered list (fragmented if `*`, normal if `-` or `+`)
		if (/^(\*|-|\+)\s/.test(line)) {
			const isFragment = /^\*\s/.test(line);
			const baseIndent = line.match(/^(\s*)/)?.[1].length ?? 0;
			const listLines: string[] = [];
			while (i < lines.length) {
				const cur = lines[i];
				if (cur.trim() === '') {
					i++;
					continue;
				}
				const curIndent = cur.match(/^(\s*)/)?.[1].length ?? 0;
				if (curIndent < baseIndent) break;
				listLines.push(cur);
				i++;
			}
			const cls = isFragment ? ' class="marp-fragment"' : '';
			output.push(renderList(listLines, baseIndent, allowHtml, 'ul', cls));
			continue;
		}

		// ── Ordered list (fragmented if `1)` style)
		if (/^\d+[.)]\s/.test(line)) {
			const isFragment = /^\d+\)\s/.test(line);
			const baseIndent = line.match(/^(\s*)/)?.[1].length ?? 0;
			const listLines: string[] = [];
			while (i < lines.length) {
				const cur = lines[i];
				if (cur.trim() === '') {
					i++;
					continue;
				}
				const curIndent = cur.match(/^(\s*)/)?.[1].length ?? 0;
				if (curIndent < baseIndent) break;
				listLines.push(cur);
				i++;
			}
			const cls = isFragment ? ' class="marp-fragment"' : '';
			output.push(renderList(listLines, baseIndent, allowHtml, 'ol', cls));
			continue;
		}

		// ── Empty line → paragraph break
		if (line.trim() === '') {
			i++;
			continue;
		}

		// ── Paragraph
		const paraLines: string[] = [];
		while (
			i < lines.length &&
			lines[i].trim() !== '' &&
			!/^(#{1,6}\s|```|\$\$|>|\*\s|-\s|\+\s|\d+[.)]\s|<style)/.test(lines[i])
		) {
			paraLines.push(lines[i]);
			i++;
		}
		if (paraLines.length > 0) {
			output.push(`<p>${renderInline(paraLines.join(' '), allowHtml)}</p>`);
		}
	}

	return output.join('\n');
}

function parseTableRow(line: string): string[] {
	return line.replace(/^\||\|$/g, '').split('|');
}

/**
 * 재귀적 중첩 리스트 렌더러
 *
 * @param lines      리스트에 속하는 줄 배열 (들여쓰기 포함)
 * @param baseIndent 현재 리스트의 기준 들여쓰기 레벨
 * @param allowHtml  HTML passthrough 허용 여부
 * @param tag        'ul' | 'ol'  — 최상위 태그
 * @param cls        최상위 태그에 추가할 class 속성 문자열 (예: ' class="marp-fragment"')
 */
function renderList(
	lines: string[],
	baseIndent: number,
	allowHtml: boolean,
	tag: 'ul' | 'ol',
	cls = ''
): string {
	const html: string[] = [];
	html.push(`<${tag}${cls}>`);

	let i = 0;
	while (i < lines.length) {
		const line = lines[i];
		const indent = line.match(/^(\s*)/)?.[1].length ?? 0;

		// 현재 레벨보다 깊은 들여쓰기 → 아직 부모 li 처리 중이어야 하므로 skip
		// (아래 자식 블록 수집 로직이 이미 소비했어야 함)
		if (indent > baseIndent) {
			i++;
			continue;
		}

		// 현재 줄이 리스트 아이템인지 확인
		const isUl = /^(\*|-|\+)\s/.test(line.trimStart());
		const isOl = /^\d+[.)]\s/.test(line.trimStart());
		if (!isUl && !isOl) {
			i++;
			continue;
		}

		// 아이템 텍스트 추출 (들여쓰기 + 마커 제거)
		const itemText = line
			.replace(/^\s*/, '') // leading whitespace
			.replace(/^(\*|-|\+)\s/, '') // ul marker
			.replace(/^\d+[.)]\s/, ''); // ol marker

		// 다음 줄들 중 현재보다 깊은 들여쓰기 → 자식 리스트 줄 수집
		const childLines: string[] = [];
		let j = i + 1;
		while (j < lines.length) {
			const childIndent = lines[j].match(/^(\s*)/)?.[1].length ?? 0;
			if (lines[j].trim() === '') {
				j++;
				continue;
			} // 빈 줄 건너뜀
			if (childIndent <= baseIndent) break; // 같은 레벨 또는 상위 → 종료
			childLines.push(lines[j]);
			j++;
		}

		if (childLines.length > 0) {
			// 자식의 기준 들여쓰기: 첫 번째 자식 줄의 들여쓰기
			const childBaseIndent = childLines[0].match(/^(\s*)/)?.[1].length ?? baseIndent + 2;
			// 자식 리스트 태그 결정: 자식 첫 아이템이 ol 패턴이면 ol, 아니면 ul
			const firstChild = childLines.find((l) => l.trim() !== '');
			const childTag: 'ul' | 'ol' =
				firstChild && /^\d+[.)]\s/.test(firstChild.trimStart()) ? 'ol' : 'ul';
			// fragment 여부: 자식 첫 마커가 `*` 이면 fragment
			const childIsFragment = firstChild ? /^\*\s/.test(firstChild.trimStart()) : false;
			const childCls = childIsFragment ? ' class="marp-fragment"' : '';

			html.push(
				`<li>${renderInline(itemText, allowHtml)}` +
					renderList(childLines, childBaseIndent, allowHtml, childTag, childCls) +
					`</li>`
			);
		} else {
			html.push(`<li>${renderInline(itemText, allowHtml)}</li>`);
		}

		i = j;
	}

	html.push(`</${tag}>`);
	return html.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Directive parser
// ─────────────────────────────────────────────────────────────────────────────

function parseDirectives(text: string): [SlideDirectives, string] {
	const directives: SlideDirectives = {};
	// Extract <!-- key: value --> comments
	const cleaned = text.replace(/<!--([\s\S]*?)-->/g, (match, inner) => {
		const lines = inner.trim().split('\n');
		let consumed = false;
		for (const line of lines) {
			const m = line.match(/^\s*_?(\w+)\s*:\s*(.+?)\s*$/);
			if (m) {
				consumed = true;
				const key = m[1] as keyof SlideDirectives;
				const val = m[2].replace(/^['"]|['"]$/g, '');
				applyDirective(directives, key, val);
			}
		}
		return consumed ? '' : match; // keep non-directive comments
	});
	return [directives, cleaned];
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal YAML parser (외부 의존성 없이 Marp Front Matter 전용)
//
// 지원 문법:
//   key: scalar value          # 단순 스칼라
//   key: "quoted value"        # 따옴표 스칼라
//   key: 'single quoted'       # 단따옴표 스칼라
//   key: true / false          # boolean
//   key: 123                   # 숫자
//   key: [1, 2, 3]             # 인라인 배열
//   key: |                     # 블록 스칼라 (literal, 개행 보존)
//     line1
//     line2
//   key: >                     # 블록 스칼라 (folded, 개행 → 공백)
//     line1
//     line2
//   # comment                  # 주석 무시
// ─────────────────────────────────────────────────────────────────────────────

type YamlScalar = string | boolean | number | string[] | number[];

function parseYamlValue(raw: string): YamlScalar {
	const s = raw.trim();

	// boolean
	if (s === 'true') return true;
	if (s === 'false') return false;

	// null / empty
	if (s === '' || s === 'null' || s === '~') return '';

	// 따옴표 제거
	if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("\'"))) {
		return s.slice(1, -1);
	}

	// 인라인 배열 [a, b, c]
	if (s.startsWith('[') && s.endsWith(']')) {
		return s
			.slice(1, -1)
			.split(',')
			.map((item) => {
				const trimmed = item.trim().replace(/^['"]|['"]$/g, '');
				const n = Number(trimmed);
				return isNaN(n) ? trimmed : n;
			}) as string[] | number[];
	}

	// 숫자
	const n = Number(s);
	if (!isNaN(n) && s !== '') return n;

	// 주석 제거 (값 뒤 # comment)
	const commentIdx = s.search(/\s+#/);
	if (commentIdx !== -1) return s.slice(0, commentIdx).trim();

	return s;
}

function parseFrontMatter(markdown: string): [SlideDirectives, string] {
	const directives: SlideDirectives = {};

	// --- 구분자 감지 (CRLF/LF 모두 처리)
	const normalized = markdown.replace(/\r\n/g, '\n');
	const fm = normalized.match(/^---\n([\s\S]*?)\n---(?:\n|$)/);
	if (!fm) return [directives, markdown];

	const body = fm[1];
	const rest = normalized.slice(fm[0].length);
	const lines = body.split('\n');

	let i = 0;
	while (i < lines.length) {
		const line = lines[i];

		// 빈 줄 / 주석 스킵
		if (line.trim() === '' || /^\s*#/.test(line)) {
			i++;
			continue;
		}

		// key: 패턴
		const keyMatch = line.match(/^(\w+)\s*:\s*(.*)/);
		if (!keyMatch) {
			i++;
			continue;
		}

		const key = keyMatch[1];
		const valueRaw = keyMatch[2].trim();

		// 블록 스칼라 | (literal) 또는 > (folded)
		if (valueRaw === '|' || valueRaw === '>') {
			const isFolded = valueRaw === '>';
			const blockLines: string[] = [];
			// 기준 들여쓰기: 다음 줄에서 결정
			i++;
			let baseIndent = -1;
			while (i < lines.length) {
				const bLine = lines[i];
				if (bLine.trim() === '') {
					blockLines.push('');
					i++;
					continue;
				}
				const indent = bLine.match(/^(\s*)/)?.[1].length ?? 0;
				if (baseIndent === -1) baseIndent = indent;
				if (indent < baseIndent) break; // 블록 종료
				blockLines.push(bLine.slice(baseIndent));
				i++;
			}
			// 후행 빈 줄 제거
			while (blockLines.length > 0 && blockLines[blockLines.length - 1].trim() === '') {
				blockLines.pop();
			}
			const blockVal = isFolded
				? blockLines.join(' ').replace(/  /g, '\n') // folded: 개행 → 공백
				: blockLines.join('\n'); // literal: 개행 보존
			applyDirective(directives, key, blockVal);
			continue;
		}

		// 일반 스칼라 / 배열
		const parsed = parseYamlValue(valueRaw);
		// headingDivider 배열 처리
		if (key === 'headingDivider' && Array.isArray(parsed)) {
			directives.headingDivider = (parsed as number[]).map(Number).filter((n) => !isNaN(n));
		} else {
			applyDirective(directives, key, String(parsed));
		}
		i++;
	}

	return [directives, rest];
}

function applyDirective(d: SlideDirectives, key: string, val: string | boolean | number): void {
	const str = String(val);
	switch (key) {
		case 'marp':
			/* marp: true — 활성화 플래그, 값 자체는 무시 */ break;
		case 'theme':
			d.theme = str;
			break;
		case 'paginate':
			d.paginate = val === true || str === 'true';
			break;
		case 'size':
			d.size = str;
			break;
		case 'style':
			d.style = str;
			break;
		case 'backgroundColor':
			d.backgroundColor = str;
			break;
		case 'color':
			d.color = str;
			break;
		case 'backgroundImage':
			d.backgroundImage = str;
			break;
		case 'backgroundSize':
			d.backgroundSize = str;
			break;
		case 'backgroundPosition':
			d.backgroundPosition = str;
			break;
		case 'backgroundRepeat':
			d.backgroundRepeat = str;
			break;
		case 'header':
			d.header = str;
			break;
		case 'footer':
			d.footer = str;
			break;
		case 'class':
			d.class = str;
			break;
		case 'headingDivider': {
			const n = Number(val);
			d.headingDivider = isNaN(n) ? undefined : n;
			break;
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Background image processing
// ─────────────────────────────────────────────────────────────────────────────

interface BgInfo {
	url: string;
	size: string;
	position: string;
	split?: 'left' | 'right';
	filter?: string;
}

function extractBackgrounds(html: string): [BgInfo[], string] {
	const bgs: BgInfo[] = [];
	const cleaned = html.replace(/<!-- bg:([^:]+):([^>]*) -->/g, (_, url, alt) => {
		const keywords = alt.split(/\s+/);
		const bg: BgInfo = { url, size: 'cover', position: 'center' };

		if (keywords.includes('contain') || keywords.includes('fit')) bg.size = 'contain';
		else if (keywords.includes('auto')) bg.size = 'auto';
		else {
			const pct = keywords.find((k: string) => /^\d+%$/.test(k));
			if (pct) bg.size = pct;
		}
		if (keywords.includes('left')) bg.split = 'left';
		if (keywords.includes('right')) bg.split = 'right';

		// Filters
		const filterKws = [
			'blur',
			'brightness',
			'contrast',
			'grayscale',
			'hue-rotate',
			'invert',
			'opacity',
			'saturate',
			'sepia'
		];
		const filters: string[] = [];
		for (const kw of filterKws) {
			const m = alt.match(new RegExp(`${kw}(?::([\\S]+))?`));
			if (m) filters.push(m[1] ? `${kw}(${m[1]})` : `${kw}(1)`);
		}
		if (filters.length > 0) bg.filter = filters.join(' ');

		bgs.push(bg);
		return '';
	});
	return [bgs, cleaned];
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide size helpers
// ─────────────────────────────────────────────────────────────────────────────

/** size 디렉티브 → [width, height] 숫자 튜플 반환 (SVG viewBox용) */
function sizeToViewBox(size: string): [number, number] {
	const presets: Record<string, [number, number]> = {
		'16:9': [1280, 720],
		'4:3': [960, 720],
		'4K': [3840, 2160]
	};
	if (presets[size]) return presets[size];
	// custom e.g. "1920px 1080px" or "1920 1080"
	const parts = size.split(/\s+/).map((s) => parseInt(s));
	if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
		return [parts[0], parts[1]];
	}
	return [1280, 720]; // fallback
}

// ─────────────────────────────────────────────────────────────────────────────
// headingDivider pre-processing
// ─────────────────────────────────────────────────────────────────────────────

function applyHeadingDivider(markdown: string, level: number | number[]): string {
	const levels = Array.isArray(level) ? level : [level];
	const pattern = new RegExp(`^(#{1,6})\\s`, 'gm');
	return markdown.replace(pattern, (match, hashes) => {
		if (levels.includes(hashes.length)) {
			return `\n---\n${match}`;
		}
		return match;
	});
}

// ─────────────────────────────────────────────────────────────────────────────
// MarpLite class
// ─────────────────────────────────────────────────────────────────────────────

export class MarpLite {
	private options: Required<MarpLiteOptions>;

	constructor(options: MarpLiteOptions = {}) {
		this.options = {
			html: options.html ?? false,
			math: options.math ?? true
		};
	}

	render(markdown: string): MarpLiteResult {
		// 1. Parse front matter (global directives)
		const [globalDirs, bodyMd] = parseFrontMatter(markdown.trimStart());

		// 2. Apply headingDivider if set
		let processedMd = bodyMd;
		if (globalDirs.headingDivider !== undefined) {
			processedMd = applyHeadingDivider(processedMd, globalDirs.headingDivider);
		}

		// 3. Split into raw slide blocks by `---`
		const rawSlides = splitSlides(processedMd);

		// 4. Parse per-slide directives (with inheritance)
		const slides = this.parseSlides(rawSlides, globalDirs);

		// 5. Render each slide to HTML
		const theme = globalDirs.theme ?? 'default';
		const slideHtmls = slides.map((slide, idx) =>
			this.renderSlide(slide, idx + 1, slides.length, theme)
		);

		// 6. Build CSS
		const css = this.buildCSS(globalDirs, slides);

		return {
			html: slideHtmls.join('\n'),
			css,
			slideCount: slides.length
		};
	}

	private parseSlides(rawSlides: string[], globalDirs: SlideDirectives): ParsedSlide[] {
		const slides: ParsedSlide[] = [];
		// Inherited local directives
		const inherited: SlideDirectives = {
			paginate: globalDirs.paginate,
			header: globalDirs.header,
			footer: globalDirs.footer,
			class: globalDirs.class,
			backgroundColor: globalDirs.backgroundColor,
			color: globalDirs.color,
			backgroundImage: globalDirs.backgroundImage
		};

		for (const raw of rawSlides) {
			const [localDirs, content] = parseDirectives(raw);

			// Scoped directives (underscore prefix) — apply only to this slide
			const scoped: SlideDirectives = {};
			const rawScoped = raw.match(/<!--([\s\S]*?)-->/g) ?? [];
			for (const comment of rawScoped) {
				const lines = comment
					.replace(/<!--|-->/g, '')
					.trim()
					.split('\n');
				for (const line of lines) {
					const m = line.match(/^\s*_(\w+)\s*:\s*(.+?)\s*$/);
					if (m) {
						const key = m[1] as keyof SlideDirectives;
						const val = m[2].replace(/^['"]|['"]$/g, '');
						applyDirective(scoped, key, val);
					}
				}
			}

			// Non-scoped local directives update inheritance
			const merged: SlideDirectives = { ...inherited, ...localDirs, ...scoped };

			// Update inherited state (non-scoped only)
			if (localDirs.paginate !== undefined) inherited.paginate = localDirs.paginate;
			if (localDirs.header !== undefined) inherited.header = localDirs.header;
			if (localDirs.footer !== undefined) inherited.footer = localDirs.footer;
			if (localDirs.class !== undefined) inherited.class = localDirs.class;
			if (localDirs.backgroundColor !== undefined)
				inherited.backgroundColor = localDirs.backgroundColor;
			if (localDirs.color !== undefined) inherited.color = localDirs.color;
			if (localDirs.backgroundImage !== undefined)
				inherited.backgroundImage = localDirs.backgroundImage;

			slides.push({ content: content.trim(), directives: merged });
		}
		return slides;
	}

	private renderSlide(
		slide: ParsedSlide,
		pageNum: number,
		total: number,
		globalTheme: string
	): string {
		const d = slide.directives;
		const allowHtml = this.options.html;

		// viewBox 크기 결정
		const [vbW, vbH] = d.size ? sizeToViewBox(d.size) : [1280, 720];

		// Render Markdown content to HTML
		let innerHtml = renderBlocks(slide.content, allowHtml);

		// Extract background image placeholders
		const [bgs, cleanedHtml] = extractBackgrounds(innerHtml);
		innerHtml = cleanedHtml;

		// section 인라인 스타일 수집
		const styles: string[] = [];
		if (d.backgroundColor) styles.push(`background-color:${d.backgroundColor}`);
		if (d.color) styles.push(`color:${d.color}`);
		if (d.backgroundImage && bgs.length === 0) {
			styles.push(`background-image:url(${d.backgroundImage})`);
			styles.push(`background-size:${d.backgroundSize ?? 'cover'}`);
			styles.push(`background-position:${d.backgroundPosition ?? 'center'}`);
			styles.push(`background-repeat:${d.backgroundRepeat ?? 'no-repeat'}`);
		}

		// Classes
		const classes = ['marp-slide'];
		if (d.class) classes.push(...d.class.split(/\s+/));
		if (globalTheme === 'gaia' && classes.includes('lead')) classes.push('lead');

		// Handle split background
		const splitBg = bgs.find((b) => b.split);
		if (splitBg) classes.push(`marp-split-${splitBg.split}`);

		const classAttr = ` class="${classes.join(' ')}"`;

		// Build background layers
		let bgLayerHtml = '';

		if (splitBg) {
			const filterStyle = splitBg.filter ? `filter:${splitBg.filter};` : '';
			bgLayerHtml = `<div class="marp-split-bg" style="background-image:url(${escapeHtml(splitBg.url)});background-size:${splitBg.size};${filterStyle}"></div>`;
			innerHtml = `<div class="marp-split-content">${innerHtml}</div>`;
		} else if (bgs.length === 1) {
			const bg = bgs[0];
			const filterStyle = bg.filter ? `filter:${bg.filter};` : '';
			styles.push(`background-image:url(${escapeHtml(bg.url)})`);
			styles.push(`background-size:${bg.size}`);
			styles.push(`background-position:${bg.position}`);
			styles.push(`background-repeat:no-repeat`);
			if (filterStyle) styles.push(filterStyle.replace(';', ''));
		} else if (bgs.length > 1) {
			// Multiple backgrounds → stacked absolute divs
			bgLayerHtml = bgs
				.map((bg) => {
					const filterStyle = bg.filter ? `filter:${bg.filter};` : '';
					return `<div style="position:absolute;inset:0;background-image:url(${escapeHtml(bg.url)});background-size:${bg.size};background-position:${bg.position};background-repeat:no-repeat;${filterStyle}opacity:${1 / bgs.length}"></div>`;
				})
				.join('');
		}

		// Header
		const headerHtml = d.header
			? `<div class="marp-header">${renderInline(d.header, allowHtml)}</div>`
			: '';

		// Footer
		const footerHtml = d.footer
			? `<div class="marp-footer">${renderInline(d.footer, allowHtml)}</div>`
			: '';

		// Pagination
		const paginationHtml = d.paginate
			? `<div class="marp-pagination">${pageNum} / ${total}</div>`
			: '';

		const sectionStyle = styles.length > 0 ? ` style="${styles.join(';')}"` : '';

		// ── SVG 래핑: viewBox 기반 자동 배율 조절
		// foreignObject 내부는 일반 XHTML이므로 기존 CSS/MathJax 그대로 동작
		const sectionHtml = [
			`<section xmlns="http://www.w3.org/1999/xhtml"${classAttr}${sectionStyle} data-page="${pageNum}">`,
			bgLayerHtml,
			headerHtml,
			innerHtml,
			footerHtml,
			paginationHtml,
			`</section>`
		]
			.filter(Boolean)
			.join('\n');

		return [
			`<svg class="marp-svg" viewBox="0 0 ${vbW} ${vbH}"`,
			`     width="100%" preserveAspectRatio="xMidYMid meet"`,
			`     xmlns="http://www.w3.org/2000/svg"`,
			`     xmlns:xhtml="http://www.w3.org/1999/xhtml"`,
			`     data-page="${pageNum}" style="display:block;max-height:100%;">`,
			`  <foreignObject width="${vbW}" height="${vbH}">`,
			sectionHtml,
			`  </foreignObject>`,
			`</svg>`
		].join('\n');
	}

	private buildCSS(globalDirs: SlideDirectives, slides: ParsedSlide[]): string {
		const theme = globalDirs.theme ?? 'default';
		const themeCSS = THEMES[theme] ?? THEMES['default'];

		const parts: string[] = [
			themeCSS,
			PAGINATION_CSS,
			HEADER_FOOTER_CSS,
			FIT_HEADING_CSS,
			FRAGMENT_CSS,
			SPLIT_BG_CSS
		];

		// Global style directive
		if (globalDirs.style) {
			parts.push(`/* global style directive */\n${globalDirs.style}`);
		}

		// Per-slide <style> and <style scoped> blocks
		slides.forEach((slide, idx) => {
			const pageNum = idx + 1;
			// Scoped styles extracted from rendered HTML
			const scopedMatches = slide.content.matchAll(
				/<style\s+data-scoped[^>]*>([\s\S]*?)<\/style>/gi
			);
			for (const m of scopedMatches) {
				// Scope to this slide
				const scopedCss = m[1].replace(/section/g, `section[data-page="${pageNum}"]`);
				parts.push(`/* scoped style: slide ${pageNum} */\n${scopedCss}`);
			}
			const globalMatches = slide.content.matchAll(
				/<style(?!\s+(?:data-scoped|scoped))[^>]*>([\s\S]*?)<\/style>/gi
			);
			for (const m of globalMatches) {
				parts.push(m[1]);
			}
		});

		// KaTeX-like math CSS (minimal)
		if (this.options.math) {
			parts.push(`
/* MathJax 렌더링 전 깜빡임 방지 — 렌더링 후 MathJax가 스타일을 덮어씀 */
.marp-math-block  { text-align: center; margin: 16px 0; }
      `);
		}

		// Wrapper
		parts.push(`
/* SVG 래퍼: viewBox 기반 자동 배율 조절 */
.marp-slide-wrapper {
  display: flex; flex-direction: column; gap: 1rem;
  align-items: center; padding: 0.5rem;
}
svg.marp-svg {
  display: block;
  /* width="100%" 은 인라인 속성으로 지정됨 */
  /* 최대 너비를 컨테이너에 맞추고 높이는 viewBox 비율로 자동 결정 */
  max-width: 100%;
  border: 1px lightgray solid;
  filter: drop-shadow(3px 3px 4px rgba(0, 0, 0, 0.3));
}
/* foreignObject 내부 section은 width/height 100%로 채움 */
svg.marp-svg > foreignObject > section.marp-slide {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  overflow: hidden;
}
    `);

		return parts.join('\n');
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Slide splitter
// ─────────────────────────────────────────────────────────────────────────────

function splitSlides(markdown: string): string[] {
	// Split on lines that are exactly `---` (not inside code blocks)
	const lines = markdown.split('\n');
	const slides: string[] = [];
	let current: string[] = [];
	let inCode = false;

	for (const line of lines) {
		if (/^```/.test(line)) inCode = !inCode;
		if (!inCode && /^---\s*$/.test(line)) {
			slides.push(current.join('\n'));
			current = [];
		} else {
			current.push(line);
		}
	}
	slides.push(current.join('\n'));
	return slides.filter((s) => s.trim().length > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Default export
// ─────────────────────────────────────────────────────────────────────────────

export default MarpLite;
