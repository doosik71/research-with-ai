/**
 * marp-lite.ts
 *
 * A lightweight Marp-compatible Markdown-to-HTML renderer.
 * Designed for Cloudflare Pages / Edge environments (no Node.js dependencies).
 * Markdown block rendering is delegated to markdown-it; all Marp-specific
 * features (directives, SVG wrapping, math protection, bg images, etc.) are
 * handled by custom renderer-rule overrides and pre/post-processing steps.
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

import MarkdownIt, { type Options as MarkdownItOptions } from 'markdown-it';
import type Token from 'markdown-it/lib/token.mjs';
import type Renderer from 'markdown-it/lib/renderer.mjs';
import type StateCore from 'markdown-it/lib/rules_core/state_core.mjs';

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
// escapeHtml  (헤더/푸터 인라인 렌더링 등 내부에서 여전히 필요)
// ─────────────────────────────────────────────────────────────────────────────

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

// ─────────────────────────────────────────────────────────────────────────────
// buildMd — markdown-it 인스턴스 + Marp 전용 renderer-rule override
//
// 대체하는 기존 함수: renderBlocks(), renderInline(), renderImage(),
//                     renderList(), parseTableRow()
//
// Override 목록:
//   image       — ![bg ...] 플레이스홀더 / 인라인 크기·필터 속성 주입
//   fence       — 기존 동작 유지 (언어 클래스 보존)
//   heading_open — <!--fit--> 감지 → .marp-fit-heading span 삽입
//   html_block  — <style scoped> → <style data-scoped> 변환
//   bullet_list_open — `* ` 첫 마커 → marp-fragment 클래스 주입
//   ordered_list_open — `1) ` 스타일 → marp-fragment 클래스 주입
// ─────────────────────────────────────────────────────────────────────────────

function buildMd(allowHtml: boolean): MarkdownIt {
	const md = new MarkdownIt({
		html: allowHtml,
		linkify: true,
		typographer: true,
		breaks: false
	});

	// ── image: ![bg ...] / 인라인 크기·필터 ──────────────────────────────────
	md.renderer.rules.image = (tokens: Token[], idx: number): string => {
		const token = tokens[idx];
		const src = token.attrGet('src') ?? '';
		const alt = token.content; // markdown-it이 children을 stringify한 값

		const keywords = alt.split(/\s+/);
		if (keywords.includes('bg')) {
			// 배경 이미지: extractBackgrounds()가 처리할 플레이스홀더로 치환
			return `<!-- bg:${src}:${alt} -->`;
		}

		// 인라인 이미지 — 크기·필터 속성 처리
		let style = '';
		const w = alt.match(/(?:width:|w:)(\S+)/);
		const h = alt.match(/(?:height:|h:)(\S+)/);
		if (w) style += `width:${w[1]};`;
		if (h) style += `height:${h[1]};`;

		const filterKws = [
			'blur',
			'brightness',
			'contrast',
			'grayscale',
			'hue-rotate',
			'invert',
			'opacity',
			'saturate',
			'sepia',
			'drop-shadow'
		];
		const filters: string[] = [];
		for (const kw of filterKws) {
			const m = alt.match(new RegExp(`${kw}(?::([\\S]+))?`));
			if (m) filters.push(m[1] ? `${kw}(${m[1]})` : `${kw}(1)`);
		}
		if (filters.length) style += `filter:${filters.join(' ')};`;

		const cleanAlt = alt
			.replace(/(?:width:|w:|height:|h:)\S+/g, '')
			.replace(new RegExp(`(?:${filterKws.join('|')})(?::\\S+)?`, 'g'), '')
			.replace(/\s+/g, ' ')
			.trim();

		return `<img src="${escapeHtml(src)}" alt="${escapeHtml(cleanAlt)}"${style ? ` style="${style}"` : ''}>`;
	};

	// ── heading_open/inline/close: <!--fit--> 감지 ──────────────────────────
	// core rule에서 inline content를 미리 수정하고,
	// heading_open/close에 data-marp-fit 속성을 마킹한다.
	// renderer rule은 해당 속성 유무에 따라 span을 삽입/제거한다.
	md.core.ruler.push('marp_fit_heading', (state: StateCore) => {
		const tokens = state.tokens;
		for (let i = 0; i < tokens.length - 2; i++) {
			if (tokens[i].type !== 'heading_open') continue;
			const inline = tokens[i + 1];
			if (!inline || inline.type !== 'inline') continue;
			if (!/<!--\s*fit\s*-->/.test(inline.content)) continue;

			// <!--fit--> 제거
			inline.content = inline.content.replace(/<!--\s*fit\s*-->\s*/g, '').trim();
			// children 재파싱을 위해 초기화 (markdown-it이 렌더 시 재생성)
			inline.children = [];
			// heading_open / heading_close 양쪽에 마킹
			tokens[i].attrSet('data-marp-fit', '1');
			if (tokens[i + 2]?.type === 'heading_close') {
				tokens[i + 2].attrSet('data-marp-fit', '1');
			}
		}
	});

	md.renderer.rules.heading_open = (
		tokens: Token[],
		idx: number,
		options: MarkdownItOptions,
		_env: unknown,
		self: Renderer
	): string => {
		const token = tokens[idx];
		const isFit = token.attrGet('data-marp-fit') === '1';
		if (isFit)
			token.attrs = (token.attrs ?? []).filter(([k]: [string, string]) => k !== 'data-marp-fit');
		const base = self.renderToken(tokens, idx, options);
		return isFit ? base + '<span class="marp-fit-heading">' : base;
	};

	md.renderer.rules.heading_close = (
		tokens: Token[],
		idx: number,
		options: MarkdownItOptions,
		_env: unknown,
		self: Renderer
	): string => {
		const token = tokens[idx];
		const isFit = token.attrGet('data-marp-fit') === '1';
		if (isFit)
			token.attrs = (token.attrs ?? []).filter(([k]: [string, string]) => k !== 'data-marp-fit');
		const base = self.renderToken(tokens, idx, options);
		return isFit ? '</span>' + base : base;
	};

	// ── html_block: <style scoped> → <style data-scoped> ─────────────────────
	if (allowHtml) {
		md.renderer.rules.html_block = (tokens: Token[], idx: number): string => {
			return tokens[idx].content.replace(/<style(\s+scoped)(\s*>)/gi, '<style data-scoped$2');
		};
	}

	// ── bullet_list_open: `* ` 마커 → marp-fragment ──────────────────────────
	// markdown-it은 `*`와 `-` 모두 bullet_list로 처리하며
	// markup 속성으로 원래 마커 문자를 보존한다.
	md.renderer.rules.bullet_list_open = (
		tokens: Token[],
		idx: number,
		options: MarkdownItOptions,
		_env: unknown,
		self: Renderer
	): string => {
		const token = tokens[idx];
		if (token.markup === '*') {
			token.attrJoin('class', 'marp-fragment');
		}
		return self.renderToken(tokens, idx, options);
	};

	// ── ordered_list_open: `1)` 스타일 → marp-fragment ───────────────────────
	// markup이 ')' 이면 `1)` 스타일
	md.renderer.rules.ordered_list_open = (
		tokens: Token[],
		idx: number,
		options: MarkdownItOptions,
		_env: unknown,
		self: Renderer
	): string => {
		const token = tokens[idx];
		if (token.markup === ')') {
			token.attrJoin('class', 'marp-fragment');
		}
		return self.renderToken(tokens, idx, options);
	};

	return md;
}

// ─────────────────────────────────────────────────────────────────────────────
// renderInline — 헤더/푸터의 인라인 Markdown 렌더링에 사용
// (markdown-it 인스턴스를 받아 renderInline() 호출)
// ─────────────────────────────────────────────────────────────────────────────

function renderInline(text: string, md: MarkdownIt): string {
	return md.renderInline(text);
}

// ─────────────────────────────────────────────────────────────────────────────
// Math content encoder/decoder
//
// md.render() 호출 전에 수식 내부 문자를 hex 인코딩으로 보호한다.
// 인라인:  $...$   →  $hex-encoded$
// 블록:    $$...$$ →  $$hex-encoded$$
//
// 인코딩 규칙: 각 문자를 4자리 hex (codePoint)로 변환
//   예) $a$      →  $0061$
//   예) $$a\nb$$ →  $$00610a0062$$  (0a = LF)
//
// 이렇게 하면 수식 내부의 개행, 마커 문자('-', '*' 등), 들여쓰기가
// markdown-it 파서에 영향을 주지 않는다.
// ─────────────────────────────────────────────────────────────────────────────

function encodeMathContent(markdown: string): string {
	// 블록 수식 $$...$$ 를 먼저 처리 (인라인보다 우선)
	let result = markdown.replace(/([ \t]*)\$\$([\s\S]*?)\$\$/g, (_, leading, inner) => {
		const encoded = Array.from(inner as string)
			.map((ch: string) => ch.codePointAt(0)!.toString(16).padStart(4, '0'))
			.join('');
		return `${leading}$$${encoded}$$`;
	});
	// 인라인 수식 $...$ (줄 내에서만, 이미 인코딩된 $$...$$는 건드리지 않음)
	result = result.replace(/(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)/g, (_, inner) => {
		const encoded = Array.from(inner as string)
			.map((ch: string) => ch.codePointAt(0)!.toString(16).padStart(4, '0'))
			.join('');
		return `$${encoded}$`;
	});
	return result;
}

function decodeMathContent(html: string): string {
	// $$...$$ 블록 복원
	let result = html.replace(/\$\$([0-9a-f]*)\$\$/g, (_, encoded) => {
		const inner =
			encoded
				.match(/.{4}/g)
				?.map((h: string) => String.fromCodePoint(parseInt(h, 16)))
				.join('') ?? '';
		return `$$${inner}$$`;
	});
	// $...$ 인라인 복원
	result = result.replace(/(?<!\$)\$(?!\$)([0-9a-f]+?)(?<!\$)\$(?!\$)/g, (_, encoded) => {
		const inner =
			encoded
				.match(/.{4}/g)
				?.map((h: string) => String.fromCodePoint(parseInt(h, 16)))
				.join('') ?? '';
		return `$${inner}$`;
	});
	return result;
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
	if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
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
				? blockLines.join(' ').replace(/ {2}/g, '\n') // folded: 개행 → 공백
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
	private md: MarkdownIt;

	constructor(options: MarpLiteOptions = {}) {
		this.options = {
			html: options.html ?? false,
			math: options.math ?? true
		};
		this.md = buildMd(this.options.html);
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
		// const allowHtml = this.options.html;

		// viewBox 크기 결정
		const [vbW, vbH] = d.size ? sizeToViewBox(d.size) : [1280, 720];

		// 수식 내부 문자를 hex 인코딩으로 보호 (마크다운 파싱 간섭 방지)
		const encodedContent = encodeMathContent(slide.content);

		// Render Markdown content to HTML (markdown-it에 위임)
		let innerHtml = this.md.render(encodedContent);

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
			? `<div class="marp-header">${renderInline(d.header, this.md)}</div>`
			: '';

		// Footer
		const footerHtml = d.footer
			? `<div class="marp-footer">${renderInline(d.footer, this.md)}</div>`
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

		return decodeMathContent(
			[
				`<svg class="marp-svg" viewBox="0 0 ${vbW} ${vbH}"`,
				`     preserveAspectRatio="xMidYMid meet"`,
				`     xmlns="http://www.w3.org/2000/svg"`,
				`     xmlns:xhtml="http://www.w3.org/1999/xhtml"`,
				`     data-page="${pageNum}" style="display:block;">`,
				`  <foreignObject width="${vbW}" height="${vbH}">`,
				sectionHtml,
				`  </foreignObject>`,
				`</svg>`
			].join('\n')
		);
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
  align-items: center; justify-content: center;
  padding: 0.5rem;
  box-sizing: border-box;
}
svg.marp-svg {
  display: block;
  /* 최대 너비를 컨테이너에 맞추고 높이는 viewBox 비율로 자동 결정 */
  max-width: min(100%, var(--slide-max-width, 100%));
  max-height: 100dvh;
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
