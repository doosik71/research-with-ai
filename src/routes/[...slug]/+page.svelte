<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/stores';
	import { browser } from '$app/environment';
	import MarkdownIt from 'markdown-it';
	import { MarpLite } from '../marp-lite';

	type Topic = {
		id: string;
		title: string;
		keyword: string[];
	};

	type Paper = {
		title: string;
		author: string;
		year: number | string;
		url: string;
		summary: string;
		slide: string;
		index: number;
	};

	type PaperTagEntry = {
		raw: string;
		updatedAt: number;
		expiresAt: number;
	};

	type PaperTagMap = Record<string, PaperTagEntry>;

	type HighlightColor = 'yellow' | 'pink' | 'blue' | 'green';

	type SummaryHighlightEntry = {
		id: string;
		groupId: string;
		docPath: string;
		color: HighlightColor;
		text: string;
		createdAt: number;
	};

	type HighlightPopoverMode = 'create' | 'edit';

	type HighlightPopoverState = {
		mode: HighlightPopoverMode;
		anchorLeft: number;
		anchorTop: number;
		left: number;
		top: number;
		groupId?: string;
		blocks?: string[];
	};

	type SelectionInteractionMode = 'mouse' | 'touch' | 'pen' | 'keyboard';

	type HighlightPopoverOffset = {
		x: number;
		y: number;
	};

	type PaperListItem =
		| { kind: 'year'; yearLabel: string; yearValue: number | null; key: string }
		| { kind: 'paper'; paper: Paper; key: string };
	let topics = $state<Topic[]>([]);
	let papers = $state<Paper[]>([]);
	let selectedTopic = $state<Topic | null>(null);
	let selectedPaper = $state<Paper | null>(null);
	let showTopicPanel = $state(true);
	let showPaperPanel = $state(true);
	let isMobile = $state(false);
	let mobileStage = $state<0 | 1 | 2>(0); // 0: topics, 1: papers, 2: preview
	let isLocalhost = $state(false);

	type RoutePaperMatch = { paper: Paper; type: 'summary' | 'slide' };
	const PAPER_TAG_STORAGE_KEY = 'paper-tags-v1';
	const PAPER_TAG_TTL_MS = 1000 * 60 * 60 * 24 * 30 * 6;
	const SUMMARY_HIGHLIGHT_STORAGE_KEY = 'summary-highlights-v1';
	const SUMMARY_HIGHLIGHT_COLORS: HighlightColor[] = ['yellow', 'pink', 'blue', 'green'];
	const HIGHLIGHT_BLOCK_SELECTOR = 'p, li, blockquote, td, th, h1, h2, h3, h4, h5, h6';
	const HIGHLIGHT_EXCLUDED_SELECTOR =
		'a, pre, code, img, script, style, mjx-container, .MathJax, .summary-highlight';

	let pdfjsModulePromise: Promise<
		typeof import('pdfjs-dist') & { GlobalWorkerOptions: { workerSrc: string } }
	> | null = null;

	async function loadPdfjs() {
		if (!browser) return null;
		if (pdfjsModulePromise) return pdfjsModulePromise;

		pdfjsModulePromise = (async () => {
			const [pdfjs, workerUrl] = await Promise.all([
				import('pdfjs-dist'),
				import('pdfjs-dist/build/pdf.worker.min.mjs?url')
			]);
			pdfjs.GlobalWorkerOptions.workerSrc = workerUrl.default;
			return pdfjs;
		})();

		return pdfjsModulePromise;
	}

	function basename(path: string): string {
		const normalized = String(path ?? '');
		const slashSplit = normalized.split('/');
		const lastSlash = slashSplit[slashSplit.length - 1] ?? normalized;
		const backSplit = lastSlash.split('\\');
		return backSplit[backSplit.length - 1] ?? lastSlash;
	}

	function paperSlugFromFilename(filename: string): string {
		return basename(filename).replace(/\.md$/i, '');
	}

	function buildPaperTagKey(topicId: string | undefined, paper: Paper | null): string | null {
		if (!topicId || !paper?.summary) return null;
		const summarySlug = paperSlugFromFilename(paper.summary);
		return summarySlug ? `${topicId}/${summarySlug}` : null;
	}

	function parseTagList(raw: string): string[] {
		const tags: string[] = [];

		for (const part of raw.split(',')) {
			const tag = part.trim();
			if (!tag || tags.includes(tag)) continue;
			tags.push(tag);
		}

		return tags;
	}

	function loadPaperTags(): PaperTagMap {
		if (typeof window === 'undefined') return {};

		try {
			const raw = localStorage.getItem(PAPER_TAG_STORAGE_KEY);
			if (!raw) return {};
			const parsed = JSON.parse(raw);
			return parsed && typeof parsed === 'object' ? parsed : {};
		} catch {
			return {};
		}
	}

	function persistPaperTags(next: PaperTagMap) {
		paperTags = next;

		if (typeof window === 'undefined') return;

		try {
			if (Object.keys(next).length === 0) {
				localStorage.removeItem(PAPER_TAG_STORAGE_KEY);
				return;
			}
			localStorage.setItem(PAPER_TAG_STORAGE_KEY, JSON.stringify(next));
		} catch {
			// Do nothing.
		}
	}

	function loadSummaryHighlights(): SummaryHighlightEntry[] {
		if (typeof window === 'undefined') return [];

		try {
			const raw = localStorage.getItem(SUMMARY_HIGHLIGHT_STORAGE_KEY);
			if (!raw) return [];
			const parsed = JSON.parse(raw);
			return Array.isArray(parsed) ? parsed : [];
		} catch {
			return [];
		}
	}

	function persistSummaryHighlights(next: SummaryHighlightEntry[]) {
		summaryHighlights = next;

		if (typeof window === 'undefined') return;

		try {
			if (next.length === 0) {
				localStorage.removeItem(SUMMARY_HIGHLIGHT_STORAGE_KEY);
				return;
			}
			localStorage.setItem(SUMMARY_HIGHLIGHT_STORAGE_KEY, JSON.stringify(next));
		} catch {
			// Do nothing.
		}
	}

	function normalizeHighlightText(text: string): string {
		return text.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
	}

	function currentSummaryHighlights(docPath: string | null): SummaryHighlightEntry[] {
		if (!docPath) return [];
		return summaryHighlights.filter((entry) => entry.docPath === docPath);
	}

	function buildHighlightGroupId(): string {
		if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
			return crypto.randomUUID();
		}
		return `hl-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
	}

	function tagsForPaper(topicId: string | undefined, paper: Paper | null): string[] {
		const key = buildPaperTagKey(topicId, paper);
		if (!key) return [];
		return parseTagList(paperTags[key]?.raw ?? '');
	}

	function firstHighlightColorForPaper(
		topicId: string | undefined,
		paper: Paper | null
	): HighlightColor | null {
		const key = buildPaperTagKey(topicId, paper);
		if (!key) return null;
		return summaryHighlights.find((entry) => entry.docPath === key)?.color ?? null;
	}

	function findPaperBySlug(paperSlug: string, allPapers: Paper[]): RoutePaperMatch | null {
		const targetFilename = `${paperSlug}.md`;

		for (const paper of allPapers) {
			if (paper.summary && basename(paper.summary) === targetFilename)
				return { paper, type: 'summary' };
			if (paper.slide && basename(paper.slide) === targetFilename) return { paper, type: 'slide' };
		}

		return null;
	}

	// SvelteKit rest params are strings like "a/b/c" (or `undefined` at `/`).
	let routeSlug = $derived($page.params.slug as string | undefined);
	let routeSegments = $derived((routeSlug ? routeSlug.split('/').filter(Boolean) : []) as string[]);
	let routeTopicId = $derived(routeSegments[0]);
	let routePaperSlug = $derived(routeSegments[1]);
	let routeHasExtra = $derived(routeSegments.length > 2);

	let appliedRouteKey = $state('');
	let routeApplySeq = 0;

	$effect(() => {
		if (!isMobile) return;

		// Ensure desktop-only panel toggles don't hide screens on mobile.
		showTopicPanel = true;
		showPaperPanel = true;

		// Only correct invalid states (don't force-forward if user navigated back).
		if (!selectedTopic && mobileStage !== 0) mobileStage = 0;
		if (mobileStage === 2 && !selectedPaper) mobileStage = 1;
	});

	// Search State
	let showSearchTopicInput = $state(false);
	let showSearchPaperInput = $state(false);
	let searchTopicQuery = $state('');
	let searchPaperQuery = $state('');
	let topicSearchInputEl = $state<HTMLInputElement | null>(null);
	let paperSearchInputEl = $state<HTMLInputElement | null>(null);
	let paperTags = $state<PaperTagMap>({});
	let editingTagKey = $state<string | null>(null);
	let tagDraft = $state('');
	let tagInputEl = $state<HTMLInputElement | null>(null);
	let skipTagBlurSave = $state(false);
	let slidePreviewContentEl = $state<HTMLDivElement | null>(null);
	let readerBodyEl = $state<HTMLDivElement | null>(null);
	let highlightPopoverEl = $state<HTMLDivElement | null>(null);
	let summaryHighlights = $state<SummaryHighlightEntry[]>([]);
	let highlightPopover = $state<HighlightPopoverState | null>(null);
	let renderedSummaryDocKey = $state<string | null>(null);
	let lastSelectionInteractionMode = $state<SelectionInteractionMode>('mouse');
	let highlightPopoverOffset = $state<HighlightPopoverOffset>({ x: 0, y: 0 });
	let touchStartX = 0;
	let touchStartY = 0;
	let selectedPaperTagKey = $derived(buildPaperTagKey(selectedTopic?.id, selectedPaper));
	let selectedPaperTags = $derived(tagsForPaper(selectedTopic?.id, selectedPaper));
	let selectedSummaryDocKey = $derived(buildPaperTagKey(selectedTopic?.id, selectedPaper));

	async function openSearchPopover(kind: 'topic' | 'paper') {
		if (kind === 'topic') {
			if (showSearchTopicInput) {
				showSearchTopicInput = false;
				return;
			}
			showSearchTopicInput = true;
			await tick();
			topicSearchInputEl?.focus();
			topicSearchInputEl?.select();
			return;
		}

		if (showSearchPaperInput) {
			showSearchPaperInput = false;
			return;
		}
		showSearchPaperInput = true;
		await tick();
		paperSearchInputEl?.focus();
		paperSearchInputEl?.select();
	}

	function closeSearchPopoverOnFocusOut(event: FocusEvent, kind: 'topic' | 'paper') {
		const container = event.currentTarget as HTMLElement | null;
		const next = event.relatedTarget as Node | null;
		if (container && next && container.contains(next)) return;

		if (kind === 'topic') showSearchTopicInput = false;
		else showSearchPaperInput = false;
	}

	async function openTagEditor(key: string | null, initialValue: string) {
		if (!key) return;
		editingTagKey = key;
		tagDraft = initialValue;
		await tick();
		tagInputEl?.focus();
		tagInputEl?.select();
	}

	async function openSelectedPaperTagEditor() {
		if (!selectedPaperTagKey) return;
		await openTagEditor(selectedPaperTagKey, paperTags[selectedPaperTagKey]?.raw ?? '');
	}

	function cancelTagEdit() {
		editingTagKey = null;
		tagDraft = '';
	}

	function saveTagEditOnBlur(key: string | null) {
		if (skipTagBlurSave) {
			skipTagBlurSave = false;
			return;
		}
		saveTagEdit(key);
	}

	function saveTagEdit(key: string | null) {
		if (!key) return;

		const normalized = tagDraft.trim();
		if (!normalized) {
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			const { [key]: _removed, ...rest } = paperTags;
			persistPaperTags(rest);
			cancelTagEdit();
			return;
		}

		const parsedTags = parseTagList(normalized);
		if (parsedTags.length === 0) {
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			const { [key]: _removed, ...rest } = paperTags;
			persistPaperTags(rest);
			cancelTagEdit();
			return;
		}

		const now = Date.now();
		persistPaperTags({
			...paperTags,
			[key]: {
				raw: parsedTags.join(', '),
				updatedAt: now,
				expiresAt: now + PAPER_TAG_TTL_MS
			}
		});
		cancelTagEdit();
	}

	function getActiveSlideScrollContainer(): HTMLElement | null {
		if (renderType !== 'slide' || isLoading || !renderHtml) return null;
		return readerMode ? readerBodyEl : slidePreviewContentEl;
	}

	function getSlideElements(container: HTMLElement): SVGElement[] {
		return Array.from(container.querySelectorAll('.marp-slide-wrapper .marp-svg'));
	}

	function getCurrentSlideIndex(container: HTMLElement, slides: SVGElement[]): number {
		if (slides.length === 0) return -1;

		const containerRect = container.getBoundingClientRect();
		let closestIndex = 0;
		let closestDistance = Number.POSITIVE_INFINITY;

		for (let index = 0; index < slides.length; index += 1) {
			const rect = slides[index].getBoundingClientRect();
			const distance = Math.abs(rect.top - containerRect.top);
			if (distance < closestDistance) {
				closestDistance = distance;
				closestIndex = index;
			}
		}

		return closestIndex;
	}

	function scrollSlideIntoView(container: HTMLElement, slide: SVGElement) {
		const containerRect = container.getBoundingClientRect();
		const slideRect = slide.getBoundingClientRect();
		const nextTop = container.scrollTop + slideRect.top - containerRect.top;
		container.scrollTo({ top: nextTop, behavior: 'smooth' });
	}

	function navigateSlides(direction: 1 | -1): boolean {
		const container = getActiveSlideScrollContainer();
		if (!container) return false;

		const slides = getSlideElements(container);
		if (slides.length === 0) return false;

		const currentIndex = getCurrentSlideIndex(container, slides);
		if (currentIndex < 0) return false;

		const nextIndex = Math.max(0, Math.min(slides.length - 1, currentIndex + direction));
		if (nextIndex === currentIndex) return false;

		scrollSlideIntoView(container, slides[nextIndex]);
		return true;
	}

	function shouldIgnoreSlideShortcut(eventTarget: EventTarget | null): boolean {
		const el = eventTarget instanceof HTMLElement ? eventTarget : null;
		if (!el) return false;
		return !!el.closest(
			'input, textarea, select, button, [contenteditable="true"], [contenteditable=""], a[href]'
		);
	}

	function onSlideKeyDown(event: KeyboardEvent) {
		if (renderType !== 'slide' || shouldIgnoreSlideShortcut(event.target)) return;

		if (event.key === 'PageDown' || event.key === ' ') {
			if (navigateSlides(1)) event.preventDefault();
			return;
		}

		if (event.key === 'PageUp') {
			if (navigateSlides(-1)) event.preventDefault();
			return;
		}
	}

	function onSlideTouchStart(event: TouchEvent) {
		const touch = event.touches[0];
		if (!touch) return;
		touchStartX = touch.clientX;
		touchStartY = touch.clientY;
	}

	function onSlideTouchEnd(event: TouchEvent) {
		if (renderType !== 'slide') return;

		const touch = event.changedTouches[0];
		if (!touch) return;

		const deltaX = touch.clientX - touchStartX;
		const deltaY = touch.clientY - touchStartY;
		const absX = Math.abs(deltaX);
		const absY = Math.abs(deltaY);

		if (absY < 48 || absY <= absX) return;

		if (deltaY < 0) {
			if (navigateSlides(1)) event.preventDefault();
			return;
		}

		if (navigateSlides(-1)) event.preventDefault();
	}

	async function clearSearchQuery(kind: 'topic' | 'paper') {
		if (kind === 'topic') {
			searchTopicQuery = '';
			await tick();
			topicSearchInputEl?.focus();
			return;
		}

		searchPaperQuery = '';
		await tick();
		paperSearchInputEl?.focus();
	}

	let filteredTopics = $derived(
		topics
			.filter(
				(p) =>
					searchTopicQuery.trim() === '' ||
					p.title.toLowerCase().includes(searchTopicQuery.toLowerCase()) ||
					p.keyword.join(' ').toLowerCase().includes(searchTopicQuery.toLowerCase())
			)
			.sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' }))
	);

	function paperYearValue(paper: Paper): number | null {
		const raw = paper.year;
		if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
		const str = String(raw ?? '').trim();
		const match = str.match(/\b(19|20)\d{2}\b/);
		if (!match) return null;
		const year = Number(match[0]);
		return Number.isFinite(year) ? year : null;
	}

	function comparePapersByYearThenTitle(a: Paper, b: Paper) {
		const ay = paperYearValue(a);
		const by = paperYearValue(b);

		if (ay !== null && by !== null && ay !== by) return by - ay; // year desc
		if (ay === null && by !== null) return 1;
		if (ay !== null && by === null) return -1;

		return a.title.localeCompare(b.title, undefined, { sensitivity: 'base' });
	}

	let filteredPapers = $derived(
		papers
			.filter(
				(p) =>
					searchPaperQuery.trim() === '' ||
					p.title.toLowerCase().includes(searchPaperQuery.toLowerCase()) ||
					p.author.toLowerCase().includes(searchPaperQuery.toLowerCase())
			)
			.toSorted(comparePapersByYearThenTitle)
	);

	let paperListItems = $derived.by(() => {
		const items: PaperListItem[] = [];
		let lastYearValue: number | null | undefined;

		for (const paper of filteredPapers) {
			const yearValue = paperYearValue(paper);
			const yearLabel = yearValue === null ? 'Unknown year' : String(yearValue);

			if (yearValue !== lastYearValue) {
				items.push({
					kind: 'year',
					yearLabel,
					yearValue,
					key: `year-${yearLabel}`
				});
				lastYearValue = yearValue;
			}

			items.push({ kind: 'paper', paper, key: `paper-${paper.index}` });
		}

		return items;
	});

	// Preview State
	let renderHtml = $state('');
	let renderHtmlWithHighlights = $state('');
	let renderType = $state<'none' | 'summary' | 'slide'>('none');
	let isLoading = $state(false);
	let appEl: HTMLDivElement | null = null;
	let theme = $state<'light' | 'dark'>('light');
	let readerMode = $state(false);
	let slideWidthPct = $state(100);
	let summaryScalePct = $state(100);
	let showSummaryScalePopover = $state(false);
	let summaryScaleRangeEl = $state<HTMLInputElement | null>(null);
	let summaryScaleButtonEl = $state<HTMLButtonElement | null>(null);
	let summaryScalePopoverEl = $state<HTMLDivElement | null>(null);
	let summaryScalePopoverLeft = $state<string | null>(null);
	let useNotoSerif = $state(true);
	let hasLoadedSummaryScale = $state(false);
	let hasLoadedTheme = $state(false);
	let hasLoadedNotoToggle = $state(false);
	let showSlideWidthPopover = $state(false);
	let slideWidthRangeEl = $state<HTMLInputElement | null>(null);
	let slideWidthButtonEl = $state<HTMLButtonElement | null>(null);
	let slideWidthPopoverEl = $state<HTMLDivElement | null>(null);
	let slideWidthPopoverLeft = $state<string | null>(null);
	let showPromptPopover = $state(false);
	let promptTextareaEl = $state<HTMLTextAreaElement | null>(null);
	let promptButtonEl = $state<HTMLButtonElement | null>(null);
	let promptPopoverEl = $state<HTMLDivElement | null>(null);
	let promptPopoverLeft = $state<string | null>(null);
	let promptPaperText = $state('');
	let promptPaperKey = $state<string | null>(null);
	let isPromptLoading = $state(false);
	let promptError = $state<string | null>(null);
	let promptDraft = $state('');
	let promptCopyStatus = $state<'idle' | 'copied' | 'failed'>('idle');
	let promptCopyResetTimer = 0;

	const PROMPT_TEMPLATE = `# 논문 상세 분석 보고서 작성

## 역할 정의

당신은 컴퓨터 비전 및 딥러닝 분야의 전문가 reviewer이다.
당신은 사용자가 논문 원문에서 추출한 텍스트를 제공받고 논문에 대한 학술 수준의 상세 분석 보고서를 생성한다.

## 논문 읽기 및 분석 규칙

- 한국어 번역으로 인해 정확성이 떨어질 수 있는 경우, 전문 용어는 영어 원문을 그대로 사용하라.
- 제목과 내용이 다르더라도, 섹션 구조를 추론하라.
- 연구 문제, 핵심 아이디어, 그리고 상세한 방법 설명은 필수적이므로 특히 주의 깊게 살펴보라.
- 방법론의 핵심인 방정식, 아키텍처, 손실 함수, 목표 변수, 그리고 학습 절차는 쉬운 한국어로 설명하라.
- 비교 대상, 측정 방법, 그리고 결과의 중요성을 이해할 수 있도록 실험 내용을 충분한 맥락과 함께 설명하라.
- 논문에 명확하게 명시되지 않은 내용은 추측하지 말고 명시적으로 언급하라.

## 보고서 요구 사항

보고서는 원문을 읽지 않고도 내용을 이해할 수 있도록 충분히 상세해야 한다.

### 출력 언어 및 어조

- 한국어로 작성한다.
- 전문 용어는 필요한 경우에만 영어로 작성한다.
- 중국어와 일본어, 인도어 등은 일절 사용하지 않는다.
- 간결한 목록 형식보다는 정확하고 설명적인 서술형 문장을 사용하라.

## 수학 공식의 서식 요건

수식을 작성할 때:

- 본문 내(inline) 수식은 "$...$" 태그로 표시한다("$"를 앞뒤에 하나씩 사용).
- 블록(block) 수식은 "$$...$$" 태그로 표시한다("$"를 앞뒤에 두개씩 사용).

### 필수 구조

- 보고서는 마크다운 형식을 따라 작성해야 하며 보고서의 구성 형식은 다음과 같다.

\`\`\`
<메타데이터 블록>

# <영어 논문 제목>

- **저자**: <authors>
- **발표연도**: <published_year>
- **arXiv**: <arxiv_url>

이후에는 다음의 섹션들을 적절한 순서로 작성한다.

## 1. 논문 개요

포함해야 하는 내용:

- 논문의 목표에 대한 요약
- 연구 문제
- 문제의 중요성

## 2. 핵심 아이디어

포함해야 하는 내용:

- 중심적인 직관 또는 설계 아이디어
- 기존 접근 방식과의 차별점 (논문에서 명확히 제시된 경우)

## 3. 상세 방법 설명

포함해야 하는 내용 (있는 경우):

- 전체 파이프라인 또는 시스템 구조
- 각 주요 구성 요소 및 역할
- 관련성이 있는 경우 훈련 목표, 손실 함수, 추론 절차 또는 알고리즘 흐름
- 주요 방정식 설명

## 4. 실험 및 결과

포함해야 하는 내용 (있는 경우):

- 데이터셋, 작업, 기준선, 지표
- 주요 정량적 또는 정성적 결과
- 실험의 실제 결과

## 5. 강점, 한계

포함해야 하는 내용:

- 논문에서 뒷받침되는 강점
- 한계, 가정 또는 미해결 질문
- 논문에 근거한 간략한 비판적 해석

## 6. 결론

포함해야 하는 내용:

- 논문의 주요 기여 사항에 대한 요약
- 이 연구가 실제 적용이나 향후 연구에 중요한 역할을 할 가능성
\`\`\`

## 메타데이터 블록의 형식

보고서의 맨 첫 줄에는 다음의 스키마를 가진 행을 추가한다.

{"title": "<논문 제목>", "author": "<저자 목록>", "year": <출판 연도>, "url": "<Arxiv URL>", "summary": "<요약 파일 이름>", "slide": ""}

- <Arxiv URL> 값은 논문 텍스트에 해당 정보가 없으면 비워둔다.
- <요약 파일 이름>은 다음과 같이 구성한다.
	- 영어 논문 제목을 영어 소문자로 변환한다.
	- 공백 문자는 "_"(underbar) 문자로 대체한다.
	- 기호 등의 특수문자는 삭제한다.
	- 파일 형식의 확장자로 "".md"를 추가한다.
  - 예시: 논문 제목이 "Attention Is All You Need"인 경우, <요약 파일 이름>은 "attention_is_all_you_need.md"가 된다.

## 분석 보고서 품질 검사

마무리 전:
- 제목 라벨이 일관적인지 확인한다.
- 코드 펜스가 균형 있게 배치되었는지 확인한다.
- 목록 형식이 올바른지 확인한다.
- 수학 구분 기호가 균형 있게 사용되었는지 확인한다.
- 마크다운 lint 오류가 명백한 경우 수정한다.
- 용어가 일관되게 사용되었는지 확인한다.
- 실제로 접근할 수 없는 부분을 읽었다고 주장하지 않는다.

## 논문에서 추출한 텍스트

- 논문에서 추출한 텍스트는 다음과 같다.

---

`;

	const SUMMARY_SCALE_STORAGE_KEY = 'summary-scale-pct-v1';
	const NOTO_TOGGLE_STORAGE_KEY = 'summary-noto-serif-v1';

	let canReader = $derived.by(() => {
		if (!selectedPaper) return false;
		if (isLoading) return false;
		if (!renderHtml) return false;
		if (renderType === 'summary') return !!selectedPaper.summary;
		if (renderType === 'slide') return !!selectedPaper.slide;
		return false;
	});

	let canSlideWidth = $derived.by(() => {
		if (!selectedPaper?.slide) return false;
		if (isLoading) return false;
		if (!renderHtml) return false;
		return renderType === 'slide';
	});

	let canSummaryScale = $derived.by(() => renderType === 'summary');

	let headDescription = $derived.by(() => {
		if (!selectedPaper) return '';
		const author = String(selectedPaper.author ?? '').trim();
		const title = String(selectedPaper.title ?? '').trim();
		const year = String(selectedPaper.year ?? '').trim();
		const parts = [author, title ? `"${title}"` : '', year].filter(Boolean);
		return parts.join(', ');
	});

	let ogType = $derived.by(() =>
		selectedPaper && (renderType === 'summary' || renderType === 'slide') ? 'article' : 'website'
	);

	$effect(() => {
		if (!readerMode) return;
		if (!canReader) readerMode = false;
	});

	$effect(() => {
		if (renderType !== 'summary' && showSummaryScalePopover) {
			showSummaryScalePopover = false;
		}
	});

	$effect(() => {
		if (renderType !== 'slide' && showSlideWidthPopover) {
			showSlideWidthPopover = false;
		}
	});

	$effect(() => {
		if (!readerMode) return;
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.key === 'Escape') readerMode = false;
		};
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	$effect(() => {
		if (typeof window === 'undefined') return;

		window.addEventListener('keydown', onSlideKeyDown);
		return () => window.removeEventListener('keydown', onSlideKeyDown);
	});

	$effect(() => {
		if (typeof window === 'undefined') return;
		if (!document.body) return;
		document.body.setAttribute('data-noto', useNotoSerif ? 'on' : 'off');
		if (!hasLoadedNotoToggle) return;
		try {
			localStorage.setItem(NOTO_TOGGLE_STORAGE_KEY, useNotoSerif ? 'on' : 'off');
		} catch {
			// Do nothing.
		}
	});

	$effect(() => {
		document.documentElement.setAttribute('data-theme', theme);
		if (!hasLoadedTheme) return;
		try {
			localStorage.setItem('app-theme', theme);
		} catch {
			// Do nothing.
		}
	});

	$effect(() => {
		if (typeof window === 'undefined') return;
		if (!hasLoadedSummaryScale) return;
		try {
			localStorage.setItem(SUMMARY_SCALE_STORAGE_KEY, String(summaryScalePct));
		} catch {
			// Do nothing.
		}
	});

	// Renderers
	const md = new MarkdownIt({ html: true, linkify: true, typographer: true });
	const marp = new MarpLite({ html: true });

	const textEncoder = new TextEncoder();
	const textDecoder = new TextDecoder();

	function encodeHex(text: string): string {
		const bytes = textEncoder.encode(text);
		let out = '';
		for (const byte of bytes) {
			out += byte.toString(16).padStart(2, '0');
		}
		return out;
	}

	function decodeHex(text: string): string {
		if (!/^[0-9a-fA-F]*$/.test(text) || text.length % 2 !== 0) {
			return text;
		}
		const bytes = new Uint8Array(text.length / 2);
		for (let i = 0; i < text.length; i += 2) {
			bytes[i / 2] = Number.parseInt(text.slice(i, i + 2), 16);
		}
		return textDecoder.decode(bytes);
	}

	const blockMathRegex = /(?<!\\)\$\$([\s\S]*?)\$\$/g;
	const inlineMathRegex = /(?<!\\)(?<!\$)\$(?!\$)([^$\n]*?)(?<!\\)(?<!\$)\$(?!\$)/g;

	function preProcessMath(text: string): string {
		const blockProcessed = text.replace(blockMathRegex, (_match, content: string) => {
			return `$$${encodeHex(content)}$$`;
		});

		return blockProcessed
			.split('\n')
			.map((line) =>
				line.replace(inlineMathRegex, (_match, content: string) => `$${encodeHex(content)}$`)
			)
			.join('\n');
	}

	function postProcessMath(html: string): string {
		const inlineRestored = html
			.split('\n')
			.map((line) =>
				line
					// ** 먼저 처리
					.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
					// 그 다음 * 처리 (단, ** 제외)
					.replace(/\*(?!\*)(.+?)\*(?!\*)/g, '<em>$1</em>')
					.replace(inlineMathRegex, (_match, content: string) => `$${decodeHex(content)}$`)
			)
			.join('\n');

		return inlineRestored.replace(blockMathRegex, (_match, content: string) => {
			return `$$${decodeHex(content)}$$`;
		});
	}

	function escapeHtml(text: string): string {
		return text
			.replaceAll('&', '&amp;')
			.replaceAll('<', '&lt;')
			.replaceAll('>', '&gt;')
			.replaceAll('"', '&quot;')
			.replaceAll("'", '&#39;');
	}

	function postProcessImage(html: string): string {
		return html.replace(/<img\b[^>]*>/gi, (imgTag) => {
			if (typeof document === 'undefined') return imgTag;

			const wrapper = document.createElement('div');
			wrapper.innerHTML = imgTag;

			const img = wrapper.querySelector('img');
			const alt = img?.getAttribute('alt')?.trim();

			if (!alt) return imgTag;

			const caption = `<p style="text-align:center;"><em>${escapeHtml(alt)}</em></p>`;
			return `${imgTag}${caption}`;
		});
	}

	async function typesetMathJax() {
		if (typeof window === 'undefined') return;
		const mj = (window as unknown as { MathJax?: { typesetPromise?: () => Promise<void> } })
			.MathJax;
		if (!mj?.typesetPromise) return;
		try {
			await tick();
			await mj.typesetPromise();
		} catch (e) {
			console.error('MathJax typeset error:', e);
		}
	}

	function getSummaryContentElements(): HTMLElement[] {
		if (!appEl) return [];
		return Array.from(appEl.querySelectorAll<HTMLElement>('.summary-content'));
	}

	function getSummaryRootFromNode(node: Node | null): HTMLElement | null {
		if (!node) return null;
		for (const root of getSummaryContentElements()) {
			if (root.contains(node)) return root;
		}
		return null;
	}

	function isExcludedHighlightNode(node: Node | null, root: HTMLElement): boolean {
		const parent = node instanceof HTMLElement ? node : node?.parentElement;
		if (!parent) return true;
		if (!root.contains(parent)) return true;
		return !!parent.closest(HIGHLIGHT_EXCLUDED_SELECTOR);
	}

	function findHighlightBlock(node: Node, root: HTMLElement): HTMLElement | null {
		const parent = node instanceof HTMLElement ? node : node.parentElement;
		if (!parent) return null;
		const block = parent.closest<HTMLElement>(HIGHLIGHT_BLOCK_SELECTOR);
		if (!block || !root.contains(block)) return null;
		return block;
	}

	function rangeIntersectsTextNode(range: Range, node: Text): boolean {
		const text = node.textContent ?? '';
		if (!text) return false;
		const nodeRange = document.createRange();
		nodeRange.selectNodeContents(node);
		return (
			range.compareBoundaryPoints(Range.END_TO_START, nodeRange) < 0 &&
			range.compareBoundaryPoints(Range.START_TO_END, nodeRange) > 0
		);
	}

	type FlattenedBlockText = {
		raw: string;
		normalized: string;
		segments: Array<{ node: Text; start: number; end: number }>;
		normalizedToRaw: number[];
	};

	function flattenBlockText(block: HTMLElement): FlattenedBlockText {
		const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, {
			acceptNode: (node) => {
				const text = node.textContent ?? '';
				if (!text) return NodeFilter.FILTER_REJECT;
				if (isExcludedHighlightNode(node, block)) return NodeFilter.FILTER_REJECT;
				return NodeFilter.FILTER_ACCEPT;
			}
		});

		const segments: Array<{ node: Text; start: number; end: number }> = [];
		let raw = '';
		let normalized = '';
		const normalizedToRaw: number[] = [];
		let pendingSpace = false;
		let pendingSpaceRawIndex = -1;
		let textNode = walker.nextNode() as Text | null;

		while (textNode) {
			const text = textNode.textContent ?? '';
			if (text.length > 0) {
				const start = raw.length;
				raw += text;
				segments.push({ node: textNode, start, end: raw.length });

				for (let index = 0; index < text.length; index += 1) {
					const char = text[index];
					const rawIndex = start + index;
					if (/\s/.test(char)) {
						if (normalized.length === 0) continue;
						if (!pendingSpace) pendingSpaceRawIndex = rawIndex;
						pendingSpace = true;
						continue;
					}
					if (pendingSpace) {
						normalized += ' ';
						normalizedToRaw.push(pendingSpaceRawIndex);
						pendingSpace = false;
					}
					normalized += char;
					normalizedToRaw.push(rawIndex);
				}
			}
			textNode = walker.nextNode() as Text | null;
		}

		return { raw, normalized, segments, normalizedToRaw };
	}

	function collectSelectionBlocks(root: HTMLElement, range: Range): string[] {
		const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
			acceptNode: (node) => {
				const text = node.textContent ?? '';
				if (!text) return NodeFilter.FILTER_REJECT;
				if (isExcludedHighlightNode(node, root)) return NodeFilter.FILTER_REJECT;
				if (!rangeIntersectsTextNode(range, node as Text)) return NodeFilter.FILTER_REJECT;
				return NodeFilter.FILTER_ACCEPT;
			}
		});

		const byBlock = new Map<HTMLElement, string>();
		let textNode = walker.nextNode() as Text | null;

		while (textNode) {
			const text = textNode.textContent ?? '';
			let start = 0;
			let end = text.length;

			if (textNode === range.startContainer) start = range.startOffset;
			if (textNode === range.endContainer) end = range.endOffset;

			if (end > start) {
				const block = findHighlightBlock(textNode, root);
				if (block) {
					const existing = byBlock.get(block) ?? '';
					byBlock.set(block, existing + text.slice(start, end));
				}
			}

			textNode = walker.nextNode() as Text | null;
		}

		return Array.from(byBlock.values())
			.map((text) => normalizeHighlightText(text))
			.filter(Boolean);
	}

	type HighlightMatch = {
		rawStart: number;
		fragments: Array<{ node: Text; startOffset: number; endOffset: number }>;
	};

	function findHighlightMatchesInBlock(
		block: HTMLElement,
		targetText: string
	): HighlightMatch[] {
		const normalizedTarget = normalizeHighlightText(targetText);
		if (!normalizedTarget) return [];

		const flattened = flattenBlockText(block);
		if (!flattened.normalized) return [];

		const matches: HighlightMatch[] = [];
		let searchFrom = 0;

		while (searchFrom <= flattened.normalized.length - normalizedTarget.length) {
			const normalizedIndex = flattened.normalized.indexOf(normalizedTarget, searchFrom);
			if (normalizedIndex < 0) break;

			const rawStart = flattened.normalizedToRaw[normalizedIndex];
			const rawEndInclusive =
				flattened.normalizedToRaw[normalizedIndex + normalizedTarget.length - 1];
			const rawEnd = rawEndInclusive + 1;
			const fragments: Array<{ node: Text; startOffset: number; endOffset: number }> = [];

			for (const segment of flattened.segments) {
				if (segment.end <= rawStart) continue;
				if (segment.start >= rawEnd) break;

				const startOffset = Math.max(0, rawStart - segment.start);
				const endOffset = Math.min(segment.end, rawEnd) - segment.start;
				if (endOffset > startOffset) {
					fragments.push({ node: segment.node, startOffset, endOffset });
				}
			}

			if (fragments.length > 0) {
				matches.push({ rawStart, fragments });
			}

			searchFrom = normalizedIndex + normalizedTarget.length;
		}

		return matches;
	}

	function wrapHighlightFragment(
		node: Text,
		startOffset: number,
		endOffset: number,
		entry: SummaryHighlightEntry
	) {
		if (endOffset <= startOffset) return;
		const selected = node.splitText(startOffset);
		selected.splitText(endOffset - startOffset);
		const span = document.createElement('span');
		span.className = `summary-highlight highlight-${entry.color}`;
		span.dataset.highlightGroupId = entry.groupId;
		span.dataset.highlightId = entry.id;
		span.dataset.highlightColor = entry.color;
		selected.parentNode?.insertBefore(span, selected);
		span.appendChild(selected);
	}

	function applyHighlightEntryToContainer(container: ParentNode, entry: SummaryHighlightEntry): number {
		const blocks = Array.from(container.querySelectorAll<HTMLElement>(HIGHLIGHT_BLOCK_SELECTOR));
		let matchCount = 0;

		for (const block of blocks) {
			const matches = findHighlightMatchesInBlock(block, entry.text);
			for (let index = matches.length - 1; index >= 0; index -= 1) {
				const match = matches[index];
				for (let fragmentIndex = match.fragments.length - 1; fragmentIndex >= 0; fragmentIndex -= 1) {
					const fragment = match.fragments[fragmentIndex];
					wrapHighlightFragment(fragment.node, fragment.startOffset, fragment.endOffset, entry);
				}
				matchCount += 1;
			}
		}

		return matchCount;
	}

	function buildHighlightedSummaryHtml(
		baseHtml: string,
		entries: SummaryHighlightEntry[]
	): { html: string; matchedIds: Set<string> } {
		if (typeof document === 'undefined' || !baseHtml || entries.length === 0) {
			return { html: baseHtml, matchedIds: new Set<string>() };
		}

		const template = document.createElement('template');
		template.innerHTML = baseHtml;
		const matchedIds = new Set<string>();

		for (const entry of entries) {
			const matches = applyHighlightEntryToContainer(template.content, entry);
			if (matches > 0) matchedIds.add(entry.id);
		}

		return { html: template.innerHTML, matchedIds };
	}

	function closeHighlightPopover() {
		highlightPopover = null;
	}

	function highlightPopoverVerticalGap() {
		return lastSelectionInteractionMode === 'touch' || lastSelectionInteractionMode === 'pen'
			? 56
			: 12;
	}

	function clampHighlightPopoverPosition(left: number, top: number, width: number, height: number) {
		const padding = 8;
		const halfWidth = width / 2;
		const clampedLeft = Math.max(
			padding + halfWidth,
			Math.min(window.innerWidth - padding - halfWidth, left)
		);
		const clampedTop = Math.max(padding, Math.min(window.innerHeight - padding - height, top));
		return { left: clampedLeft, top: clampedTop };
	}

	function buildHighlightPopoverPosition(anchorLeft: number, anchorTop: number) {
		return {
			left: anchorLeft + highlightPopoverOffset.x,
			top: anchorTop + highlightPopoverOffset.y
		};
	}

	async function positionHighlightPopover() {
		if (!highlightPopoverEl || !highlightPopover) return;
		await tick();
		const rect = highlightPopoverEl.getBoundingClientRect();
		let { left, top } = buildHighlightPopoverPosition(
			highlightPopover.anchorLeft,
			highlightPopover.anchorTop
		);
		const gap = highlightPopoverVerticalGap();

		if (highlightPopoverOffset.y === 0 && top + rect.height > window.innerHeight - 8) {
			top = highlightPopover.anchorTop - rect.height - gap * 2;
		}
		({ left, top } = clampHighlightPopoverPosition(left, top, rect.width, rect.height));

		highlightPopover = { ...highlightPopover, left, top };
	}

	async function openCreateHighlightPopover(blocks: string[], rect: DOMRect) {
		if (blocks.length === 0) {
			closeHighlightPopover();
			return;
		}

		const gap = highlightPopoverVerticalGap();
		const anchorLeft = rect.left + rect.width / 2;
		const anchorTop = rect.bottom + gap;
		const { left, top } = buildHighlightPopoverPosition(anchorLeft, anchorTop);
		highlightPopover = {
			mode: 'create',
			blocks,
			anchorLeft,
			anchorTop,
			left,
			top
		};
		await positionHighlightPopover();
	}

	async function openEditHighlightPopover(groupId: string, rect: DOMRect) {
		const gap = highlightPopoverVerticalGap();
		const anchorLeft = rect.left + rect.width / 2;
		const anchorTop = rect.bottom + gap;
		const { left, top } = buildHighlightPopoverPosition(anchorLeft, anchorTop);
		highlightPopover = {
			mode: 'edit',
			groupId,
			anchorLeft,
			anchorTop,
			left,
			top
		};
		await positionHighlightPopover();
	}

	function applyNewHighlight(color: HighlightColor) {
		if (highlightPopover?.mode !== 'create' || !selectedSummaryDocKey || !highlightPopover.blocks) return;

		const groupId = buildHighlightGroupId();
		const createdAt = Date.now();
		const next = [...summaryHighlights];

		for (const text of highlightPopover.blocks) {
			next.push({
				id: `${groupId}:${next.length}`,
				groupId,
				docPath: selectedSummaryDocKey,
				color,
				text,
				createdAt
			});
		}

		persistSummaryHighlights(next);
		window.getSelection()?.removeAllRanges();
		closeHighlightPopover();
	}

	function updateHighlightGroupColor(groupId: string, color: HighlightColor) {
		const next = summaryHighlights.map((entry) =>
			entry.groupId === groupId ? { ...entry, color } : entry
		);
		persistSummaryHighlights(next);
		closeHighlightPopover();
	}

	function deleteHighlightGroup(groupId: string) {
		persistSummaryHighlights(summaryHighlights.filter((entry) => entry.groupId !== groupId));
		closeHighlightPopover();
	}

	async function syncHighlightPopoverFromSelection(root: HTMLElement | null) {
		if (renderType !== 'summary' || !root) return;

		await tick();
		const selection = window.getSelection();
		if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
			if (highlightPopover?.mode === 'create') closeHighlightPopover();
			return;
		}

		const range = selection.getRangeAt(0);
		if (!root.contains(range.commonAncestorContainer)) {
			if (highlightPopover?.mode === 'create') closeHighlightPopover();
			return;
		}

		const blocks = collectSelectionBlocks(root, range);
		if (blocks.length === 0) {
			if (highlightPopover?.mode === 'create') closeHighlightPopover();
			return;
		}

		await openCreateHighlightPopover(blocks, range.getBoundingClientRect());
	}

	async function handleSummarySelectionChange(event: MouseEvent | KeyboardEvent) {
		const root = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
		lastSelectionInteractionMode = event instanceof KeyboardEvent ? 'keyboard' : 'mouse';
		await syncHighlightPopoverFromSelection(root);
	}

	function handleSummaryClick(event: MouseEvent) {
		const target = event.target instanceof HTMLElement ? event.target : null;
		const highlight = target?.closest<HTMLElement>('.summary-highlight');
		if (!highlight) return;

		event.preventDefault();
		event.stopPropagation();
		window.getSelection()?.removeAllRanges();

		const groupId = highlight.dataset.highlightGroupId;
		if (!groupId) return;
		void openEditHighlightPopover(groupId, highlight.getBoundingClientRect());
	}

	function onHighlightPopoverDragStart(event: PointerEvent) {
		if (!highlightPopover || !highlightPopoverEl) return;
		if (event.pointerType === 'mouse' && event.button !== 0) return;

		const rect = highlightPopoverEl.getBoundingClientRect();
		const width = rect.width;
		const height = rect.height;
		const offsetX = event.clientX - rect.left;
		const offsetY = event.clientY - rect.top;

		event.preventDefault();
		event.stopPropagation();
		highlightPopoverEl.setPointerCapture?.(event.pointerId);

		const move = (moveEvent: PointerEvent) => {
			if (!highlightPopover) return;
			if (moveEvent.pointerId !== event.pointerId) return;

			const halfWidth = width / 2;
			const nextCenterX = moveEvent.clientX - offsetX + halfWidth;
			const nextTop = moveEvent.clientY - offsetY;
			const { left, top } = clampHighlightPopoverPosition(nextCenterX, nextTop, width, height);

			highlightPopover = {
				...highlightPopover,
				left,
				top
			};
		};

		const up = (upEvent: PointerEvent) => {
			if (upEvent.pointerId !== event.pointerId) return;
			if (highlightPopover) {
				highlightPopoverOffset = {
					x: highlightPopover.left - highlightPopover.anchorLeft,
					y: highlightPopover.top - highlightPopover.anchorTop
				};
			}
			highlightPopoverEl?.releasePointerCapture?.(event.pointerId);
			window.removeEventListener('pointermove', move);
			window.removeEventListener('pointerup', up);
			window.removeEventListener('pointercancel', up);
		};

		window.addEventListener('pointermove', move);
		window.addEventListener('pointerup', up);
		window.addEventListener('pointercancel', up);
	}

	function rebuildRenderedSummaryHtml() {
		if (renderType !== 'summary') {
			renderHtmlWithHighlights = renderHtml;
			closeHighlightPopover();
			return;
		}

		const entries = currentSummaryHighlights(renderedSummaryDocKey);
		if (entries.length === 0) {
			renderHtmlWithHighlights = renderHtml;
			return;
		}

		const { html, matchedIds } = buildHighlightedSummaryHtml(renderHtml, entries);
		renderHtmlWithHighlights = html;

		if (matchedIds.size !== entries.length) {
			const failedIds = new Set(
				entries.filter((entry) => !matchedIds.has(entry.id)).map((entry) => entry.id)
			);
			if (failedIds.size > 0) {
				persistSummaryHighlights(
					summaryHighlights.filter((entry) => !failedIds.has(entry.id))
				);
			}
		}
	}

	$effect(() => {
		renderHtml;
		summaryHighlights;
		renderedSummaryDocKey;
		renderType;
		rebuildRenderedSummaryHtml();
	});

	$effect(() => {
		const htmlForDisplay = renderType === 'summary' ? renderHtmlWithHighlights : renderHtml;
		if (!htmlForDisplay) return;
		void typesetMathJax();
	});

	onMount(() => {
		const MOBILE_MEDIA_QUERY = '(max-width: 960px)';
		let selectionSyncTimer = 0;
		let isSummarySelectionPointerActive = false;
		let hasPendingSelectionSync = false;

		const ensureMobileStageValid = () => {
			if (!isMobile) return;
			if (!selectedTopic) {
				mobileStage = 0;
				return;
			}
			if (mobileStage === 2 && !selectedPaper) mobileStage = 1;
		};

		const mql = window.matchMedia(MOBILE_MEDIA_QUERY);
		let prevMatches = mql.matches;

		const applyMobileMode = (matches: boolean, enteringMobile: boolean) => {
			isMobile = matches;
			if (!matches) return;

			// Desktop-only panel toggles shouldn't affect mobile stack navigation.
			showTopicPanel = true;
			showPaperPanel = true;

			if (enteringMobile) {
				mobileStage = !selectedTopic ? 0 : !selectedPaper ? 1 : 2;
			}

			ensureMobileStageValid();
		};

		const onMediaChange = (event: MediaQueryListEvent) => {
			const enteringMobile = !prevMatches && event.matches;
			prevMatches = event.matches;
			applyMobileMode(event.matches, enteringMobile);
		};

		applyMobileMode(mql.matches, mql.matches);
		mql.addEventListener('change', onMediaChange);
		isLocalhost =
			window.location.hostname === 'localhost' ||
			window.location.hostname === '127.0.0.1' ||
			window.location.hostname === '0.0.0.0';

		const onGlobalPointerDown = (event: PointerEvent | MouseEvent) => {
			const target = event.target instanceof Node ? event.target : null;
			if (target && highlightPopoverEl?.contains(target)) return;
			closeHighlightPopover();
		};

		const scheduleSelectionSync = (delay = 0) => {
			if (selectionSyncTimer) window.clearTimeout(selectionSyncTimer);
			selectionSyncTimer = window.setTimeout(() => {
				selectionSyncTimer = 0;
				const selection = window.getSelection();
				const root = getSummaryRootFromNode(selection?.rangeCount ? selection.getRangeAt(0).commonAncestorContainer : null);
				void syncHighlightPopoverFromSelection(root);
			}, delay);
		};

		const onSelectionChange = () => {
			if (renderType !== 'summary') return;
			if (isSummarySelectionPointerActive) {
				hasPendingSelectionSync = true;
				if (highlightPopover?.mode === 'create') closeHighlightPopover();
				return;
			}
			const delay =
				lastSelectionInteractionMode === 'touch' || lastSelectionInteractionMode === 'pen'
					? 120
					: 0;
			scheduleSelectionSync(delay);
		};

		const onDocumentPointerDown = (event: PointerEvent) => {
			lastSelectionInteractionMode =
				event.pointerType === 'touch' || event.pointerType === 'pen' ? event.pointerType : 'mouse';
			const target = event.target instanceof Node ? event.target : null;
			isSummarySelectionPointerActive = !!(
				target &&
				!highlightPopoverEl?.contains(target) &&
				getSummaryRootFromNode(target)
			);
			hasPendingSelectionSync = false;
			onGlobalPointerDown(event);
		};

		const onDocumentPointerUp = (event: PointerEvent) => {
			lastSelectionInteractionMode =
				event.pointerType === 'touch' || event.pointerType === 'pen' ? event.pointerType : 'mouse';
			const shouldSync = isSummarySelectionPointerActive || hasPendingSelectionSync;
			isSummarySelectionPointerActive = false;
			hasPendingSelectionSync = false;
			if (renderType === 'summary' && shouldSync) {
				scheduleSelectionSync(event.pointerType === 'mouse' ? 0 : 120);
			}
		};

		const onDocumentTouchEnd = () => {
			lastSelectionInteractionMode = 'touch';
			const shouldSync = isSummarySelectionPointerActive || hasPendingSelectionSync;
			isSummarySelectionPointerActive = false;
			hasPendingSelectionSync = false;
			if (renderType === 'summary' && shouldSync) scheduleSelectionSync(120);
		};

		const onDocumentPointerCancel = () => {
			isSummarySelectionPointerActive = false;
			hasPendingSelectionSync = false;
		};

		const onDocumentKeyUp = () => {
			lastSelectionInteractionMode = 'keyboard';
			if (renderType === 'summary') scheduleSelectionSync(0);
		};

		const onViewportResize = () => {
			if (!highlightPopover) return;
			void positionHighlightPopover();
		};

		const onViewportScroll = () => closeHighlightPopover();

		document.addEventListener('pointerdown', onDocumentPointerDown, true);
		document.addEventListener('pointerup', onDocumentPointerUp, true);
		document.addEventListener('pointercancel', onDocumentPointerCancel, true);
		document.addEventListener('touchend', onDocumentTouchEnd, true);
		document.addEventListener('selectionchange', onSelectionChange);
		document.addEventListener('keyup', onDocumentKeyUp, true);
		window.addEventListener('resize', onViewportResize);
		window.addEventListener('scroll', onViewportScroll, true);

		void (async () => {
			try {
				const saved = localStorage.getItem('app-theme');
				if (saved === 'dark' || saved === 'light') theme = saved;
				const savedScale = localStorage.getItem(SUMMARY_SCALE_STORAGE_KEY);
				const parsedScale = savedScale ? Number(savedScale) : NaN;
				if (Number.isFinite(parsedScale)) {
					summaryScalePct = Math.min(200, Math.max(100, Math.round(parsedScale)));
				}
				hasLoadedSummaryScale = true;
				const savedNoto = localStorage.getItem(NOTO_TOGGLE_STORAGE_KEY);
				if (savedNoto === 'on' || savedNoto === 'off') {
					useNotoSerif = savedNoto === 'on';
				}
				hasLoadedNotoToggle = true;
				paperTags = loadPaperTags();
				summaryHighlights = loadSummaryHighlights();
			} catch {
				// Do nothing.
			}
			hasLoadedTheme = true;

			try {
				const res = await fetch('/docs/manifest.json');

				if (res.ok) {
					topics = await res.json();
					// Sort topics by title
					topics.sort((a, b) => (a.title || a.id).localeCompare(b.title || b.id));
				} else {
					console.error('Failed to load manifest.json');
				}
			} catch (e) {
				console.error('Error fetching manifest:', e);
			}
		})();

		return () => {
			if (promptCopyResetTimer) {
				window.clearTimeout(promptCopyResetTimer);
				promptCopyResetTimer = 0;
			}
			if (selectionSyncTimer) window.clearTimeout(selectionSyncTimer);
			mql.removeEventListener('change', onMediaChange);
			document.removeEventListener('pointerdown', onDocumentPointerDown, true);
			document.removeEventListener('pointerup', onDocumentPointerUp, true);
			document.removeEventListener('pointercancel', onDocumentPointerCancel, true);
			document.removeEventListener('touchend', onDocumentTouchEnd, true);
			document.removeEventListener('selectionchange', onSelectionChange);
			document.removeEventListener('keyup', onDocumentKeyUp, true);
			window.removeEventListener('resize', onViewportResize);
			window.removeEventListener('scroll', onViewportScroll, true);
		};
	});

	async function toggleSummaryScalePopover() {
		if (showSummaryScalePopover || !canSummaryScale) return;
		showSummaryScalePopover = true;
		await tick();
		positionSummaryScalePopover();
		summaryScaleRangeEl?.focus();
	}

	function closeSummaryScalePopoverOnFocusOut(event: FocusEvent) {
		const container = event.currentTarget as HTMLElement | null;
		if (!container) {
			showSummaryScalePopover = false;
			return;
		}
		requestAnimationFrame(() => {
			const active = document.activeElement;
			if (active && container.contains(active)) return;
			showSummaryScalePopover = false;
		});
	}

	function setSummaryScale(next: number) {
		summaryScalePct = Math.min(200, Math.max(100, Math.round(next)));
	}

	function positionSummaryScalePopover() {
		if (typeof window === 'undefined') return;
		if (!summaryScalePopoverEl || !summaryScaleButtonEl) return;

		const popoverRect = summaryScalePopoverEl.getBoundingClientRect();
		const buttonRect = summaryScaleButtonEl.getBoundingClientRect();
		const wrapRect = summaryScalePopoverEl.parentElement?.getBoundingClientRect();
		if (!wrapRect) return;

		const padding = 8;
		const viewportWidth = window.innerWidth;
		let left = buttonRect.left;

		if (left < padding) left = padding;
		if (left + popoverRect.width > viewportWidth - padding) {
			left = Math.max(padding, viewportWidth - padding - popoverRect.width);
		}

		summaryScalePopoverLeft = `${left - wrapRect.left}px`;
	}

	$effect(() => {
		if (typeof window === 'undefined') return;
		if (!showSummaryScalePopover) return;

		const onResize = () => positionSummaryScalePopover();
		window.addEventListener('resize', onResize);
		return () => window.removeEventListener('resize', onResize);
	});

	async function toggleSlideWidthPopover() {
		if (showSlideWidthPopover || !canSlideWidth) return;
		showSlideWidthPopover = true;
		await tick();
		positionSlideWidthPopover();
		slideWidthRangeEl?.focus();
	}

	function closeSlideWidthPopoverOnFocusOut(event: FocusEvent) {
		const container = event.currentTarget as HTMLElement | null;
		if (!container) {
			showSlideWidthPopover = false;
			return;
		}
		requestAnimationFrame(() => {
			const active = document.activeElement;
			if (active && container.contains(active)) return;
			showSlideWidthPopover = false;
		});
	}

	function positionSlideWidthPopover() {
		if (typeof window === 'undefined') return;
		if (!slideWidthPopoverEl || !slideWidthButtonEl) return;

		const popoverRect = slideWidthPopoverEl.getBoundingClientRect();
		const buttonRect = slideWidthButtonEl.getBoundingClientRect();
		const wrapRect = slideWidthPopoverEl.parentElement?.getBoundingClientRect();
		if (!wrapRect) return;

		const padding = 8;
		const viewportWidth = window.innerWidth;
		let left = buttonRect.left;

		if (left < padding) left = padding;
		if (left + popoverRect.width > viewportWidth - padding) {
			left = Math.max(padding, viewportWidth - padding - popoverRect.width);
		}

		slideWidthPopoverLeft = `${left - wrapRect.left}px`;
	}

	$effect(() => {
		if (typeof window === 'undefined') return;
		if (!showSlideWidthPopover) return;

		const onResize = () => positionSlideWidthPopover();
		window.addEventListener('resize', onResize);
		return () => window.removeEventListener('resize', onResize);
	});

	$effect(() => {
		if (!selectedPaperTagKey) {
			editingTagKey = null;
			tagDraft = '';
			return;
		}
		if (editingTagKey === selectedPaperTagKey) return;
		editingTagKey = null;
		tagDraft = '';
	});

	$effect(() => {
		// Disallow more than /{topic}/{paper}.
		if (routeHasExtra) {
			void goto(resolve('/'), { replaceState: true, noScroll: true, keepFocus: true });
			return;
		}

		const topicId = routeTopicId;
		const paperSlug = routePaperSlug;
		const nextKey = `${topicId ?? ''}/${paperSlug ?? ''}`;

		if (nextKey === appliedRouteKey) return;

		// If route references a topic but manifest not loaded yet, wait.
		if ((topicId || paperSlug) && topics.length === 0) return;

		appliedRouteKey = nextKey;
		const seq = ++routeApplySeq;

		void (async () => {
			if (seq !== routeApplySeq) return;

			// Root: clear selection.
			if (!topicId) {
				selectedTopic = null;
				selectedPaper = null;
				papers = [];
				renderType = 'none';
				renderHtml = '';
				if (isMobile) mobileStage = 0;
				return;
			}

			const topic = topics.find((t) => t.id === topicId) ?? null;
			if (!topic) {
				void goto(resolve('/'), { replaceState: true, noScroll: true, keepFocus: true });
				return;
			}

			// Ensure topic is selected and papers are loaded.
			if (selectedTopic?.id !== topic.id || papers.length === 0) {
				await selectTopic(topic, { syncUrl: false });
			}

			// /{topic}: no paper selected.
			if (!paperSlug) {
				selectedPaper = null;
				renderType = 'none';
				renderHtml = '';
				if (isMobile) mobileStage = 1;
				return;
			}

			const match = findPaperBySlug(paperSlug, papers);
			if (!match) {
				void goto(resolve(`/${topic.id}`), { replaceState: true, noScroll: true, keepFocus: true });
				return;
			}

			await loadPreview(match.paper, match.type, { syncUrl: false });
		})();
	});

	function setColumnVar(name: '--col1' | '--col2', value: number) {
		document.documentElement.style.setProperty(name, `${value}px`);
	}

	function applyResize(which: 'left' | 'middle', clientX: number) {
		if (!appEl) return;
		if (window.innerWidth <= 960) return;
		if (which === 'left' && !showTopicPanel) return;
		if (which === 'middle' && !showPaperPanel) return;

		const min1 = 180;
		const min2 = 240;
		const min3 = 320;
		const total = appEl.clientWidth;
		const styles = getComputedStyle(document.documentElement);
		const currentCol1 = parseFloat(styles.getPropertyValue('--col1')) || 250;

		if (which === 'left') {
			const newCol1 = Math.max(min1, Math.min(clientX, total - min2 - min3 - 12));
			setColumnVar('--col1', newCol1);
		} else {
			const leftEdge = currentCol1 + 6;
			const newCol2 = Math.max(min2, Math.min(clientX - leftEdge, total - currentCol1 - min3 - 12));
			setColumnVar('--col2', newCol2);
		}
	}

	function onSplitterDown(which: 'left' | 'middle', event: PointerEvent) {
		event.preventDefault();
		const move = (moveEvent: PointerEvent) => applyResize(which, moveEvent.clientX);
		const up = () => {
			window.removeEventListener('pointermove', move);
			window.removeEventListener('pointerup', up);
		};
		window.addEventListener('pointermove', move);
		window.addEventListener('pointerup', up);
	}

	function paperTone(paper: Paper) {
		if (paper.slide) return 'paper-pink';
		if (paper.summary) return 'paper-blue';
		return 'paper-gray';
	}

	function buildArxivUrl(url: string | undefined) {
		if (!url) return '';
		if (url.includes('/abs/')) return url.replace(/^http:\/\//, 'https://');
		if (url.includes('/pdf/')) {
			return url
				.replace(/^http:\/\//, 'https://')
				.replace('/pdf/', '/abs/')
				.replace(/\.pdf$/i, '')
				.replace(/v\d+$/i, (match) => match);
		}
		return '';
	}

	function buildAr5ivUrl(url: string | undefined) {
		if (!url) return '';
		const regex = /arxiv\.org\/(abs|pdf)\/(\d+\.\d+(v\d+)?)/;
		const match = url.match(regex);

		if (match) {
			const arxivId = match[2];
			return `https://ar5iv.labs.arxiv.org/html/${arxivId}`;
		}

		return '';
	}

	function buildPdfUrl(url: string | undefined) {
		if (!url) return '';
		if (url.includes('/pdf/')) return url.replace(/^http:\/\//, 'https://');
		if (url.includes('/abs/')) {
			return url.replace(/^http:\/\//, 'https://').replace('/abs/', '/pdf/') + '.pdf';
		}
		return url;
	}

	function buildPromptPaperKey(paper: Paper | null): string | null {
		if (!paper?.url) return null;
		const arxiv = buildArxivUrl(paper.url);
		if (arxiv) return arxiv;
		if (paper.summary) return paper.summary;
		if (paper.slide) return paper.slide;
		return paper.title;
	}

	async function extractTextFromPdf(url: string): Promise<string> {
		const pdfjs = await loadPdfjs();
		if (!pdfjs) throw new Error('PDF parser unavailable.');

		const res = await fetchWithProxy(url);
		if (!res.ok) throw new Error('Failed to fetch PDF.');
		const buffer = await res.arrayBuffer();

		const doc = await pdfjs.getDocument({ data: buffer }).promise;
		const totalPages = doc.numPages ?? 0;
		const chunks: string[] = [];

		for (let pageNum = 1; pageNum <= totalPages; pageNum += 1) {
			const page = await doc.getPage(pageNum);
			const content = await page.getTextContent();
			const text = content.items
				.map((item) => ('str' in item ? String(item.str) : ''))
				.filter(Boolean)
				.join(' ');
			if (text) chunks.push(text);
		}

		return chunks.join('\n');
	}

	async function fetchWithProxy(url: string): Promise<Response> {
		try {
			const direct = await fetch(url);
			if (direct.ok) return direct;
			throw new Error('Direct fetch failed.');
		} catch {
			if (!isLocalhost) throw new Error('Fetch blocked by CORS.');
			const proxyUrl = `/api/dev-proxy?url=${encodeURIComponent(url)}`;
			const proxied = await fetch(proxyUrl);
			if (!proxied.ok) throw new Error('Proxy fetch failed.');
			return proxied;
		}
	}

	async function fetchPromptPaperText(paper: Paper) {
		if (!paper.url) return '';
		const ar5ivUrl = buildAr5ivUrl(paper.url);
		const pdfUrl = buildPdfUrl(paper.url);

		if (ar5ivUrl) {
			try {
				const res = await fetchWithProxy(ar5ivUrl);
				if (res.ok) {
					const html = await res.text();
					const noScript = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
					const noTags = noScript.replace(/<[^>]+>/g, '');
					const noTrailing = noTags.replace(/[ \t]+$/gm, '');
					const collapsed = noTrailing.replace(/\n{3,}/g, '\n\n');
					const text = collapsed.trim();

					if (text.length > 1000) {
						return text;
					}
				}
			} catch {
				// Fall back to PDF extraction.
			}
		}

		if (pdfUrl) {
			try {
				const pdfText = await extractTextFromPdf(pdfUrl);
				const noTrailing = pdfText.replace(/[ \t]+$/gm, '');
				const collapsed = noTrailing.replace(/\n{3,}/g, '\n\n');
				return collapsed.trim();
			} catch {
				// Fall back.
			}
		}

		return '';
	}

	async function openPromptPopover() {
		if (showPromptPopover) return;
		showPromptPopover = true;
		promptError = null;
		promptCopyStatus = 'idle';
		await tick();
		positionPromptPopover();
		promptTextareaEl?.focus();
		await refreshPromptPaperText();
	}

	function closePromptPopoverOnFocusOut(event: FocusEvent) {
		const container = event.currentTarget as HTMLElement | null;
		if (!container) {
			showPromptPopover = false;
			return;
		}
		requestAnimationFrame(() => {
			const active = document.activeElement;
			if (active && container.contains(active)) return;
			showPromptPopover = false;
		});
	}

	async function refreshPromptPaperText() {
		if (!selectedPaper) return;
		const key = buildPromptPaperKey(selectedPaper);
		if (!key) return;

		if (promptPaperKey === key && promptPaperText) return;

		isPromptLoading = true;
		promptError = null;

		try {
			promptDraft = '';
			const text = await fetchPromptPaperText(selectedPaper);
			promptPaperText = text;
			promptPaperKey = key;
			promptDraft = `${PROMPT_TEMPLATE}\n\n${text}`;

			if (!text) promptError = 'Failed to extract paper text.';
		} catch (err) {
			promptError = err instanceof Error ? err.message : 'Failed to extract paper text.';
		} finally {
			isPromptLoading = false;
		}
	}

	function positionPromptPopover() {
		if (typeof window === 'undefined') return;
		if (!promptPopoverEl || !promptButtonEl) return;

		const popoverRect = promptPopoverEl.getBoundingClientRect();
		const buttonRect = promptButtonEl.getBoundingClientRect();
		const wrapRect = promptPopoverEl.parentElement?.getBoundingClientRect();
		if (!wrapRect) return;

		const padding = 8;
		const viewportWidth = window.innerWidth;
		let left = buttonRect.left;

		if (left < padding) left = padding;
		if (left + popoverRect.width > viewportWidth - padding) {
			left = Math.max(padding, viewportWidth - padding - popoverRect.width);
		}

		promptPopoverLeft = `${left - wrapRect.left}px`;
	}

	async function copyPromptToClipboard() {
		if (!promptDraft) return;
		if (promptCopyResetTimer) {
			window.clearTimeout(promptCopyResetTimer);
			promptCopyResetTimer = 0;
		}
		try {
			await navigator.clipboard.writeText(promptDraft);
			promptCopyStatus = 'copied';
			promptCopyResetTimer = window.setTimeout(() => {
				promptCopyStatus = 'idle';
				promptCopyResetTimer = 0;
			}, 1000);
		} catch {
			promptCopyStatus = 'failed';
		}
	}

	$effect(() => {
		if (typeof window === 'undefined') return;
		if (!showPromptPopover) return;

		const onResize = () => positionPromptPopover();
		window.addEventListener('resize', onResize);
		return () => window.removeEventListener('resize', onResize);
	});

	function selectPaper(paper: Paper) {
		selectedPaper = paper;
		renderType = 'none';
		renderHtml = '';
		renderedSummaryDocKey = null;

		if (paper.summary) {
			loadPreview(paper, 'summary', { syncUrl: true });
		} else if (paper.slide) {
			loadPreview(paper, 'slide', { syncUrl: true });
		}

		if (isMobile) mobileStage = 2;
	}

	function openExternal(kind: 'arxiv' | 'ar5iv' | 'pdf') {
		if (!selectedPaper?.url) return;

		let targetUrl;

		if (kind === 'arxiv') targetUrl = buildArxivUrl(selectedPaper.url);
		else if (kind === 'ar5iv') targetUrl = buildAr5ivUrl(selectedPaper.url);
		else if (kind === 'pdf') targetUrl = buildPdfUrl(selectedPaper.url);

		if (!targetUrl) return;

		window.open(targetUrl, '_blank', 'noopener,noreferrer');
	}

	async function selectTopic(topic: Topic, options?: { syncUrl?: boolean }) {
		const syncUrl = options?.syncUrl ?? true;
		selectedTopic = topic;
		selectedPaper = null;
		renderType = 'none';
		renderHtml = '';
		renderedSummaryDocKey = null;
		papers = [];
		isLoading = true;

		if (isMobile) mobileStage = 1;

		if (syncUrl) {
			appliedRouteKey = `${topic.id}/`;
			void goto(resolve(`/${topic.id}`), { noScroll: true, keepFocus: true });
		}

		try {
			const res = await fetch(`/docs/${topic.id}/paper_list.jsonl`);
			if (res.ok) {
				const text = await res.text();
				papers = text
					.split('\n')
					.filter((line) => line.trim())
					.map((line, index) => {
						try {
							return { ...JSON.parse(line), index };
						} catch {
							return null;
						}
					})
					.filter((p): p is Paper => Boolean(p)); // null???�거??
			}
		} catch (e) {
			console.error('Error fetching paper list:', e);
		} finally {
			isLoading = false;
		}
	}

	async function loadPreview(
		paper: Paper,
		type: 'summary' | 'slide',
		options?: { syncUrl?: boolean }
	) {
		const syncUrl = options?.syncUrl ?? false;
		if (!selectedTopic) return;

		selectedPaper = paper;
		isLoading = true;

		if (isMobile) mobileStage = 2;

		const filename = type === 'summary' ? paper.summary : paper.slide;
		let nextRenderType: 'summary' | 'slide' | 'none' = type;
		let nextRenderHtml = renderHtml;
		let nextRenderedSummaryDocKey: string | null =
			type === 'summary' ? buildPaperTagKey(selectedTopic.id, paper) : null;
		if (!filename) {
			nextRenderType = 'none';
			nextRenderHtml = '<p>No file specified.</p>';
			renderType = nextRenderType;
			renderHtml = nextRenderHtml;
			renderedSummaryDocKey = null;
			isLoading = false;
			return;
		}

		if (syncUrl) {
			const slug = paperSlugFromFilename(filename);
			appliedRouteKey = `${selectedTopic.id}/${slug}`;
			void goto(resolve(`/${selectedTopic.id}/${slug}`), { noScroll: true, keepFocus: true });
		}

		try {
			const res = await fetch(`/docs/${selectedTopic.id}/${filename}`);
			if (res.ok) {
				const text = await res.text();
				const processed = preProcessMath(text);
				if (type === 'summary') {
					nextRenderHtml = postProcessImage(postProcessMath(md.render(processed)));
				} else if (type === 'slide') {
					const { html, css } = marp.render(processed);
					const processedHtml = postProcessMath(html);
					nextRenderHtml = `<style>${css}</style><div class="marp-slide-wrapper">${processedHtml}</div>`;
				}
			} else {
				nextRenderHtml = `<p>File not found: ${filename}</p>`;
			}
		} catch (e) {
			if (e instanceof Error) {
				nextRenderHtml = `<p>Error loading file: ${e.message}</p>`;
			} else {
				nextRenderHtml = `<p>Unknown error</p>`;
			}
		} finally {
			renderType = nextRenderType;
			renderHtml = nextRenderHtml;
			renderedSummaryDocKey = nextRenderType === 'summary' ? nextRenderedSummaryDocKey : null;
			isLoading = false;
		}
	}
</script>

<svelte:head>
	<title>Research-with-AI</title>
	{#if headDescription}
		<meta name="description" content={headDescription} />
		<meta property="og:description" content={headDescription} />
	{/if}
	<meta property="og:title" content="Research-with-AI" />
	<meta property="og:type" content={ogType} />
	<script>
		(() => {
			try {
				const saved = localStorage.getItem('app-theme');
				if (saved === 'dark' || saved === 'light') {
					document.documentElement.setAttribute('data-theme', saved);
				}
			} catch {
				// Do nothing.
			}
		})();
	</script>
	<script>
		window.MathJax = {
			tex: {
				inlineMath: [
					['$', '$'],
					['\\(', '\\)']
				],
				displayMath: [
					['$$', '$$'],
					['\\[', '\\]']
				],
				packages: { '[+]': ['ams'] }
			},
			svg: {
				fontCache: 'local'
			},
			options: {
				skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
			}
		};
	</script>
	<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</svelte:head>

<div
	class="app-container"
	data-show-topic={showTopicPanel}
	data-show-paper={showPaperPanel}
	data-mobile-stage={mobileStage}
	data-reader={readerMode}
	style:--mobile-stage={mobileStage}
	style:--slide-max-width={`${slideWidthPct}vw`}
	bind:this={appEl}
>
	<!-- 1. Topic List -->
	<div class="topic-list">
		<div class="topic-list-header">
			<h2>
				<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 576 512"
					><path
						fill="currentColor"
						d="M97.5 400l50-160 379.4 0-50 160-379.4 0zm190.7 48L477 448c21 0 39.6-13.6 45.8-33.7l50-160c9.7-30.9-13.4-62.3-45.8-62.3l-379.4 0c-21 0-39.6 13.6-45.8 33.7L80.2 294.4 80.2 96c0-8.8 7.2-16 16-16l138.7 0c3.5 0 6.8 1.1 9.6 3.2L282.9 112c13.8 10.4 30.7 16 48 16l117.3 0c8.8 0 16 7.2 16 16l48 0c0-35.3-28.7-64-64-64L330.9 80c-6.9 0-13.7-2.2-19.2-6.4L273.3 44.8C262.2 36.5 248.8 32 234.9 32L96.2 32c-35.3 0-64 28.7-64 64l0 288c0 35.3 28.7 64 64 64l192 0z"
					/></svg
				>
				Topics
			</h2>
			<div class="search-container" onfocusout={(e) => closeSearchPopoverOnFocusOut(e, 'topic')}>
				<button
					class="search-button"
					title="Search topics"
					onclick={() => openSearchPopover('topic')}
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 512 512">
						<path
							fill="currentColor"
							d="M416 208c0 45.9-14.9 88.3-40 122.7L500 457.3c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 378c-34.4 25.1-76.8 40-122.7 40C93.1 418 0 324.9 0 208S93.1 0 208 0s208 93.1 208 208zM208 352c79.5 0 144-64.5 144-144S287.5 64 208 64 64 128.5 64 208s64.5 144 144 144z"
						/>
					</svg>
				</button>
				{#if showSearchTopicInput}
					<div class="search-popover">
						<div class="search-input">
							<input
								type="text"
								bind:value={searchTopicQuery}
								placeholder="Filter by keywords"
								bind:this={topicSearchInputEl}
							/>
							{#if searchTopicQuery.trim() !== ''}
								<button
									type="button"
									class="search-clear"
									aria-label="Clear topic search"
									onclick={() => clearSearchQuery('topic')}
								>
									×
								</button>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		</div>
		<div class="topic-content">
			<ul>
				{#each filteredTopics as topic (topic.id)}
					<li>
						<button
							class:active={selectedTopic?.id === topic.id}
							onclick={() => selectTopic(topic)}
						>
							{topic.title || topic.id}
						</button>
					</li>
				{/each}
			</ul>
		</div>
	</div>

	<button
		class="splitter left-splitter"
		aria-label="Left splitter"
		onpointerdown={(event) => onSplitterDown('left', event)}
	></button>

	<!-- 2. Paper List -->
	<div class="paper-list">
		<div class="paper-list-header">
			{#if isMobile}
				<button
					type="button"
					class="nav-back"
					aria-label="Back to topics"
					onclick={() => (mobileStage = 0)}
				>
					<span aria-hidden="true">←</span>
					<span class="nav-back-label">Topics</span>
				</button>
			{/if}
			<h2>
				<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512"
					><path
						fill="currentColor"
						d="M40 48C26.7 48 16 58.7 16 72l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24L40 48zM192 64c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L192 64zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zM16 232l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0c-13.3 0-24 10.7-24 24zM40 368c-13.3 0-24 10.7-24 24l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0z"
					/></svg
				>
				Papers {selectedTopic ? `(${filteredPapers.length})` : ''}
			</h2>
			<div class="search-container" onfocusout={(e) => closeSearchPopoverOnFocusOut(e, 'paper')}>
				<button
					class="search-button"
					title="Search papers"
					onclick={() => openSearchPopover('paper')}
				>
					<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 512 512">
						<path
							fill="currentColor"
							d="M416 208c0 45.9-14.9 88.3-40 122.7L500 457.3c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 378c-34.4 25.1-76.8 40-122.7 40C93.1 418 0 324.9 0 208S93.1 0 208 0s208 93.1 208 208zM208 352c79.5 0 144-64.5 144-144S287.5 64 208 64 64 128.5 64 208s64.5 144 144 144z"
						/>
					</svg>
				</button>
				{#if showSearchPaperInput}
					<div class="search-popover">
						<div class="search-input">
							<input
								type="text"
								bind:value={searchPaperQuery}
								placeholder="Filter by title or author"
								bind:this={paperSearchInputEl}
							/>
							{#if searchPaperQuery.trim() !== ''}
								<button
									type="button"
									class="search-clear"
									aria-label="Clear paper search"
									onclick={() => clearSearchQuery('paper')}
								>
									×
								</button>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		</div>

		{#if isLoading && papers.length === 0}
			<div class="loading">Loading...</div>
		{/if}
		<div class="list-content">
			{#each paperListItems as item (item.key)}
				{#if item.kind === 'year'}
					<div class="paper-year-header">{item.yearLabel}</div>
				{:else}
					<button
						class={`paper-item ${paperTone(item.paper)}`}
						class:selected={selectedPaper?.index === item.paper.index}
						onclick={() => selectPaper(item.paper)}
					>
						<p class="title">{item.paper.title}</p>
						<p class="meta">{item.paper.author} ({item.paper.year})</p>
						{#if firstHighlightColorForPaper(selectedTopic?.id, item.paper) ||
							tagsForPaper(selectedTopic?.id, item.paper).length > 0}
							<div class="paper-tag-list" aria-label="Paper tags">
								{#if firstHighlightColorForPaper(selectedTopic?.id, item.paper)}
									<span
										class={`highlight-dot color-${firstHighlightColorForPaper(selectedTopic?.id, item.paper)}`}
										aria-label="This paper has highlights"
										title="Has highlights"
									></span>
								{/if}
								{#each tagsForPaper(selectedTopic?.id, item.paper) as tag (tag)}
									<span class="tag-bubble">{tag}</span>
								{/each}
							</div>
						{/if}
					</button>
				{/if}
			{/each}
		</div>
	</div>

	<button
		class="splitter middle-splitter"
		aria-label="Middle splitter"
		onpointerdown={(event) => onSplitterDown('middle', event)}
	></button>

	<!-- 3. Preview Area -->
	<main class="preview">
		{#if !readerMode}
			<div class="preview-header">
				<div class="preview-toolbar">
					{#if isMobile}
						<button
							id="preview-toolbar-back-papers"
							type="button"
							class="nav-back"
							aria-label="Back to papers"
							onclick={() => (mobileStage = 1)}
						>
							<span aria-hidden="true">←</span>
							<span class="nav-back-label">Papers</span>
						</button>
					{/if}
					<button
						id="preview-toolbar-toggle-topics"
						type="button"
						class="toolbar-button"
						title="Toggle topics"
						aria-pressed={!showTopicPanel}
						onclick={() => (showTopicPanel = !showTopicPanel)}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 576 512"
							><path
								fill="currentColor"
								d="M97.5 400l50-160 379.4 0-50 160-379.4 0zm190.7 48L477 448c21 0 39.6-13.6 45.8-33.7l50-160c9.7-30.9-13.4-62.3-45.8-62.3l-379.4 0c-21 0-39.6 13.6-45.8 33.7L80.2 294.4 80.2 96c0-8.8 7.2-16 16-16l138.7 0c3.5 0 6.8 1.1 9.6 3.2L282.9 112c13.8 10.4 30.7 16 48 16l117.3 0c8.8 0 16 7.2 16 16l48 0c0-35.3-28.7-64-64-64L330.9 80c-6.9 0-13.7-2.2-19.2-6.4L273.3 44.8C262.2 36.5 248.8 32 234.9 32L96.2 32c-35.3 0-64 28.7-64 64l0 288c0 35.3 28.7 64 64 64l192 0z"
							/></svg
						>
					</button>
					<button
						id="preview-toolbar-toggle-papers"
						type="button"
						class="toolbar-button"
						title="Toggle papers"
						aria-pressed={!showPaperPanel}
						onclick={() => (showPaperPanel = !showPaperPanel)}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512"
							><path
								fill="currentColor"
								d="M40 48C26.7 48 16 58.7 16 72l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24L40 48zM192 64c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L192 64zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zM16 232l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0c-13.3 0-24 10.7-24 24zM40 368c-13.3 0-24 10.7-24 24l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0z"
							/></svg
						>
					</button>
					{#if !isMobile}
						<span class="preview-toolbar-splitter" aria-hidden="true"></span>
					{/if}
					<button
						id="preview-toolbar-view-report"
						type="button"
						class="toolbar-button"
						title="View report"
						disabled={!selectedPaper?.summary}
						onclick={() => {
							if (selectedPaper) loadPreview(selectedPaper, 'summary', { syncUrl: true });
						}}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512"
							><path
								fill="currentColor"
								d="M168 80c-13.3 0-24 10.7-24 24l0 304c0 8.4-1.4 16.5-4.1 24L440 432c13.3 0 24-10.7 24-24l0-304c0-13.3-10.7-24-24-24L168 80zM72 480c-39.8 0-72-32.2-72-72L0 112C0 98.7 10.7 88 24 88s24 10.7 24 24l0 296c0 13.3 10.7 24 24 24s24-10.7 24-24l0-304c0-39.8 32.2-72 72-72l272 0c39.8 0 72 32.2 72 72l0 304c0 39.8-32.2 72-72 72L72 480zM192 152c0-13.3 10.7-24 24-24l48 0c13.3 0 24 10.7 24 24l0 48c0 13.3-10.7 24-24 24l-48 0c-13.3 0-24-10.7-24-24l0-48zm152 24l48 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-48 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zM216 256l176 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-176 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zm0 80l176 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-176 0c-13.3 0-24-10.7-24-24s10.7-24 24-24z"
							/></svg
						>
					</button>
					<button
						id="preview-toolbar-view-presentation"
						type="button"
						class="toolbar-button"
						title="View presentation"
						disabled={!selectedPaper?.slide}
						onclick={() => {
							if (selectedPaper) loadPreview(selectedPaper, 'slide', { syncUrl: true });
						}}
						><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512"
							><path
								fill="currentColor"
								d="M448 96l0 256-384 0 0-256 384 0zM64 32C28.7 32 0 60.7 0 96L0 352c0 35.3 28.7 64 64 64l144 0-16 48-72 0c-13.3 0-24 10.7-24 24s10.7 24 24 24l272 0c13.3 0 24-10.7 24-24s-10.7-24-24-24l-72 0-16-48 144 0c35.3 0 64-28.7 64-64l0-256c0-35.3-28.7-64-64-64L64 32z"
							/></svg
						>
					</button>
					<button
						id="preview-toolbar-reader-mode"
						type="button"
						class="toolbar-button"
						title="Toggle reader mode"
						aria-label="Toggle reader mode"
						aria-pressed={readerMode}
						disabled={!canReader}
						onclick={() => (readerMode = !readerMode)}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 448 512">
							<path
								fill="currentColor"
								d="M168 32L24 32C10.7 32 0 42.7 0 56L0 200c0 9.7 5.8 18.5 14.8 22.2S34.1 223.8 41 217l40-40 79 79-79 79-40-40c-6.9-6.9-17.2-8.9-26.2-5.2S0 302.3 0 312L0 456c0 13.3 10.7 24 24 24l144 0c9.7 0 18.5-5.8 22.2-14.8s1.7-19.3-5.2-26.2l-40-40 79-79 79 79-40 40c-6.9 6.9-8.9 17.2-5.2 26.2S270.3 480 280 480l144 0c13.3 0 24-10.7 24-24l0-144c0-9.7-5.8-18.5-14.8-22.2s-19.3-1.7-26.2 5.2l-40 40-79-79 79-79 40 40c6.9 6.9 17.2 8.9 26.2 5.2S448 209.7 448 200l0-144c0-13.3-10.7-24-24-24L280 32c-9.7 0-18.5 5.8-22.2 14.8S256.2 66.1 263 73l40 40-79 79-79-79 40-40c6.9-6.9 8.9-17.2 5.2-26.2S177.7 32 168 32z"
							/>
						</svg>
					</button>
					<button
						id="preview-toolbar-open-arxiv"
						type="button"
						class="toolbar-button"
						title="Open arXiv page"
						aria-label="Open arXiv page"
						disabled={!selectedPaper?.url}
						onclick={() => openExternal('arxiv')}
					>
						arXiv
					</button>
					<button
						id="preview-toolbar-open-ar5iv"
						type="button"
						class="toolbar-button"
						title="Open ar5iv page"
						aria-label="Open ar5iv page"
						disabled={!selectedPaper?.url}
						onclick={() => openExternal('ar5iv')}
					>
						ar5iv
					</button>
					<button
						id="preview-toolbar-open-pdf"
						type="button"
						class="toolbar-button"
						title="Open PDF"
						aria-label="Open PDF"
						disabled={!selectedPaper?.url}
						onclick={() => openExternal('pdf')}
					>
						PDF
					</button>
					<button
						id="theme-toggle"
						type="button"
						class="toolbar-button theme-toggle"
						title="Toggle theme"
						style="padding-left: 8px;"
						aria-label="Toggle theme"
						aria-pressed={theme === 'dark'}
						onclick={() => (theme = theme === 'light' ? 'dark' : 'light')}
					>
						{theme === 'light' ? '☀ Light' : '◑ Dark'}
					</button>
					{#if renderType === 'summary'}
						<button
							id="toggle-sans-serif"
							type="button"
							class="toolbar-button"
							title="Toggle Noto Serif"
							aria-label="Toggle Noto Serif"
							aria-pressed={useNotoSerif}
							onclick={() => (useNotoSerif = !useNotoSerif)}
						>
							{#if useNotoSerif}
								Serif
							{:else}
								Sans
							{/if}
						</button>
						<div class="summary-scale-popover-wrap" onfocusout={closeSummaryScalePopoverOnFocusOut}>
							<button
								id="summary-scale-button"
								type="button"
								class="toolbar-button summary-scale-button"
								title="Adjust summary scale"
								aria-label="Adjust summary scale"
								aria-expanded={showSummaryScalePopover}
								disabled={showSummaryScalePopover}
								onclick={toggleSummaryScalePopover}
								bind:this={summaryScaleButtonEl}
							>
								F ⇅
							</button>
							{#if showSummaryScalePopover}
								<div
									class="summary-scale-popover"
									tabindex="-1"
									bind:this={summaryScalePopoverEl}
									style={summaryScalePopoverLeft
										? `--summary-popover-left: ${summaryScalePopoverLeft}; --summary-popover-right: auto;`
										: undefined}
								>
									<label class="summary-scale-control">
										<span class="summary-scale-label">Scale</span>
										<input
											class="summary-scale-range"
											type="range"
											min="100"
											max="200"
											step="1"
											value={summaryScalePct}
											oninput={(event) =>
												setSummaryScale(Number((event.currentTarget as HTMLInputElement).value))}
											bind:this={summaryScaleRangeEl}
											aria-label="Summary scale (percent)"
										/>
										<span class="summary-scale-value">{summaryScalePct}%</span>
									</label>
								</div>
							{/if}
						</div>
						<span class="preview-toolbar-splitter" aria-hidden="true"></span>
						<div class="summary-tag-panel">
							<div class="summary-tag-panel-inner">
								{#if editingTagKey === selectedPaperTagKey}
									<div class="tag-editor-row">
										<input
											class="tag-editor-input"
											type="text"
											maxlength="29"
											placeholder="태그1, 태그2"
											bind:value={tagDraft}
											bind:this={tagInputEl}
											onkeydown={(event) => {
												if (event.key === 'Enter') {
													event.preventDefault();
													skipTagBlurSave = true;
													saveTagEdit(selectedPaperTagKey);
												} else if (event.key === 'Escape') {
													event.preventDefault();
													skipTagBlurSave = true;
													cancelTagEdit();
												}
											}}
											onblur={() => saveTagEditOnBlur(selectedPaperTagKey)}
										/>
									</div>
								{:else if selectedPaperTags.length > 0}
									<div class="tag-bubble-list">
										{#each selectedPaperTags as tag (tag)}
											<button
												type="button"
												class="tag-bubble editable"
												onclick={() => openSelectedPaperTagEditor()}
											>
												{tag}
											</button>
										{/each}
									</div>
								{:else}
									<button
										type="button"
										class="tag-add-button"
										onclick={() => openSelectedPaperTagEditor()}
									>
										+태그
									</button>
								{/if}
							</div>
						</div>
					{/if}
					{#if renderType === 'slide'}
						<div class="slide-width-popover-wrap" onfocusout={closeSlideWidthPopoverOnFocusOut}>
							<button
								type="button"
								class="toolbar-button"
								title="Adjust slide width"
								style="letter-spacing: -3px; padding-left: 8px;"
								aria-label="Adjust slide width"
								aria-expanded={showSlideWidthPopover}
								disabled={showSlideWidthPopover || !canSlideWidth}
								onclick={toggleSlideWidthPopover}
								bind:this={slideWidthButtonEl}
							>
								|⟺|
							</button>
							{#if showSlideWidthPopover}
								<div
									class="slide-width-popover"
									tabindex="-1"
									bind:this={slideWidthPopoverEl}
									style={slideWidthPopoverLeft
										? `--slide-popover-left: ${slideWidthPopoverLeft}; --slide-popover-right: auto;`
										: undefined}
								>
									<label
										class="slide-width-control"
										class:disabled={!canSlideWidth}
										aria-label="Slide width control"
									>
										<span class="slide-width-label">Width</span>
										<input
											class="slide-width-range"
											type="range"
											min="30"
											max="100"
											step="1"
											bind:value={slideWidthPct}
											disabled={!canSlideWidth}
											aria-label="Slide width (percent of viewport width)"
											bind:this={slideWidthRangeEl}
										/>
										<span class="slide-width-value">{slideWidthPct}%</span>
									</label>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			</div>
		{/if}
		{#if readerMode}
			<button
				type="button"
				class="toolbar-button reader-close"
				aria-label="Exit reader mode"
				onclick={() => (readerMode = false)}
			>
				X
			</button>
			<div
				class="reader-body"
				role="presentation"
				bind:this={readerBodyEl}
				ontouchstart={onSlideTouchStart}
				ontouchend={onSlideTouchEnd}
			>
				{#if renderType === 'summary'}
					<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
					<article
						class="summary-content markdown-body"
						style={`font-size: ${summaryScalePct}%`}
						onmouseup={handleSummarySelectionChange}
						onkeyup={handleSummarySelectionChange}
						onclick={handleSummaryClick}
					>
						{#if isLoading}
							<p>Loading content...</p>
						{:else}
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html renderHtmlWithHighlights}
						{/if}
					</article>
				{:else if renderType === 'slide'}
					<div class="preview-content markdown-body slide-content">
						{#if isLoading}
							<p>Loading content...</p>
						{:else}
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html renderHtml}
						{/if}
					</div>
				{/if}
			</div>
		{:else if renderType === 'summary'}
			<div class="preview-content markdown-body">
				<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
				<article
					class="summary-content markdown-body"
					style={`font-size: ${summaryScalePct}%`}
					onmouseup={handleSummarySelectionChange}
					onkeyup={handleSummarySelectionChange}
					onclick={handleSummaryClick}
				>
					{#if isLoading}
						<p>Loading content...</p>
					{:else}
						<!-- eslint-disable-next-line svelte/no-at-html-tags -->
						{@html renderHtmlWithHighlights}
					{/if}
				</article>
			</div>
		{:else if renderType === 'slide'}
			<div
				class="preview-content markdown-body slide-content"
				role="presentation"
				bind:this={slidePreviewContentEl}
				ontouchstart={onSlideTouchStart}
				ontouchend={onSlideTouchEnd}
			>
				{#if isLoading}
					<p>Loading content...</p>
				{:else}
					<!-- eslint-disable-next-line svelte/no-at-html-tags -->
					{@html renderHtml}
				{/if}
			</div>
		{:else}
			<div class="placeholder">
				{#if selectedTopic}
					<h3>
						{selectedTopic.title}
						({selectedTopic.id})
					</h3>

					{#if selectedPaper}
						<h2>{selectedPaper?.title}</h2>
						<div class="clamp-5">
							{selectedPaper?.author}
							{#if selectedPaper?.year}
								({selectedPaper?.year})
							{/if}
						</div>
						<div>
							Select
							<button
								type="button"
								class="toolbar-button"
								disabled={!selectedPaper?.url}
								onclick={() => openExternal('arxiv')}
							>
								arXiv
							</button>,
							<button
								type="button"
								class="toolbar-button"
								disabled={!selectedPaper?.url}
								onclick={() => openExternal('ar5iv')}
							>
								ar5iv
							</button>
							or
							<button
								type="button"
								class="toolbar-button"
								disabled={!selectedPaper?.url}
								onclick={() => openExternal('pdf')}
							>
								PDF
							</button>
							{#if isLocalhost}
								<div class="prompt-popover-wrap" onfocusout={closePromptPopoverOnFocusOut}>
									Click&nbsp;
									<button
										type="button"
										class="toolbar-button prompt-button"
										disabled={!selectedPaper?.url || isPromptLoading}
										onclick={openPromptPopover}
										bind:this={promptButtonEl}
									>
										Prompt
									</button>
									{#if showPromptPopover}
										<div
											class="prompt-popover"
											tabindex="-1"
											bind:this={promptPopoverEl}
											style={promptPopoverLeft
												? `--prompt-popover-left: ${promptPopoverLeft}; --prompt-popover-right: auto;`
												: undefined}
										>
											<div class="prompt-popover-header">
												<span>Prompt Builder</span>
												<div class="prompt-actions">
													<button
														type="button"
														class="prompt-action"
														disabled={isPromptLoading}
														onclick={() => refreshPromptPaperText()}
													>
														Refresh
													</button>
													<button
														type="button"
														class:copied={promptCopyStatus === 'copied'}
														class="prompt-action"
														disabled={!promptDraft}
														onclick={copyPromptToClipboard}
													>
														{promptCopyStatus === 'copied' ? 'Copied' : 'Copy'}
													</button>
												</div>
											</div>
											<textarea
												class="prompt-textarea"
												bind:this={promptTextareaEl}
												bind:value={promptDraft}
											></textarea>
											{#if isPromptLoading}
												<div class="prompt-status">Extracting paper text...</div>
											{:else if promptError}
												<div class="prompt-status error">{promptError}</div>
											{:else if promptCopyStatus === 'failed'}
												<div class="prompt-status error">Copy failed.</div>
											{/if}
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{:else}
						<div>Select a paper.</div>
					{/if}
				{:else}
					<div>Select a research topic.</div>
				{/if}
			</div>
		{/if}
	</main>
</div>

{#if highlightPopover}
	<div
		class="highlight-popover"
		bind:this={highlightPopoverEl}
		style={`left: ${highlightPopover.left}px; top: ${highlightPopover.top}px;`}
	>
		<button
			type="button"
			class="highlight-popover-drag-handle left"
			aria-label="Move highlight popover"
			onpointerdown={onHighlightPopoverDragStart}
		></button>
		<button
			type="button"
			class="highlight-popover-drag-handle right"
			aria-label="Move highlight popover"
			onpointerdown={onHighlightPopoverDragStart}
		></button>
		<button
			type="button"
			class="highlight-popover-title"
			onpointerdown={onHighlightPopoverDragStart}
		>
			Highlight
		</button>
		<div class="highlight-color-row">
			{#each SUMMARY_HIGHLIGHT_COLORS as color (color)}
				<button
					type="button"
					class={`highlight-color-button color-${color}`}
					aria-label={`Use ${color} highlight`}
					onclick={() =>
						highlightPopover?.mode === 'create'
							? applyNewHighlight(color)
							: highlightPopover?.groupId && updateHighlightGroupColor(highlightPopover.groupId, color)}
				></button>
			{/each}
			{#if highlightPopover.mode === 'edit' && highlightPopover.groupId}
				<button
					type="button"
					class="highlight-color-button highlight-delete-icon"
					aria-label="Delete highlight"
					onclick={() => highlightPopover?.groupId && deleteHighlightGroup(highlightPopover.groupId)}
				>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
						<path fill="currentColor" d="M136.7 5.9C141.1-7.2 153.3-16 167.1-16l113.9 0c13.8 0 26 8.8 30.4 21.9L320 32 416 32c17.7 0 32 14.3 32 32s-14.3 32-32 32L32 96C14.3 96 0 81.7 0 64S14.3 32 32 32l96 0 8.7-26.1zM32 144l384 0 0 304c0 35.3-28.7 64-64 64L96 512c-35.3 0-64-28.7-64-64l0-304zm88 64c-13.3 0-24 10.7-24 24l0 192c0 13.3 10.7 24 24 24s24-10.7 24-24l0-192c0-13.3-10.7-24-24-24zm104 0c-13.3 0-24 10.7-24 24l0 192c0 13.3 10.7 24 24 24s24-10.7 24-24l0-192c0-13.3-10.7-24-24-24zm104 0c-13.3 0-24 10.7-24 24l0 192c0 13.3 10.7 24 24 24s24-10.7 24-24l0-192c0-13.3-10.7-24-24-24z"/>
					</svg>
				</button>
			{/if}
		</div>
	</div>
{/if}

<style>
	/* Layout variables (theme-independent) */
	:global(:root) {
		--col1: 250px;
		--col2: 350px;
	}

	/* Light Theme */
	:global([data-theme='light']) {
		/* Surface */
		--bg-base: #f4f5f7;
		--bg-panel: #ffffff;
		--bg-panel-alt: #f9fafb;
		--bg-hover: #eef0f5;
		--bg-selected: #eaf0fd;
		--bg-header: linear-gradient(135deg, #f0f2f8 0%, #e8ecf6 100%);

		/* Border */
		--border-subtle: rgba(0, 0, 0, 0.07);
		--border-default: rgba(0, 0, 0, 0.12);
		--border-strong: rgba(0, 0, 0, 0.22);

		/* Text */
		--text-primary: #1a1d2e;
		--text-secondary: #4a5070;
		--text-muted: #8a90a8;
		--text-disabled: #c0c4d0;
		--link-color: #2a5cc8;

		/* Accent - indigo */
		--accent: #4c5fd5;
		--accent-dim: #7b8de0;
		--accent-subtle: rgba(76, 95, 213, 0.1);
		--accent-text: #3347b8;
		--paper-tag-bg: #ffe082;
		--paper-tag-text: #5f4200;

		/* Paper tone colors */
		--tone-gray-fg: #7a849e;
		--tone-gray-bar: #b0b8cc;
		--tone-black-fg: #2a2e3a;
		--tone-black-bar: #6a7080;
		--tone-blue-fg: #2a5cc8;
		--tone-blue-bar: #6090e0;
		--tone-pink-fg: #5c2ac8;
		--tone-pink-bar: #9060e0;

		/* Shadows */
		--shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
		--shadow-md: 0 4px 12px rgba(0, 0, 0, 0.1);
		--shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);

		/* Summary card */
		--summary-bg: #ffffff;
		--summary-color: #1a1a1a;
		--summary-border: rgba(0, 0, 0, 0.08);
		--table-head-bg: #eef0f6;
		--table-head-color: #2a2e45;
		--table-border: #d0d5e0;
		--table-row-hover: #f3f5fb;
		--highlight-yellow: rgba(255, 232, 122, 0.72);
		--highlight-pink: rgba(255, 173, 205, 0.78);
		--highlight-blue: rgba(160, 209, 255, 0.78);
		--highlight-green: rgba(185, 230, 140, 0.78);
		--highlight-dot-yellow: #8a5a00;
		--highlight-dot-pink: #8f2d5a;
		--highlight-dot-blue: #1f5fa8;
		--highlight-dot-green: #456b16;

		/* Select widget */
		--select-bg: #ffffff;
		--select-border: rgba(0, 0, 0, 0.15);
		--select-color: #4a5070;

		/* Slide background */
		--slide-wrapper-bg: #e8eaef;
	}

	/* Dark Theme */
	:global([data-theme='dark']) {
		--bg-base: #0f1117;
		--bg-panel: #16181f;
		--bg-panel-alt: #1c1f2a;
		--bg-hover: #21263a;
		--bg-selected: #1e2640;
		--bg-header: linear-gradient(135deg, #1a1e2e 0%, #1f2438 100%);

		--border-subtle: rgba(255, 255, 255, 0.06);
		--border-default: rgba(255, 255, 255, 0.1);
		--border-strong: rgba(255, 255, 255, 0.18);

		--text-primary: #e8eaf0;
		--text-secondary: #9097b0;
		--text-muted: #5a6080;
		--text-disabled: #363b52;
		--link-color: #7eb3ff;

		--accent: #6474f0;
		--accent-dim: #3d4a9e;
		--accent-subtle: rgba(100, 116, 240, 0.12);
		--accent-text: #a8b4ff;
		--paper-tag-bg: #5e4300;
		--paper-tag-text: #ffd86b;

		--tone-gray-fg: #6b7a9e;
		--tone-gray-bar: #3a415a;
		--tone-black-fg: #c8ccd8;
		--tone-black-bar: #8890a8;
		--tone-blue-fg: #7eb3ff;
		--tone-blue-bar: #4478cc;
		--tone-pink-fg: #b37eff;
		--tone-pink-bar: #7844cc;

		--shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4);
		--shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
		--shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);

		--summary-bg: #1c1f2a;
		--summary-color: #dde0ea;
		--summary-border: rgba(255, 255, 255, 0.07);
		--table-head-bg: #252a3a;
		--table-head-color: #a8b0cc;
		--table-border: #2e3448;
		--table-row-hover: #222638;
		--highlight-yellow: rgba(180, 125, 0, 0.58);
		--highlight-pink: rgba(154, 53, 105, 0.62);
		--highlight-blue: rgba(34, 92, 168, 0.62);
		--highlight-green: rgba(69, 120, 28, 0.64);
		--highlight-dot-yellow: #ffd86b;
		--highlight-dot-pink: #ffb7da;
		--highlight-dot-blue: #9fd0ff;
		--highlight-dot-green: #c7ef8d;

		--select-bg: #1c1f2a;
		--select-border: rgba(255, 255, 255, 0.12);
		--select-color: #9097b0;

		/* Slide background */
		--slide-wrapper-bg: #0f1117;
	}

	/* Reset / Base */
	:global(body) {
		margin: 0;
		font-size: 14px;
		font-family:
			'Pretendard Variable',
			'Pretendard',
			-apple-system,
			BlinkMacSystemFont,
			'Apple SD Gothic Neo',
			'Noto Sans KR',
			'Malgun Gothic',
			'Segoe UI',
			sans-serif;
		line-height: 1.5;
		background: var(--bg-base);
		color: var(--text-primary);
		height: 100vh;
		overflow: hidden;
		-webkit-font-smoothing: antialiased;
		transition:
			background 0.2s,
			color 0.2s;
	}

	/* Layout Grid */
	.app-container {
		display: grid;
		grid-template-columns:
			minmax(180px, var(--col1))
			6px
			minmax(240px, var(--col2))
			6px
			minmax(320px, 1fr);
		height: 100vh;
		background: var(--bg-base);
	}

	.topic-list {
		grid-column: 1;
	}
	.left-splitter {
		grid-column: 2;
	}
	.paper-list {
		grid-column: 3;
	}
	.middle-splitter {
		grid-column: 4;
	}
	.preview {
		grid-column: 5;
	}

	/* Side Panels */
	.topic-list,
	.paper-list {
		display: flex;
		flex-direction: column;
		background: var(--bg-panel);
		border-right: 1px solid var(--border-subtle);
		height: 100vh;
		overflow: hidden;
		transition:
			background 0.2s,
			border-color 0.2s;
	}

	.topic-list-header,
	.paper-list-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.65rem 0.9rem;
		background: var(--bg-header);
		border-bottom: 1px solid var(--border-default);
	}

	.topic-list-header h2,
	.paper-list-header h2 {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0;
		font-size: 0.75rem;
		font-weight: bold;
		text-transform: uppercase;
		flex-grow: 0;
		max-height: 18px;
	}

	.topic-list-header h2 svg,
	.paper-list-header h2 svg {
		opacity: 0.5;
	}

	/* Search UI */
	.search-container {
		position: relative;
		max-height: 18px;
	}
	.search-container .search-button {
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--text-secondary);
		padding: 0;
		opacity: 0.5;
	}
	.search-container .search-button:hover {
		background: var(--bg-hover);
		opacity: 1;
	}

	.search-popover {
		position: absolute;
		top: 100%;
		/* width: 14rem; */
		right: 0;
		margin-top: 6px;
		background: var(--bg-panel);
		border: 1px solid var(--border-default);
		border-radius: 6px;
		padding: 0.5rem;
		box-shadow: var(--shadow-md);
		z-index: 10;
	}
	.search-popover .search-input {
		position: relative;
		display: flex;
		align-items: center;
	}
	.search-popover input {
		width: 10rem;
		padding: 0.4rem 2rem 0.4rem 0.6rem;
		font-family: inherit;
		font-size: 0.85rem;
		border: 1px solid var(--border-strong);
		border-radius: 4px;
		background: var(--bg-base);
		color: var(--text-primary);
	}
	.search-popover input:focus {
		outline: 2px solid var(--accent);
		border-color: transparent;
	}
	.search-popover .search-clear {
		position: absolute;
		right: 6px;
		top: 50%;
		transform: translateY(-50%);
		width: 22px;
		height: 22px;
		padding: 0;
		margin: 0;
		vertical-align: middle;
		border-radius: 999px;
		border: none;
		background: transparent;
		color: var(--text-secondary);
		cursor: pointer;
		opacity: 0.65;
		line-height: 1;
		display: grid;
		place-items: center;
		font-size: 1rem;
	}
	.search-popover .search-clear:hover {
		background: var(--bg-hover);
		opacity: 1;
	}

	.highlight-popover {
		position: fixed;
		z-index: 60;
		transform: translateX(-50%);
		background: var(--bg-panel);
		border: 1px solid var(--border-default);
		border-radius: 12px;
		box-shadow: var(--shadow-md);
		padding: 0.65rem 0.75rem;
		min-width: 12rem;
	}

	.highlight-popover-drag-handle {
		position: absolute;
		top: 0.4rem;
		bottom: 0.4rem;
		width: 0.8rem;
		appearance: none;
		background: transparent;
		border: 0;
		padding: 0;
		cursor: grab;
		touch-action: none;
		user-select: none;
	}

	.highlight-popover-drag-handle.left {
		left: 0.1rem;
	}

	.highlight-popover-drag-handle.right {
		right: 0.1rem;
	}

	.highlight-popover-drag-handle:active {
		cursor: grabbing;
	}

	.highlight-popover-title {
		appearance: none;
		background: none;
		border: 0;
		padding: 0;
		width: 100%;
		text-align: left;
		font-size: 0.72rem;
		font-weight: 700;
		letter-spacing: 0.03em;
		text-transform: uppercase;
		color: var(--text-secondary);
		margin-bottom: 0.55rem;
		cursor: grab;
		touch-action: none;
		user-select: none;
	}

	.highlight-popover-title:active {
		cursor: grabbing;
	}

	.highlight-color-row {
		display: flex;
		align-items: center;
		gap: 0.45rem;
	}

	.highlight-color-button {
		width: 1.45rem;
		height: 1.45rem;
		border-radius: 999px;
		border: 1px solid var(--border-default);
		cursor: pointer;
		box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.45);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font: inherit;
	}

	.highlight-color-button.color-yellow {
		background: #fff3a3;
	}

	.highlight-color-button.color-pink {
		background: #ffd3e1;
	}

	.highlight-color-button.color-blue {
		background: #cfe7ff;
	}

	.highlight-color-button.color-green {
		background: #d9f3bf;
	}

	.highlight-delete-icon {
		background: var(--bg-panel-alt);
		color: var(--text-secondary);
		font-size: 0.9rem;
		font-weight: 700;
		line-height: 1;
		text-transform: lowercase;
	}

	.highlight-delete-icon:hover {
		background: var(--bg-hover);
		color: var(--text-primary);
	}

	/* Scrollbars */
	.topic-content::-webkit-scrollbar,
	.list-content::-webkit-scrollbar,
	.preview-content::-webkit-scrollbar {
		width: 4px;
	}
	.topic-content::-webkit-scrollbar-track,
	.list-content::-webkit-scrollbar-track,
	.preview-content::-webkit-scrollbar-track {
		background: transparent;
	}
	.topic-content::-webkit-scrollbar-thumb,
	.list-content::-webkit-scrollbar-thumb,
	.preview-content::-webkit-scrollbar-thumb {
		background: var(--border-strong);
		border-radius: 99px;
	}

	.topic-content,
	.list-content {
		overflow-y: auto;
		flex: 1;
		padding: 0.4rem;
		-webkit-overflow-scrolling: touch;
		touch-action: pan-y;
	}

	/* Topic List */
	.topic-list ul {
		list-style: none;
		padding: 0;
		margin: 0;
	}
	.topic-list button {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.42rem 0.7rem;
		margin: 2px 0;
		border: 1px solid transparent;
		background: none;
		color: var(--text-secondary);
		cursor: pointer;
		border-radius: 5px;
		font-size: 0.83rem;
		font-family: inherit;
		transition:
			background 0.15s,
			color 0.15s,
			border-color 0.15s;
	}
	.topic-list button:hover {
		background: var(--bg-hover);
		color: var(--text-primary);
		border-color: var(--border-subtle);
	}
	.topic-list button.active {
		background: var(--accent-subtle);
		color: var(--accent-text);
		border-color: var(--accent-dim);
		font-weight: 600;
	}

	/* Paper List */
	.paper-year-header {
		margin: 0.75rem 0 0.25rem;
		padding: 0.2rem 0.15rem;
		font-size: 0.72rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--text-muted);
		border-bottom: 1px solid var(--border-subtle);
	}
	.paper-item {
		display: block;
		width: 100%;
		text-align: left;
		margin: 3px 0;
		padding: 0.55rem 0.7rem 0.55rem 0.85rem;
		max-height: 15em;
		overflow: hidden;
		background: var(--bg-panel-alt);
		border: 1px solid var(--border-subtle);
		border-left-width: 3px;
		border-radius: 5px;
		cursor: pointer;
		font-family: inherit;
		transition:
			background 0.15s,
			border-color 0.15s,
			transform 0.1s;
	}
	.paper-item:hover {
		background: var(--bg-hover);
		border-color: var(--border-default);
		transform: translateX(2px);
	}
	.paper-item.selected {
		background: var(--bg-selected);
		border-color: var(--accent-dim);
		border-left-color: var(--accent);
		box-shadow: var(--shadow-sm);
		border-right-width: 2px;
		border-top-width: 2px;
		border-bottom-width: 2px;
	}
	.paper-item.paper-gray {
		color: var(--tone-gray-fg);
		border-left-color: var(--tone-gray-bar);
	}
	.paper-item.paper-black {
		color: var(--tone-black-fg);
		border-left-color: var(--tone-black-bar);
	}
	.paper-item.paper-blue {
		color: var(--tone-blue-fg);
		border-left-color: var(--tone-blue-bar);
	}

	.paper-item.paper-pink {
		color: var(--tone-pink-fg);
		border-left-color: var(--tone-pink-bar);
	}

	.paper-item .title {
		margin: 0 0 0.25rem 0;
		font-size: 0.83rem;
		font-weight: 600;
		line-height: 1.35;
		color: inherit;
	}
	.paper-item .meta {
		margin: 0;
		font-size: 0.72rem;
		color: var(--text-muted);
		overflow: hidden;
		display: -webkit-box;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 3;
	}

	.paper-tag-list {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		margin-top: 0.45rem;
		align-items: center;
	}

	.paper-tag-list .tag-bubble {
		border-color: transparent;
		background: var(--paper-tag-bg);
		color: var(--paper-tag-text);
		padding: 0.2rem 0.55rem;
		font-weight: 700;
	}

	.paper-tag-list .highlight-dot {
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 999px;
		flex: 0 0 auto;
		box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.25);
	}

	.paper-tag-list .highlight-dot.color-yellow {
		background: var(--highlight-dot-yellow);
	}

	.paper-tag-list .highlight-dot.color-pink {
		background: var(--highlight-dot-pink);
	}

	.paper-tag-list .highlight-dot.color-blue {
		background: var(--highlight-dot-blue);
	}

	.paper-tag-list .highlight-dot.color-green {
		background: var(--highlight-dot-green);
	}

	.summary-tag-panel {
		display: inline-flex;
		align-items: center;
		min-height: 28px;
		max-width: min(100%, 28rem);
	}

	.summary-tag-panel-inner {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.tag-editor-row {
		width: 100%;
		max-width: 100%;
	}

	.tag-editor-input {
		width: min(100%, 16rem);
		padding: 0.2rem 0.8rem;
		border: 1px solid var(--border-default);
		border-radius: 999px;
		background: var(--bg-panel-alt);
		color: var(--text-primary);
		font: inherit;
		min-height: 20px;
	}

	.tag-editor-input:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent-subtle);
	}

	.tag-add-button,
	.tag-bubble {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0.1rem 0.4rem;
		border-radius: 999px;
		border: 1px solid var(--border-default);
		background: var(--paper-tag-bg);
		color: var(--paper-tag-text);
		font-size: 0.6rem;
		font-weight: 600;
	}

	.tag-bubble-list {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
	}

	.tag-add-button {
		cursor: pointer;
		transition:
			background 0.15s,
			color 0.15s,
			border-color 0.15s;
	}

	.tag-bubble.editable {
		cursor: pointer;
		font-size: 0.65rem;
	}

	.tag-add-button:hover,
	.tag-bubble.editable:hover {
		background: var(--accent-subtle);
		color: var(--accent-text);
		border-color: var(--accent-dim);
	}

	.placeholder .clamp-5 {
		overflow: hidden;
		display: -webkit-box;
		line-clamp: 5;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 5;
	}

	/* Splitter */
	.splitter {
		background: var(--bg-base);
		cursor: col-resize;
		position: relative;
		border: none;
		padding: 0;
		width: 6px;
		transition: background 0.2s;
	}
	.splitter:hover {
		background: var(--bg-hover);
	}
	.splitter::after {
		content: '';
		position: absolute;
		inset: 0;
		margin: auto;
		width: 2px;
		height: 48px;
		border-radius: 99px;
		background: var(--border-strong);
		transition:
			background 0.2s,
			height 0.2s;
	}
	.splitter:hover::after {
		background: var(--accent-dim);
		height: 72px;
	}

	/* Preview */
	.preview {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: var(--bg-panel);
		transition: background 0.2s;
	}
	.preview-header {
		padding: 0.34rem 0.5rem;
		background: var(--bg-header);
		border-bottom: 1px solid var(--border-default);
		flex-shrink: 0;
	}
	.preview-header-top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
	}
	.preview-toolbar {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: center;
	}
	.preview-toolbar-splitter {
		width: 1px;
		height: 1.4rem;
		background: var(--border-default);
		opacity: 0.9;
		flex: 0 0 auto;
	}

	.slide-width-control {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		border: 1px solid var(--border-default);
		border-radius: 999px;
		background: var(--bg-panel-alt);
		color: var(--text-secondary);
		font-size: 0.72rem;
		font-weight: 600;
		padding: 0 0.6rem;
		min-height: 28px;
	}
	.slide-width-label {
		opacity: 0.8;
	}
	.slide-width-range {
		width: 7.5rem;
	}
	.slide-width-value {
		font-variant-numeric: tabular-nums;
		min-width: 3.25ch;
		text-align: right;
		opacity: 0.85;
	}
	.slide-width-control.disabled {
		background: transparent;
		border-color: var(--border-subtle);
		color: var(--text-disabled);
	}
	.slide-width-popover-wrap {
		position: relative;
		display: inline-flex;
		align-items: center;
	}
	.slide-width-popover {
		position: absolute;
		top: calc(100% + 0.4rem);
		left: var(--slide-popover-left, auto);
		right: var(--slide-popover-right, 0);
		z-index: 20;
		background: var(--bg-panel);
		border: 1px solid var(--border-default);
		border-radius: 12px;
		box-shadow: var(--shadow-md);
		padding: 0.6rem 0.75rem;
		min-width: 12rem;
	}

	/* Reader Mode */
	.app-container[data-reader='true'] {
		grid-template-columns: 1fr;
	}
	.app-container[data-reader='true'] .topic-list,
	.app-container[data-reader='true'] .paper-list,
	.app-container[data-reader='true'] .left-splitter,
	.app-container[data-reader='true'] .middle-splitter {
		display: none;
	}
	.app-container[data-reader='true'] .preview {
		grid-column: 1;
	}
	.reader-body {
		flex: 1;
		overflow-y: auto;
		background: var(--bg-base);
		padding-top: 3.25rem;
	}
	.reader-body .preview-content {
		overflow: visible;
		background: transparent;
	}
	.reader-close {
		position: fixed;
		top: 1rem;
		right: 2rem;
		z-index: 50;
		width: 2rem;
		height: 2rem;
		padding: 0;
	}

	/* Toolbar Buttons */
	.toolbar-button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 0.3rem;
		border: 1px solid var(--border-default);
		border-radius: 999px;
		background: var(--bg-panel-alt);
		color: var(--text-secondary);
		font: inherit;
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		padding: 0 0.75rem;
		min-height: 28px;
		cursor: pointer;
		transition:
			background 0.15s,
			color 0.15s,
			border-color 0.15s,
			transform 0.1s,
			box-shadow 0.15s;
	}
	.toolbar-button:hover {
		background: var(--accent-subtle);
		color: var(--accent-text);
		border-color: var(--accent-dim);
		transform: translateY(-1px);
		box-shadow: 0 2px 8px rgba(100, 116, 240, 0.15);
	}
	.toolbar-button:active {
		transform: translateY(0);
		box-shadow: none;
	}
	.toolbar-button:disabled {
		opacity: 1;
		cursor: not-allowed;
		transform: none;
		color: var(--text-disabled);
		background: transparent;
		border-color: var(--border-subtle);
		box-shadow: none;
		pointer-events: none;
	}
	button svg {
		flex-shrink: 0;
	}

	.theme-toggle {
		padding: 0 0.75rem;
	}

	.summary-scale-popover-wrap {
		position: relative;
		display: inline-flex;
		align-items: center;
	}
	.summary-scale-button {
		padding: 0 0.6rem;
	}
	.summary-scale-popover {
		position: absolute;
		top: calc(100% + 0.4rem);
		left: var(--summary-popover-left, auto);
		right: var(--summary-popover-right, 0);
		z-index: 20;
		background: var(--bg-panel);
		border: 1px solid var(--border-default);
		border-radius: 12px;
		box-shadow: var(--shadow-md);
		padding: 0.6rem 0.75rem;
		min-width: 12rem;
	}
	.summary-scale-control {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		color: var(--text-secondary);
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.02em;
	}
	.summary-scale-label {
		opacity: 0.7;
	}
	.summary-scale-range {
		width: 7.5rem;
	}
	.summary-scale-value {
		font-variant-numeric: tabular-nums;
		min-width: 3.25ch;
		text-align: right;
		opacity: 0.85;
	}

	.prompt-popover-wrap {
		position: relative;
		display: flex;
		align-items: center;
		margin: 1rem auto 10rem;
	}

	.prompt-popover {
		position: absolute;
		top: calc(100% + 0.4rem);
		left: var(--prompt-popover-left, 0);
		right: var(--prompt-popover-right, auto);
		z-index: 20;
		background: var(--bg-panel);
		border: 1px solid var(--border-default);
		border-radius: 12px;
		box-shadow: var(--shadow-md);
		padding: 0.6rem 0.75rem;
		min-width: min(32rem, 92vw);
		max-width: min(32rem, 92vw);
	}

	.prompt-popover-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		color: var(--text-secondary);
		font-size: 0.72rem;
		font-weight: 700;
		letter-spacing: 0.03em;
		text-transform: uppercase;
		margin-bottom: 0.4rem;
	}

	.prompt-actions {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
	}

	.prompt-action {
		border: 1px solid var(--border-default);
		border-radius: 999px;
		background: var(--bg-panel-alt);
		color: var(--text-secondary);
		font-size: 0.65rem;
		font-weight: 600;
		padding: 0.1rem 0.55rem;
		cursor: pointer;
	}

	.prompt-action:hover {
		background: var(--bg-hover);
	}

	.prompt-action.copied {
		background: var(--text-primary);
		color: var(--bg-panel);
		border-color: var(--text-primary);
	}

	.prompt-action.copied:hover {
		background: var(--text-primary);
	}

	.prompt-textarea {
		width: 100%;
		min-height: 12rem;
		max-height: 50vh;
		resize: vertical;
		box-sizing: border-box;
		border: 1px solid var(--border-default);
		border-radius: 10px;
		background: var(--bg-panel-alt);
		color: var(--text-primary);
		padding: 0.6rem 0.7rem;
		margin: 0;
		font: inherit;
		line-height: 1.5;
	}

	.prompt-status {
		margin-top: 0.35rem;
		font-size: 0.72rem;
		color: var(--text-muted);
	}

	.prompt-status.error {
		color: var(--tone-pink-fg);
	}

	/* Preview Content */
	.preview-content {
		flex: 1;
		overflow-x: auto;
		overflow-y: auto;
		background: var(--bg-base);
		-webkit-overflow-scrolling: touch;
		touch-action: auto;
		transition: background 0.2s;
	}
	.summary-content {
		max-width: 7.2in;
		padding: 2.5rem 4rem 5rem;
		margin: 2rem auto;
		background: var(--summary-bg);
		color: var(--summary-color);
		border-radius: 12px;
		box-shadow: var(--shadow-lg);
		border: 1px solid var(--summary-border);
		transition:
			background 0.2s,
			box-shadow 0.2s;
	}

	:global(body[data-noto='on']) .summary-content {
		font-family: 'Noto Serif KR', serif;
	}
	:global(body[data-noto='off']) .summary-content {
		font-family: 'Pretendard Variable', 'Pretendard';
	}

	.summary-content :global(h1),
	.summary-content :global(h2),
	.summary-content :global(h3),
	.summary-content :global(h4),
	.summary-content :global(h5),
	.summary-content :global(em),
	.summary-content :global(strong),
	.summary-content :global(pre),
	.summary-content :global(code) {
		font-family: 'Pretendard Variable', 'Pretendard';
	}

	.summary-content :global(strong) {
		font-weight: 600;
	}
	.summary-content :global(a) {
		color: var(--link-color);
	}

	.summary-content :global(table) {
		width: 100%;
		border-collapse: collapse;
		border: 1px solid var(--table-border);
		font-size: 0.9rem;
		margin: 1rem 0;
	}
	.summary-content :global(table th),
	.summary-content :global(table td) {
		border: 1px solid var(--table-border);
		padding: 0.4em 0.75em;
	}
	.summary-content :global(table th) {
		background: var(--table-head-bg);
		font-weight: 700;
		color: var(--table-head-color);
		border-bottom: 2px solid var(--table-border);
		letter-spacing: 0.02em;
	}
	.summary-content :global(table tr:hover td) {
		background: var(--table-row-hover);
	}
	.summary-content :global(p) {
		line-height: 1.7;
	}

	.summary-content :global(.summary-highlight) {
		border-top-left-radius: 0.2em;
		border-top-right-radius: 0.2em;
		padding: 0 0.05em;
		cursor: pointer;
		box-decoration-break: clone;
		-webkit-box-decoration-break: clone;
		border-bottom-width: 1px;
		border-bottom-style: solid;
	}

	.summary-content :global(.summary-highlight.highlight-yellow) {
		background: var(--highlight-yellow);
		border-bottom-color: var(--highlight-dot-yellow);
	}

	.summary-content :global(.summary-highlight.highlight-pink) {
		background: var(--highlight-pink);
		border-bottom-color: var(--highlight-dot-pink);
	}

	.summary-content :global(.summary-highlight.highlight-blue) {
		background: var(--highlight-blue);
		border-bottom-color: var(--highlight-dot-blue);
	}

	.summary-content :global(.summary-highlight.highlight-green) {
		background: var(--highlight-green);
		border-bottom-color: var(--highlight-dot-green);
	}

	.summary-content :global(p > img:only-of-type) {
    display: block;
		max-height: 100dvh;
		max-width: 100%;
		margin: 1em auto;
	}

	.marp-slide-wrapper {
		display: flex;
		justify-content: center;
		width: var(--slide-max-width);
		max-width: 100%;
		padding: 0.5rem;
		margin: 0 auto;
		background: var(--slide-wrapper-bg);
		min-height: 100%;
	}
	.marp-slide-wrapper :global(.marp-svg) {
		display: block;
		max-height: 100dvh;
		width: 100%;
		height: auto;
		padding: 2em;
	}

	/* Placeholder & Loading */
	.placeholder {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 1.25rem;
		height: 100%;
		color: var(--text-muted);
		font-size: 0.85rem;
		padding: 2em;
	}
	.placeholder-header {
		align-self: flex-end;
		padding: 0 1rem;
	}
	.loading {
		padding: 1.5rem;
		color: var(--text-muted);
		text-align: center;
		font-size: 0.82rem;
		letter-spacing: 0.05em;
	}

	/* Panel Visibility */
	.app-container[data-show-topic='false'][data-show-paper='true'] {
		grid-template-columns: 0 0 minmax(240px, var(--col2)) 6px minmax(320px, 1fr);
	}
	.app-container[data-show-topic='true'][data-show-paper='false'] {
		grid-template-columns: minmax(180px, var(--col1)) 6px 0 0 minmax(320px, 1fr);
	}
	.app-container[data-show-topic='false'][data-show-paper='false'] {
		grid-template-columns: 0 0 0 0 minmax(320px, 1fr);
	}
	.app-container[data-show-topic='false'] .topic-list,
	.app-container[data-show-topic='false'] .left-splitter,
	.app-container[data-show-paper='false'] .paper-list,
	.app-container[data-show-paper='false'] .middle-splitter {
		visibility: hidden;
		pointer-events: none;
		overflow: hidden;
	}

	/* Responsive */
	@media (max-width: 1400px) {
		.summary-content {
			padding: 2rem;
		}
	}
	@media (max-width: 960px) {
		.app-container {
			grid-template-columns: 1fr;
			grid-template-rows: 1fr;
			position: relative;
			height: 100dvh;
			overflow: hidden;
		}
		#preview-toolbar-toggle-topics,
		#preview-toolbar-toggle-papers {
			display: none;
		}
		.topic-list,
		.paper-list,
		.preview {
			grid-column: 1;
			grid-row: 1;
			position: absolute;
			inset: 0;
			height: 100dvh;
			border-right: 0;
			border-bottom: 0;
			pointer-events: none;
			transform: translateX(calc((var(--panel-index) - var(--mobile-stage)) * 100%));
			transition: transform 240ms cubic-bezier(0.2, 0.8, 0.2, 1);
			will-change: transform;
		}
		.topic-list {
			--panel-index: 0;
		}
		.paper-list {
			--panel-index: 1;
		}
		.preview {
			--panel-index: 2;
		}
		.splitter {
			display: none;
		}
		.app-container[data-mobile-stage='0'] .topic-list {
			pointer-events: auto;
		}
		.app-container[data-mobile-stage='1'] .paper-list {
			pointer-events: auto;
		}
		.app-container[data-mobile-stage='2'] .preview {
			pointer-events: auto;
		}

		.nav-back {
			display: inline-flex;
			align-items: center;
			gap: 0.35rem;
			padding: 0.35rem 0.55rem;
			border-radius: 10px;
			border: 1px solid var(--border-default);
			background: var(--bg-panel);
			color: var(--text-secondary);
			font-size: 0.78rem;
			line-height: 1;
			white-space: nowrap;
		}
		.nav-back:hover {
			background: var(--bg-hover);
		}
		.paper-list-header,
		.preview-toolbar {
			gap: 0.5rem;
		}
		.paper-list-header {
			justify-content: flex-start;
		}
		.paper-list-header .search-container {
			margin-left: auto;
		}
		.summary-content {
			padding: 1rem;
			margin: 0.5rem auto;
			border-radius: 8px;
		}
	}

	@media (max-width: 960px) and (prefers-reduced-motion: reduce) {
		.topic-list,
		.paper-list,
		.preview {
			transition: none;
		}
	}

	/* Print */
	@media print {
		:global(html) {
			height: auto;
			overflow: visible;
		}
		:global(body) {
			background: white;
			height: auto;
			overflow: visible;
		}
		.app-container {
			display: block;
			height: auto;
			min-height: 0;
			overflow: visible;
			background: white;
		}
		.preview {
			display: block;
			height: auto;
			min-height: 0;
			overflow: visible;
			background: white;
		}
		.topic-list,
		.left-splitter,
		.paper-list,
		.middle-splitter,
		.preview-header {
			display: none;
		}
		.preview-content {
			display: block;
			flex: none;
			width: 100%;
			height: auto;
			min-height: 0;
			overflow: visible;
			background: white;
			border: none;
		}
		.summary-content {
			max-width: none;
			box-shadow: none;
			border: none;
			padding: 0;
			margin: 0 auto;
			background: white;
			color: black;
			break-inside: auto;
			page-break-inside: auto;
		}
		.summary-content :global(h1) {
			break-after: avoid;
			page-break-after: avoid;
		}
		.summary-content :global(h2) {
			break-after: avoid;
			page-break-after: avoid;
		}
		.summary-content :global(h3) {
			break-after: avoid;
			page-break-after: avoid;
		}
		.summary-content :global(h4) {
			break-after: avoid;
			page-break-after: avoid;
		}

		:global(div.marp-slide-wrapper) {
			padding: 0.2rem;
			background: white;
		}
		:global(div.marp-slide-wrapper .marp-svg) {
			margin: 2rem 0 4rem;
		}
	}
</style>
