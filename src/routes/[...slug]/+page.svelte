<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
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

	type RoutePaperMatch = { paper: Paper; type: 'summary' | 'slide' };

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

	function findPaperBySlug(paperSlug: string, allPapers: Paper[]): RoutePaperMatch | null {
		const targetFilename = `${paperSlug}.md`;

		for (const paper of allPapers) {
			if (paper.summary && basename(paper.summary) === targetFilename) return { paper, type: 'summary' };
			if (paper.slide && basename(paper.slide) === targetFilename) return { paper, type: 'slide' };
		}

		return null;
	}

	// SvelteKit rest params are strings like "a/b/c" (or `undefined` at `/`).
	let routeSlug = $derived($page.params.slug as string | undefined);
	let routeSegments = $derived(
		(routeSlug ? routeSlug.split('/').filter(Boolean) : []) as string[]
	);
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

	function closeSearchPopoverOnFocusOut(
		event: FocusEvent,
		kind: 'topic' | 'paper'
	) {
		const container = event.currentTarget as HTMLElement | null;
		const next = event.relatedTarget as Node | null;
		if (container && next && container.contains(next)) return;

		if (kind === 'topic') showSearchTopicInput = false;
		else showSearchPaperInput = false;
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
		topics.filter(
			(p) =>
				searchTopicQuery.trim() === '' ||
				p.title.toLowerCase().includes(searchTopicQuery.toLowerCase()) ||
				p.keyword.join(' ').toLowerCase().includes(searchTopicQuery.toLowerCase())
		).sort((a, b) => 
      a.title.localeCompare(b.title, undefined, { sensitivity: 'base' })
    )
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
	let renderType = $state<'none' | 'summary' | 'slide'>('none');
	let isLoading = $state(false);
	let appEl: HTMLDivElement | null = null;
	let theme = $state<'light' | 'dark'>('light');
	let readerMode = $state(false);
	let slideWidthPct = $state(100);

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
		if (!readerMode) return;
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.key === 'Escape') readerMode = false;
		};
		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	});

	$effect(() => {
		document.documentElement.setAttribute('data-theme', theme);
		try {
			localStorage.setItem('app-theme', theme);
		} catch {
			// Do nothing.
		}
	});

	// Renderers
	const md = new MarkdownIt({ html: true, linkify: true, typographer: true });
	const marp = new MarpLite({ html: true });

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

	$effect(() => {
		if (!renderHtml) return;
		void typesetMathJax();
	});

	onMount(() => {
		const MOBILE_MEDIA_QUERY = '(max-width: 960px)';

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

		void (async () => {
			try {
				const saved = localStorage.getItem('app-theme');
				if (saved === 'dark' || saved === 'light') theme = saved;
			} catch {
				// Do nothing.
			}

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
			mql.removeEventListener('change', onMediaChange);
		};
	});

	$effect(() => {
		// Disallow more than /{topic}/{paper}.
		if (routeHasExtra) {
			void goto('/', { replaceState: true, noScroll: true, keepFocus: true });
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
				void goto('/', { replaceState: true, noScroll: true, keepFocus: true });
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
				void goto(`/${topic.id}`, { replaceState: true, noScroll: true, keepFocus: true });
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

	function selectPaper(paper: Paper) {
		selectedPaper = paper;
		renderType = 'none';
		renderHtml = '';

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
		
		if (kind === 'arxiv')
			 targetUrl = buildArxivUrl(selectedPaper.url);
		else if (kind === 'ar5iv')
			 targetUrl = buildAr5ivUrl(selectedPaper.url);
		else if (kind === 'pdf')
				targetUrl = buildPdfUrl(selectedPaper.url);

		if (!targetUrl) return;
		
		window.open(targetUrl, '_blank', 'noopener,noreferrer');
	}

	async function selectTopic(topic: Topic, options?: { syncUrl?: boolean }) {
		const syncUrl = options?.syncUrl ?? true;
		selectedTopic = topic;
		selectedPaper = null;
		renderType = 'none';
		renderHtml = '';
		papers = [];
		isLoading = true;

		if (isMobile) mobileStage = 1;

		if (syncUrl) {
			appliedRouteKey = `${topic.id}/`;
			void goto(`/${topic.id}`, { noScroll: true, keepFocus: true });
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
		renderType = type;
		isLoading = true;
		renderHtml = '';

		if (isMobile) mobileStage = 2;

		const filename = type === 'summary' ? paper.summary : paper.slide;
		if (!filename) {
			renderHtml = '<p>No file specified.</p>';
			isLoading = false;
			return;
		}

		if (syncUrl) {
			const slug = paperSlugFromFilename(filename);
			appliedRouteKey = `${selectedTopic.id}/${slug}`;
			void goto(`/${selectedTopic.id}/${slug}`, { noScroll: true, keepFocus: true });
		}

		try {
			const res = await fetch(`/docs/${selectedTopic.id}/${filename}`);
			if (res.ok) {
				const text = await res.text();
				if (type === 'summary') {
					renderHtml = md.render(text);
				} else if (type === 'slide') {
					const { html, css } = marp.render(text);
					renderHtml = `<style>${css}</style><div class="marp-slide-wrapper">${html}</div>`;
				}
			} else {
				renderHtml = `<p>File not found: ${filename}</p>`;
			}
		} catch (e) {
			if (e instanceof Error) {
				renderHtml = `<p>Error loading file: ${e.message}</p>`;
			} else {
				renderHtml = `<p>Unknown error</p>`;
			}
		} finally {
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
			<div class="search-container" onfocusout={(e) => closeSearchPopoverOnFocusOut(e, 'topic')} >
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
					type="button"
					class="toolbar-button"
					disabled={!selectedPaper?.url}
					onclick={() => openExternal('arxiv')}
				>
					arXiv
				</button>
				<button
					type="button"
					class="toolbar-button"
					disabled={!selectedPaper?.url}
					onclick={() => openExternal('pdf')}
				>
					PDF
				</button>
				<button
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
				<select class="theme-select" bind:value={theme} title="Select theme" aria-label="Select theme">
					<option value="light">☀ Light</option>
					<option value="dark">◑ Dark</option>
				</select>
				<button
					type="button"
					class="toolbar-button"
					title="Reader mode"
					aria-pressed={readerMode}
					disabled={!canReader}
					onclick={() => (readerMode = !readerMode)}
				>
					Reader
				</button>
				{#if renderType === 'slide'}
					<label
						class="slide-width-control"
						class:disabled={!canSlideWidth}
						aria-label="Slide width control"
					>
						<span class="slide-width-label">Slide</span>
						<input
							class="slide-width-range"
							type="range"
							min="30"
							max="100"
							step="1"
							bind:value={slideWidthPct}
							disabled={!canSlideWidth}
							aria-label="Slide width (percent of viewport width)"
						/>
						<span class="slide-width-value">{slideWidthPct}%</span>
					</label>
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
			<div class="reader-body">
				{#if renderType === 'summary'}
					<article class="summary-content markdown-body">
						{#if isLoading}
							<p>Loading content...</p>
						{:else}
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html renderHtml}
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
		{:else}
			{#if renderType === 'summary'}
				<div class="preview-content markdown-body">
					<article class="summary-content">
						{#if isLoading}
							<p>Loading content...</p>
						{:else}
							<!-- eslint-disable-next-line svelte/no-at-html-tags -->
							{@html renderHtml}
						{/if}
					</article>
				</div>
			{:else if renderType === 'slide'}
				<div class="preview-content markdown-body slide-content">
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
							</div>
						{:else}
							<div>Select a paper.</div>
						{/if}
					{:else}
						<div>Select a research topic.</div>
					{/if}
				</div>
			{/if}
		{/if}
	</main>
</div>

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

		--select-bg: #1c1f2a;
		--select-border: rgba(255, 255, 255, 0.12);
		--select-color: #9097b0;

		/* Slide background */
		--slide-wrapper-bg: #0f1117;
	}

	/* Reset / Base */
	:global(body) {
		margin: 0;
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
		font-size: 14px;
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
  	-webkit-box-orient: vertical;
  	-webkit-line-clamp: 3;
	}

	.placeholder .clamp-5 {
		overflow: hidden;
		display: -webkit-box;
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

	/* Theme Select */
	.theme-select {
		appearance: none;
		-webkit-appearance: none;
		border: 1px solid var(--select-border);
		border-radius: 999px;
		background: var(--select-bg);
		color: var(--select-color);
		font: inherit;
		font-size: 0.72rem;
		font-weight: 600;
		padding: 0.2rem 1.8rem 0.2rem 0.75rem;
		min-height: 28px;
		cursor: pointer;
		/* chevron arrow via background-image */
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%239097b0' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
		background-repeat: no-repeat;
		background-position: right 0.6rem center;
		transition:
			border-color 0.15s,
			background-color 0.15s,
			color 0.15s;
	}
	.theme-select:hover {
		border-color: var(--accent-dim);
		color: var(--accent-text);
	}
	.theme-select:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent-subtle);
	}
	/* Light theme: darker arrow */
	:global([data-theme='light']) .theme-select {
		background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%234a5070' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
	}

	/* Preview Content */
	.preview-content {
		flex: 1;
		overflow-y: auto;
		background: var(--bg-base);
		-webkit-overflow-scrolling: touch;
		touch-action: pan-y;
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
		.preview-toolbar button[aria-pressed] {
			display: none;
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
			margin: 0.5rem;
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
		:global(body) {
			background: white;
		}
		.app-container {
			display: block;
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
			width: 100%;
			height: auto;
			overflow: visible;
			background: white;
			border: none;
		}
		.summary-content {
			box-shadow: none;
			border: none;
			padding: 0;
			margin: 0 auto;
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

