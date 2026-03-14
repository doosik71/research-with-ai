<script lang="ts">
	import { onMount, tick } from 'svelte';
	import MarkdownIt from 'markdown-it';
	import { MarpLite } from './marp-lite';

	type Topic = {
		id: string;
		title: string;
		keyword: string[];
	};

	type Paper = {
		title: string;
		author: string;
		year: string;
		url: string;
		summary: string;
		slide: string;
		index: number;
	};
	let topics = $state<Topic[]>([]);
	let papers = $state<Paper[]>([]);
	let selectedTopic = $state<Topic | null>(null);
	let selectedPaper = $state<Paper | null>(null);
	let showTopicPanel = $state(true);
	let showPaperPanel = $state(true);
	let previewTitle = $state('No paper selected');

	// Preview State
	let renderHtml = $state('');
	let renderType = $state('none'); // 'none', 'summary', 'slide'
	let isLoading = $state(false);
	let appEl: HTMLDivElement | null = null;

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

		console.log(renderHtml);
		void typesetMathJax();
	});

	onMount(async () => {
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
		if (paper.slide) return 'paper-blue';
		if (paper.summary) return 'paper-black';
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
		previewTitle = paper?.title || 'No paper selected';
	}

	function openExternal(kind: 'arxiv' | 'pdf') {
		if (!selectedPaper?.url) return;
		const targetUrl =
			kind === 'arxiv' ? buildArxivUrl(selectedPaper.url) : buildPdfUrl(selectedPaper.url);
		if (!targetUrl) return;
		window.open(targetUrl, '_blank', 'noopener,noreferrer');
	}

	async function selectTopic(topic: Topic) {
		selectedTopic = topic;
		selectedPaper = null;
		previewTitle = 'No paper selected';
		renderType = 'none';
		renderHtml = '';
		papers = [];
		isLoading = true;

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
					.filter((p) => p);
			}
		} catch (e) {
			console.error('Error fetching paper list:', e);
		} finally {
			isLoading = false;
		}
	}

	async function loadPreview(paper: Paper, type: 'summary' | 'slide') {
		if (!selectedTopic) return;

		selectedPaper = paper;
		previewTitle = paper.title || 'Select a paper';
		renderType = type;
		isLoading = true;
		renderHtml = '';

		const filename = type === 'summary' ? paper.summary : paper.slide;
		if (!filename) {
			renderHtml = '<p>No file specified.</p>';
			isLoading = false;
			return;
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
				fontCache: 'global'
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
	bind:this={appEl}
>
	<!-- 1. Topic List -->
	<div class="topic-list">
		<h2>
			<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 576 512"
				><path
					fill="currentColor"
					d="M97.5 400l50-160 379.4 0-50 160-379.4 0zm190.7 48L477 448c21 0 39.6-13.6 45.8-33.7l50-160c9.7-30.9-13.4-62.3-45.8-62.3l-379.4 0c-21 0-39.6 13.6-45.8 33.7L80.2 294.4 80.2 96c0-8.8 7.2-16 16-16l138.7 0c3.5 0 6.8 1.1 9.6 3.2L282.9 112c13.8 10.4 30.7 16 48 16l117.3 0c8.8 0 16 7.2 16 16l48 0c0-35.3-28.7-64-64-64L330.9 80c-6.9 0-13.7-2.2-19.2-6.4L273.3 44.8C262.2 36.5 248.8 32 234.9 32L96.2 32c-35.3 0-64 28.7-64 64l0 288c0 35.3 28.7 64 64 64l192 0z"
				/></svg
			>
			Topics
		</h2>
		<div class="topic-content">
			<ul>
				{#each topics as topic (topic.id)}
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
		<h2>
			<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 512 512"
				><path
					fill="currentColor"
					d="M40 48C26.7 48 16 58.7 16 72l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24L40 48zM192 64c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32L192 64zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zm0 160c-17.7 0-32 14.3-32 32s14.3 32 32 32l288 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-288 0zM16 232l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0c-13.3 0-24 10.7-24 24zM40 368c-13.3 0-24 10.7-24 24l0 48c0 13.3 10.7 24 24 24l48 0c13.3 0 24-10.7 24-24l0-48c0-13.3-10.7-24-24-24l-48 0z"
				/></svg
			>
			Papers {selectedTopic ? `(${papers.length})` : ''}
		</h2>
		{#if isLoading && papers.length === 0}
			<div class="loading">Loading...</div>
		{/if}
		<div class="list-content">
			{#each papers as paper (paper.index)}
				<button
					class={`paper-item ${paperTone(paper)}`}
					class:selected={selectedPaper?.index === paper.index}
					onclick={() => selectPaper(paper)}
				>
					<p class="title">{paper.title}</p>
					<p class="meta">{paper.author} ({paper.year})</p>
				</button>
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
		<div class="preview-header">
			<!-- <div class="preview-header-top">
				<h2>{previewTitle}</h2>
			</div> -->
			<div class="preview-toolbar">
				<button
					type="button"
					class="toolbar-button"
					title="분석 보고서 보기"
					disabled={!selectedPaper?.summary}
					onclick={() => {
						if (selectedPaper) loadPreview(selectedPaper, 'summary');
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
					title="발표자료 보기"
					disabled={!selectedPaper?.slide}
					onclick={() => {
						if (selectedPaper) loadPreview(selectedPaper, 'slide');
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
					title="토픽 보이기/숨기기"
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
					title="목록 보이기/숨기기"
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
				<button type="button" class="toolbar-button" title="도움말" disabled> ? </button>
			</div>
		</div>
		{#if renderType === 'summary'}
			<div class="preview-content markdown-body">
				<div class="summary-content">
					{#if isLoading}
						<p>Loading content...</p>
					{:else}
						<!-- eslint-disable-next-line svelte/no-at-html-tags -->
						{@html renderHtml}
					{/if}
				</div>
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
				<span
					>Select a
					<button
						type="button"
						class="toolbar-button"
						title="분석 보고서 보기"
						disabled={!selectedPaper?.summary}
						onclick={() => {
							if (selectedPaper) loadPreview(selectedPaper, 'summary');
						}}
					>
						<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 512 512"
							><path
								fill="currentColor"
								d="M168 80c-13.3 0-24 10.7-24 24l0 304c0 8.4-1.4 16.5-4.1 24L440 432c13.3 0 24-10.7 24-24l0-304c0-13.3-10.7-24-24-24L168 80zM72 480c-39.8 0-72-32.2-72-72L0 112C0 98.7 10.7 88 24 88s24 10.7 24 24l0 296c0 13.3 10.7 24 24 24s24-10.7 24-24l0-304c0-39.8 32.2-72 72-72l272 0c39.8 0 72 32.2 72 72l0 304c0 39.8-32.2 72-72 72L72 480zM192 152c0-13.3 10.7-24 24-24l48 0c13.3 0 24 10.7 24 24l0 48c0 13.3-10.7 24-24 24l-48 0c-13.3 0-24-10.7-24-24l0-48zm152 24l48 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-48 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zM216 256l176 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-176 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zm0 80l176 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-176 0c-13.3 0-24-10.7-24-24s10.7-24 24-24z"
							/></svg
						>
					</button>
					or
					<button
						type="button"
						class="toolbar-button"
						title="발표자료 보기"
						disabled={!selectedPaper?.slide}
						onclick={() => {
							if (selectedPaper) loadPreview(selectedPaper, 'slide');
						}}
						><svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 512 512"
							><path
								fill="currentColor"
								d="M448 96l0 256-384 0 0-256 384 0zM64 32C28.7 32 0 60.7 0 96L0 352c0 35.3 28.7 64 64 64l144 0-16 48-72 0c-13.3 0-24 10.7-24 24s10.7 24 24 24l272 0c13.3 0 24-10.7 24-24s-10.7-24-24-24l-72 0-16-48 144 0c35.3 0 64-28.7 64-64l0-256c0-35.3-28.7-64-64-64L64 32z"
							/></svg
						>
					</button> to preview</span
				>
			</div>
		{/if}
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		font-family: sans-serif;
		height: 100vh;
		overflow: hidden;
	}
	:global(:root) {
		--col1: 250px;
		--col2: 350px;
	}

	.app-container {
		display: grid;
		grid-template-columns: minmax(180px, var(--col1)) 6px minmax(240px, var(--col2)) 6px minmax(
				320px,
				1fr
			);
		height: 100vh;
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

	.topic-list,
	.paper-list {
		border-right: 1px solid #ddd;
		display: flex;
		flex-direction: column;
		background: #fff;
		height: 100vh;
	}

	.topic-list h2,
	.paper-list h2 {
		padding: 0.5rem;
		margin: 0;
		background-color: Linen;
		border-bottom: 1px solid lightgray;
		font-size: 1.2rem;
	}

	.topic-list ul {
		list-style: none;
		padding: 0;
		margin: 0;
	}
	.topic-list button {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.2rem 0.5rem;
		margin: 0.2rem 0;
		border: none;
		background: none;
		cursor: pointer;
		border-radius: 4px;
	}
	.topic-list button:hover {
		background: #e4e4e7;
	}
	.topic-list button.active {
		background: #e0e7ff;
		color: #3730a3;
		font-weight: bold;
		border-left: 3px solid dodgerblue;
		border-right: 3px solid dodgerblue;
	}
	.topic-content,
	.list-content {
		overflow-y: scroll;
		flex: 1;
		padding: 0.5rem;
	}
	.paper-item {
		margin: 0.2rem 0;
		padding: 0.4rem;
		width: 100%;
		border-top: 1px solid #eee;
		border-right: 1px solid #eee;
		border-bottom: 1px solid #eee;
		border-left: 4px solid transparent;
		border-radius: 0.4rem;
		cursor: pointer;
	}
	.paper-item.selected {
		background: #f0f9ff;
		border-top: 1px gray solid;
		border-bottom: 1px gray solid;
		border-right: 1px gray solid;
	}
	.paper-item.paper-gray {
		color: #808080;
		border-left-color: #c2c2c2;
	}
	.paper-item.paper-black {
		color: #171717;
		border-left-color: #303030;
	}
	.paper-item.paper-blue {
		color: dodgerblue;
		border-left-color: dodgerblue;
	}
	.paper-item .title {
		margin: 0 0 0.3rem 0;
		font-size: 1rem;
		font-weight: bold;
	}
	.paper-item .meta {
		margin: 0;
		color: #666;
		font-size: 0.85rem;
	}
	.paper-item:hover {
		border-left: 1px solid gray;
		border-right: 4px solid gray;
		/* border-color: orange; */
		background-color: white;
	}
	.actions {
		display: flex;
		gap: 0.5rem;
	}

	button svg {
		margin-top: 0.2rem;
	}

	.preview {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: #fff;
	}

	.preview-header {
		padding: 0.5rem;
		border-bottom: 1px solid lightgray;
		background-color: Linen;
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
		gap: 0.5rem;
		/* margin: 0; */
	}
	.toolbar-button {
		border: 1px solid #d1d5db;
		border-radius: 999px;
		background: #fff;
		color: #374151;
		font: inherit;
		font-size: 0.75rem;
		font-weight: 700;
		padding: 0.1rem 0.8rem;
		cursor: pointer;
		transition:
			background 0.18s ease,
			color 0.18s ease,
			border-color 0.18s ease,
			transform 0.12s ease;
	}
	.toolbar-button:hover {
		transform: translateY(-1px);
		border-color: rgba(30, 110, 255, 0.32);
		color: #111827;
	}
	.toolbar-button:disabled {
		opacity: 0.45;
		cursor: not-allowed;
		transform: none;
	}
	.preview-content {
		flex: 1;
		overflow-y: auto;
		padding: 0;
		background-color: lightgray;
	}

	.preview-content .summary-content {
		max-width: 60rem;
		padding: 2rem 4rem 4rem;
		margin: 0 auto;
		background-color: white;
		filter: drop-shadow(3px 3px 4px rgba(0, 0, 0, 0.6));
		border: lightgray 1px solid;
	}

	.marp-slide-wrapper {
		padding: 0.2rem;
	}

	.placeholder {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 1rem;
		height: 100%;
		color: #999;
	}
	.placeholder-header {
		align-self: flex-end;
		padding: 0 1rem;
	}
	.loading {
		padding: 1rem;
		color: #666;
		text-align: center;
	}

	.splitter {
		background: #eee;
		cursor: col-resize;
		position: relative;
		border: none;
		padding: 0;
		width: 0.4rem;
	}
	.splitter::after {
		content: '';
		position: absolute;
		inset: 0;
		margin: auto;
		width: 3px;
		height: 72px;
		border-radius: 999px;
		background: rgba(70, 65, 58, 0.38);
	}
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

	@media (max-width: 1400px) {
		.preview-content .summary-content {
			padding: 2rem;
		}
	}

	@media (max-width: 960px) {
		.app-container {
			grid-template-columns: 1fr;
			grid-template-rows: 28vh 30vh 1fr;
		}

		.topic-list,
		.left-splitter,
		.paper-list,
		.middle-splitter,
		.preview {
			grid-column: 1;
		}

		.topic-list {
			grid-row: 1;
			border-right: 0;
			border-bottom: 1px solid #ddd;
		}

		.paper-list {
			grid-row: 2;
			border-right: 0;
			border-bottom: 1px solid #ddd;
		}

		.preview {
			grid-row: 3;
		}

		.app-container[data-show-topic='false'][data-show-paper='true'] {
			grid-template-columns: 1fr;
			grid-template-rows: 0 30vh 1fr;
		}
		.app-container[data-show-topic='true'][data-show-paper='false'] {
			grid-template-columns: 1fr;
			grid-template-rows: 28vh 0 1fr;
		}
		.app-container[data-show-topic='false'][data-show-paper='false'] {
			grid-template-columns: 1fr;
			grid-template-rows: 0 0 1fr;
		}

		.splitter {
			display: none;
		}

		.preview-content .summary-content {
			padding: 1rem;
		}
	}

	@media print {
		/* 1. 나머지 컬럼을 완전히 제거 */
		.app-container {
			display: block; /* grid 해제 → .preview가 자동으로 전체 폭 차지 */
		}

		/* 2. 숨길 패널들 */
		.topic-list,
		.left-splitter,
		.paper-list,
		.middle-splitter,
		.preview-header {
			display: none; /* visibility:hidden 대신 display:none → 공간도 완전히 제거 */
		}

		/* 3. 프리뷰 전체 폭 */
		.preview {
			width: 100%;
			height: auto;
			overflow: visible; /* 인쇄 시 잘리지 않게 */
			background-color: white;
			border: none;
		}

		.preview-content .summary-content {
			filter: none;
			border: 1px white solid;
			padding: 0;
		}

		:global(div.marp-slide-wrapper) {
			padding: 0.2rem;
			background-color: white;
		}

		:global(div.marp-slide-wrapper .marp-svg) {
			margin: 2rem 0 4rem;
		}
	}
</style>
