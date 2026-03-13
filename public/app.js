const state = {
  topics: [],
  currentTopic: null,
  papers: [],
  currentPaper: null,
  showTopicPanel: true,
  showPaperPanel: true,
};

let slideFitScheduled = false;

const els = {
  topicList: document.getElementById('topic-list'),
  topicCount: document.getElementById('topic-count'),
  paperList: document.getElementById('paper-list'),
  paperCount: document.getElementById('paper-count'),
  app: document.getElementById('app'),
  paperPanelTitle: document.getElementById('paper-panel-title'),
  viewerTitle: document.getElementById('viewer-title'),
  viewerMeta: document.getElementById('viewer-meta'),
  pdfFrame: document.getElementById('pdf-frame'),
  pdfEmpty: document.getElementById('pdf-empty'),
  summaryContent: document.getElementById('summary-content'),
  slideContent: document.getElementById('slide-content'),
  openArxiv: document.getElementById('open-arxiv'),
  openPdf: document.getElementById('open-pdf'),
  promptPopover: document.getElementById('prompt-popover'),
  promptBackdrop: document.getElementById('prompt-backdrop'),
  promptEyebrow: document.getElementById('prompt-popover-eyebrow'),
  promptTitle: document.getElementById('prompt-popover-title'),
  promptText: document.getElementById('prompt-popover-text'),
  copyPromptButton: document.getElementById('copy-prompt-button'),
  closePromptButton: document.getElementById('close-prompt-button'),
  toggleTopicPanel: document.getElementById('toggle-topic-panel'),
  togglePaperPanel: document.getElementById('toggle-paper-panel'),
};

const SUMMARY_PROMPT_TEMPLATE = `모든 문서는 UTF-8로 인코딩되어 있다.
먼저 \`docs\` 폴더를 작업 기준 폴더로 사용하라.
\`README.md\` 파일을 읽고 프로젝트의 성격을 이해하라.
\`TOPIC.md\` 파일을 읽고 연구 분야와 관련 폴더의 경로를 파악하라.
\`RULE.md\` 파일을 읽고 문서 작성 규칙과 보고서 작성 규칙 등을 숙지하라.
\`RULE_SEARCH.md\` 파일을 읽고 arXiv 논문의 검색 규칙을 숙지하라.

이제부터 \`docs/[folder_name]\` 폴더에서 작업하겠다.

\`[논문명]\` 논문의 상세 분석 보고서를 작성하라.`;

const SLIDE_PROMPT_TEMPLATE = `모든 문서는 UTF-8로 인코딩되어 있다.
먼저 \`docs\` 폴더를 작업 기준 폴더로 사용하라.
\`README.md\` 파일을 읽고 프로젝트의 성격을 이해하라.
\`TOPIC.md\` 파일을 읽고 연구 분야와 관련 폴더의 경로를 파악하라.
\`RULE.md\` 파일을 읽고 문서 작성 규칙과 보고서 작성 규칙 등을 숙지하라.
\`RULE_SEARCH.md\` 파일을 읽고 arXiv 논문의 검색 규칙을 숙지하라.

이제부터 \`docs/[folder_name]\` 폴더에서 작업하겠다.

\`[논문명]\` 논문의 발표자료를 작성하라.`;

function setActionLink(link, href) {
  if (href) {
    link.href = href;
    link.setAttribute('aria-disabled', 'false');
  } else {
    link.removeAttribute('href');
    link.setAttribute('aria-disabled', 'true');
  }
}

function paperClass(paper) {
  if (paper.slide) return 'blue';
  if (paper.summary) return 'black';
  return 'gray';
}

function buildArxivUrl(url) {
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

function buildPrompt(template) {
  if (!state.currentTopic || !state.currentPaper) return '';
  return template
    .replace('[folder_name]', state.currentTopic.id)
    .replace('[논문명]', state.currentPaper.title);
}

function closePromptPopover() {
  els.promptPopover.hidden = true;
}

function openPromptPopover(kind) {
  const isSlide = kind === 'slide';
  els.promptEyebrow.textContent = isSlide ? 'Slide Prompt' : 'Summary Prompt';
  els.promptTitle.textContent = isSlide
    ? '발표자료 생성 프롬프트'
    : '논문 분석 보고서 생성 프롬프트';
  els.promptText.textContent = buildPrompt(
    isSlide ? SLIDE_PROMPT_TEMPLATE : SUMMARY_PROMPT_TEMPLATE,
  );
  els.promptPopover.hidden = false;
}

function renderEmptyState(container, className, message, promptConfig = null) {
  container.className = className;
  container.innerHTML = '';

  const text = document.createElement('p');
  text.className = 'empty-state-text';
  text.textContent = message;
  container.appendChild(text);

  if (!promptConfig) return;

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'prompt-trigger-button';
  button.textContent = promptConfig.label;
  button.addEventListener('click', () => openPromptPopover(promptConfig.kind));
  container.appendChild(button);
}

function renderSummaryEmptyState(message, showPromptButton = false) {
  renderEmptyState(
    els.summaryContent,
    'markdown-body empty-state',
    message,
    showPromptButton
      ? { label: '논문 분석 보고서 생성 프롬프트', kind: 'summary' }
      : null,
  );
}

function renderSlideEmptyState(message, showPromptButton = false) {
  renderEmptyState(
    els.slideContent,
    'slide-html empty-state',
    message,
    showPromptButton
      ? { label: '발표자료 생성 프롬프트', kind: 'slide' }
      : null,
  );
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

async function renderMath(container) {
  if (!window.MathJax || !window.MathJax.typesetPromise) return;
  await window.MathJax.typesetPromise([container]);
}

function fitSlidesToView() {
  const firstSlide = els.slideContent.querySelector('svg[data-marpit-svg]');
  if (!firstSlide || !els.slideContent.clientWidth || !els.slideContent.clientHeight) return;

  const viewBox = firstSlide.getAttribute('viewBox');
  let slideWidth = 0;
  let slideHeight = 0;

  if (viewBox) {
    const values = viewBox.split(/[\s,]+/).map(Number);
    if (values.length === 4 && values.every(Number.isFinite)) {
      slideWidth = values[2];
      slideHeight = values[3];
    }
  }

  if (!slideWidth || !slideHeight) {
    const widthAttr = Number(firstSlide.getAttribute('width'));
    const heightAttr = Number(firstSlide.getAttribute('height'));
    if (Number.isFinite(widthAttr) && Number.isFinite(heightAttr) && widthAttr > 0 && heightAttr > 0) {
      slideWidth = widthAttr;
      slideHeight = heightAttr;
    }
  }

  if (!slideWidth || !slideHeight) return;

  const availableWidth = Math.max(0, els.slideContent.clientWidth - 64);
  const availableHeight = Math.max(0, els.slideContent.clientHeight - 64);
  const scale = Math.max(0.1, Math.min(availableWidth / slideWidth, availableHeight / slideHeight));

  els.slideContent.style.setProperty('--slide-base-width', `${slideWidth}px`);
  els.slideContent.style.setProperty('--slide-scale', String(scale));
}

function scheduleSlideFit() {
  if (slideFitScheduled) return;
  slideFitScheduled = true;
  window.requestAnimationFrame(() => {
    slideFitScheduled = false;
    fitSlidesToView();
  });
}

function renderTopics() {
  els.topicCount.textContent = `${state.topics.length} topics`;
  els.topicList.innerHTML = '';
  for (const topic of state.topics) {
    const li = document.createElement('li');
    li.className = 'item' + (state.currentTopic && state.currentTopic.id === topic.id ? ' active' : '');
    li.innerHTML = `
      <div class="paper-title">${topic.title}</div>
      <div class="topic-keywords">${topic.keyword.join(', ')}</div>
    `;
    li.addEventListener('click', () => selectTopic(topic.id));
    els.topicList.appendChild(li);
  }
}

function renderColumnVisibility() {
  els.app.dataset.showTopic = String(state.showTopicPanel);
  els.app.dataset.showPaper = String(state.showPaperPanel);
  els.toggleTopicPanel.setAttribute('aria-pressed', String(state.showTopicPanel));
  els.togglePaperPanel.setAttribute('aria-pressed', String(state.showPaperPanel));
}

function renderPapers() {
  els.paperList.innerHTML = '';
  if (!state.currentTopic) {
    els.paperPanelTitle.textContent = 'Papers';
    els.paperCount.textContent = 'Select a topic.';
    return;
  }

  els.paperPanelTitle.textContent = state.currentTopic.title;
  els.paperCount.textContent = `${state.papers.length} papers`;
  for (const paper of state.papers) {
    const isActive = state.currentPaper && state.currentPaper.index === paper.index;
    const li = document.createElement('li');
    li.className = `item paper-item ${paperClass(paper)}${isActive ? ' active' : ''}`;
    li.innerHTML = `
      <div class="paper-title">${paper.title}</div>
      <div class="paper-meta">${paper.author} <span class="paper-year">(${paper.year})</span></div>
    `;
    li.addEventListener('click', () => selectPaper(paper.index));
    els.paperList.appendChild(li);
  }
}

async function loadTopics() {
  const data = await fetchJson('/api/topics');
  state.topics = data.topics;
  state.currentTopic = data.topics[0] || null;
  renderTopics();
  if (state.currentTopic) await loadPapers(state.currentTopic.id);
}

async function loadPapers(topicId) {
  const data = await fetchJson(`/api/papers?topic=${encodeURIComponent(topicId)}`);
  state.papers = data.papers;
  state.currentPaper = state.papers[0] || null;
  renderPapers();
  await renderCurrentPaper();
}

async function selectTopic(topicId) {
  if (state.currentTopic && state.currentTopic.id === topicId) return;
  state.currentTopic = state.topics.find((topic) => topic.id === topicId) || null;
  renderTopics();
  await loadPapers(topicId);
}

async function selectPaper(index) {
  state.currentPaper = state.papers.find((paper) => paper.index === index) || null;
  renderPapers();
  await renderCurrentPaper();
}

async function renderCurrentPaper() {
  const paper = state.currentPaper;
  if (!paper) {
    els.viewerTitle.textContent = 'Select a paper';
    els.viewerMeta.textContent = 'Choose a topic on the left, then a paper in the middle column.';
    els.pdfFrame.hidden = true;
    els.pdfEmpty.hidden = false;
    renderSummaryEmptyState('Select a paper with a summary.');
    renderSlideEmptyState('Select a paper with a slide deck.');
    setActionLink(els.openArxiv, '');
    setActionLink(els.openPdf, '');
    closePromptPopover();
    return;
  }

  els.viewerTitle.textContent = paper.title;
  els.viewerMeta.textContent = `${paper.author} | ${paper.year}`;

  const pdfUrl = paper.url
    ? (paper.url.includes('/pdf/') ? paper.url : `${paper.url.replace('/abs/', '/pdf/')}.pdf`)
    : '';
  const arxivUrl = buildArxivUrl(paper.url);

  if (pdfUrl) {
    els.pdfFrame.hidden = false;
    els.pdfEmpty.hidden = true;
    els.pdfFrame.src = pdfUrl;
  } else {
    els.pdfFrame.removeAttribute('src');
    els.pdfFrame.hidden = true;
    els.pdfEmpty.hidden = false;
  }
  setActionLink(els.openArxiv, arxivUrl);
  setActionLink(els.openPdf, pdfUrl);

  if (paper.summary) {
    try {
      const html = await fetch(`/api/render-summary?topic=${encodeURIComponent(state.currentTopic.id)}&file=${encodeURIComponent(paper.summary)}`).then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.text();
      });
      els.summaryContent.className = 'markdown-body';
      els.summaryContent.innerHTML = html;
      await renderMath(els.summaryContent);
      closePromptPopover();
    } catch (error) {
      renderSummaryEmptyState(`Failed to load summary: ${error.message}`);
      closePromptPopover();
    }
  } else {
    renderSummaryEmptyState(
      'No summary markdown file is linked for this paper.',
      true,
    );
  }

  if (paper.slide) {
    try {
      const html = await fetch(`/api/render-slide?topic=${encodeURIComponent(state.currentTopic.id)}&file=${encodeURIComponent(paper.slide)}`).then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.text();
      });
      els.slideContent.className = 'slide-html';
      els.slideContent.innerHTML = html;
      scheduleSlideFit();
    } catch (error) {
      renderSlideEmptyState(`Failed to render slide deck: ${error.message}`);
    }
  } else {
    renderSlideEmptyState(
      'No slide markdown file is linked for this paper.',
      true,
    );
  }
}

function setupTabs() {
  const buttons = document.querySelectorAll('.tab-button');
  const panels = document.querySelectorAll('.tab-panel');
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const tab = button.dataset.tab;
      buttons.forEach((item) => item.classList.toggle('active', item === button));
      panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === tab));
      if (tab === 'slide') scheduleSlideFit();
    });
  }
}

function setupSlideFit() {
  const observer = new ResizeObserver(() => {
    scheduleSlideFit();
  });
  observer.observe(els.slideContent);
  window.addEventListener('resize', scheduleSlideFit);
  scheduleSlideFit();
}

function setupSplitters() {
  const min1 = 180;
  const min2 = 240;
  const min3 = 320;

  function applyResize(which, clientX) {
    if (window.innerWidth <= 960) return;
    if (which === 'left' && !state.showTopicPanel) return;
    if (which === 'middle' && !state.showPaperPanel) return;
    const total = els.app.clientWidth;
    const styles = getComputedStyle(document.documentElement);
    const currentCol1 = parseFloat(styles.getPropertyValue('--col1'));
    if (which === 'left') {
      const newCol1 = Math.max(min1, Math.min(clientX, total - min2 - min3 - 12));
      document.documentElement.style.setProperty('--col1', `${newCol1}px`);
    } else {
      const leftEdge = currentCol1 + 6;
      const newCol2 = Math.max(min2, Math.min(clientX - leftEdge, total - currentCol1 - min3 - 12));
      document.documentElement.style.setProperty('--col2', `${newCol2}px`);
    }
  }

  document.querySelectorAll('.splitter').forEach((splitter) => {
    splitter.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      const which = splitter.dataset.resize;
      const move = (moveEvent) => applyResize(which, moveEvent.clientX);
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
  });
}

function setupColumnToggles() {
  renderColumnVisibility();

  els.toggleTopicPanel.addEventListener('click', () => {
    state.showTopicPanel = !state.showTopicPanel;
    renderColumnVisibility();
  });

  els.togglePaperPanel.addEventListener('click', () => {
    state.showPaperPanel = !state.showPaperPanel;
    renderColumnVisibility();
  });
}

function setupPromptPopover() {
  els.closePromptButton.addEventListener('click', closePromptPopover);
  els.promptBackdrop.addEventListener('click', closePromptPopover);
  els.copyPromptButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(els.promptText.textContent || '');
      els.copyPromptButton.textContent = '복사됨';
      window.setTimeout(() => {
        els.copyPromptButton.textContent = '클립보드로 복사';
      }, 1500);
    } catch (error) {
      els.copyPromptButton.textContent = '복사 실패';
      window.setTimeout(() => {
        els.copyPromptButton.textContent = '클립보드로 복사';
      }, 1500);
    }
  });
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !els.promptPopover.hidden) {
      closePromptPopover();
    }
  });
}

setupTabs();
setupSlideFit();
setupSplitters();
setupColumnToggles();
setupPromptPopover();
loadTopics().catch((error) => {
  els.viewerTitle.textContent = 'Failed to load viewer';
  els.viewerMeta.textContent = error.message;
});
