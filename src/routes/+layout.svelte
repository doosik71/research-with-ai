<script lang="ts">
	import 'pretendard/dist/web/static/pretendard.css';
  import '@fontsource-variable/noto-serif-kr';
	import favicon from '$lib/assets/favicon.ico';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}
