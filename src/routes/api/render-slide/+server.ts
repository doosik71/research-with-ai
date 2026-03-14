import { Marp } from '@marp-team/marp-core';
import { json } from '@sveltejs/kit';

const marp = new Marp({ html: true, script: false });

/**
 * Handles POST requests to render markdown slides.
 * @param {{ request: Request }} param
 * @returns {Promise<Response>}
 */
export async function POST({ request }): Promise<Response> {
	try {
		const { markdown } = await request.json();
		if (typeof markdown !== 'string') {
			return json({ error: 'Markdown content must be a string.' }, { status: 400 });
		}
		const { html, css } = marp.render(markdown);
		return json({ html, css });
	} catch (error) {
		const details = error instanceof Error ? error.message : 'Unknown error';
		return json({ error: 'Failed to render slide.', details }, { status: 500 });
	}
}
