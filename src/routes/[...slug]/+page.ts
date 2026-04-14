import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ params }) => {
	const segments = params.slug?.split('/').filter(Boolean) ?? [];

	if (segments.length > 2) {
		throw error(404, 'Not found');
	}

	return {};
};
