/// <reference types="node" />

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(projectRoot, "static", "docs");
const reportsRoot = path.join(docsRoot, "reports");

async function shouldCopy(sourcePath, targetPath) {
	try {
		const [sourceStat, targetStat] = await Promise.all([
			fs.stat(sourcePath),
			fs.stat(targetPath),
		]);
		return sourceStat.mtimeMs > targetStat.mtimeMs;
	} catch (error) {
		if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
			try {
				await fs.access(sourcePath);
				return true;
			} catch {
				return false;
			}
		}

		throw error;
	}
}

async function main() {
	const entries = await fs.readdir(docsRoot, { withFileTypes: true });

	for (const entry of entries) {
		if (!entry.isDirectory()) {
			continue;
		}

		const sourcePath = path.join(docsRoot, entry.name, "report-12", "final-report.md");
		const targetPath = path.join(reportsRoot, `${entry.name}.md`);

		if (!(await shouldCopy(sourcePath, targetPath))) {
			continue;
		}

		await fs.copyFile(sourcePath, targetPath);
		console.log(entry.name);
	}
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
