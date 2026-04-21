/// <reference types="node" />

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");

function help() {
  console.log("Usage: node scripts/generate-manifest.js");
  console.log("");
  console.log("Scans static/docs for topic metadata.json files and writes static/docs/manifest.json.");
}

function generateManifest() {
  const docsDir = path.join(projectRoot, "static", "docs");
  const manifestPath = path.join(projectRoot, "static", "docs", "manifest.json");

  if (!fs.existsSync(docsDir)) {
    console.error(`Error: Docs directory not found at ${docsDir}`);
    process.exit(1);
  }

  console.log(`Scanning ${docsDir} to generate manifest...`);

  const manifest = [];
  const items = fs.readdirSync(docsDir, { withFileTypes: true });

  for (const item of items) {
    if (item.isDirectory()) {
      const metaPath = path.join(docsDir, item.name, "metadata.json");

      if (fs.existsSync(metaPath)) {
        try {
          const metadata = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
          manifest.push({ id: item.name, ...metadata });
        } catch (e) {
          console.error(`Failed to parse ${metaPath}: ${e.message}`);
        }
      }
    }
  }

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(
    `Successfully created manifest at ${manifestPath} with ${manifest.length} items.`,
  );
}

function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    help();
    return;
  }

  generateManifest();
}

main();
