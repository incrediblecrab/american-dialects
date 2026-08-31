import { defineCollection } from "astro:content";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

/**
 * Starlight reads every page from src/content/docs/.
 *
 * The generated figures live one directory up, in src/content/index.ts and
 * src/content/generated.json, which the loader does not glob. That is
 * deliberate: those are the single source of truth for every number the site
 * prints, they are emitted by model/export_web.py, and they are not pages.
 */
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
