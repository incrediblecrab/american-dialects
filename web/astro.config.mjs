// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";

/**
 * The site is published to https://incrediblecrab.github.io/american-dialects/,
 * so every asset URL has to carry that prefix. Anything that builds a URL at
 * runtime must go through import.meta.env.BASE_URL rather than assuming "/".
 *
 * Output is static. Nothing here needs a server: the model runs in the browser
 * and every number is baked in at build time from generated.json.
 *
 * Routing is plain file routing over src/pages. There was a docs theme here
 * before, and it was the wrong shape: a sidebar tree, a search field and a
 * per-page heading rail are for a reader looking something up, not for one
 * being walked through an argument. Each page is a chapter of one essay, so
 * they are pages, and the shell they share is src/layouts/Base.astro.
 */
export default defineConfig({
  site: "https://incrediblecrab.github.io",
  base: "/american-dialects/",
  output: "static",
  integrations: [mdx(), react()],
  build: {
    assets: "assets",
  },
  vite: {
    build: {
      assetsInlineLimit: 0,
    },
  },
});