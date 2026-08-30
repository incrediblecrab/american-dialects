// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";

/**
 * The site is published to https://incrediblecrab.github.io/american-dialects/,
 * so every asset URL has to carry that prefix. Anything that builds a URL at
 * runtime must go through import.meta.env.BASE_URL rather than assuming "/".
 *
 * Output is static. Nothing on this page needs a server: the model runs in the
 * browser and every number is baked in at build time from generated.json.
 *
 * The prose is authored in MDX. The essay is the artifact here, so it is
 * written as Markdown and the interactive parts are imported into it, rather
 * than the prose living as JSX inside components that only ever printed it.
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
