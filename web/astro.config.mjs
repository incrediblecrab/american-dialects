// @ts-check
import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";

import starlight from "@astrojs/starlight";

/**
 * The site is published to https://incrediblecrab.github.io/american-dialects/,
 * so every asset URL has to carry that prefix. Anything that builds a URL at
 * runtime must go through import.meta.env.BASE_URL rather than assuming "/".
 *
 * Output is static. Nothing here needs a server: the model runs in the browser
 * and every number is baked in at build time from generated.json.
 *
 * Starlight owns the routing. The essay is a set of pages in the docs
 * collection rather than one scrolling column, which is what gives the sidebar
 * a real tree and each act its own "On this page" rail. The interactive parts
 * are React islands imported into the MDX, so the prose stays prose.
 *
 * tokens.css has to load through customCss because the islands read its
 * variables for type and colour; Starlight's own theme is layered under it.
 */
export default defineConfig({
  site: "https://incrediblecrab.github.io",
  base: "/american-dialects/",
  output: "static",
  integrations: [
    starlight({
      title: "American dialects",
      description:
        "The Harvard Dialect Survey published its maps but never its data. This recovers the geography from the pixels of those maps, and uses it to guess where you grew up.",
      customCss: ["./src/styles/tokens.css", "./src/styles/starlight.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/incrediblecrab/american-dialects",
        },
      ],
      sidebar: [
        {
          label: "The essay",
          items: [
            { label: "The quiz", link: "/" },
            { label: "Recovering the data", link: "/recovery/" },
            { label: "Isoglosses", link: "/isogloss/" },
            { label: "The mistake", link: "/mistake/" },
            { label: "How many questions", link: "/questions/" },
            { label: "What is known", link: "/limits/" },
          ],
        },
        {
          label: "About",
          items: [{ label: "Colophon", link: "/colophon/" }],
        },
      ],
    }),
    mdx(),
    react(),
  ],
  build: {
    assets: "assets",
  },
  vite: {
    build: {
      assetsInlineLimit: 0,
    },
  },
});