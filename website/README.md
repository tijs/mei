# Mei website

Mobile-first static website. Hono renders JSX full pages and HTML fragments during
the build; htmx switches result views without a full reload. Every result link
also targets a complete static page, so the site works without JavaScript.
No server, analytics, external CDN or model downloads run in production.

```sh
npm ci
npm run check
npm test
npm run build
npm run dev
```

Local preview: `http://127.0.0.1:4173/mei/`. `BASE_PATH` defaults to `/mei/`
for local development. Production uses `BASE_PATH=/` and
`SITE_URL=https://mei.computer/` to serve the custom domain from its root.
`dist/` is generated.
GitHub Actions builds and publishes it. There is intentionally no CNAME file.

## Updating content

`src/data.ts` freezes the curated benchmark snapshot and links to a full source
commit. Values come from the published four-profile comparison and separate
context-speed report at local-model-bench commit
`abb0cbd7caa23c4ea41817559cc8dde3b0e0fabf` (verified on GitHub). Only the three tuned Mei profiles are
plotted. The latest six-model evaluation is linked, with Gemma's tradeoffs stated.

Refresh source commit, date, model revisions, values and interpretation together.
Preserve metric definitions, failed attempts, equal run weighting and synthetic
workload limits. Wired-memory increase is **not** total model memory. Do not use
unfinished qualifications as completed results or silently blend runtimes.
`/data/results.json` contains the rendered snapshot and definitions.

## Assets

- Hero: original illustration supplied by Tijs in the website design session;
  encoded as JPEG for delivery without changing the composition.
- Logo: exact approved carved-seal v3, identical to `../assets/mei-logo.png`.
  The inline SVG viewport shows the seal from this approved lockup without
  redrawing it. The source lockup retains the wordmark as well.
- Anton and Manrope: self-hosted fonts; OFL licenses alongside them.
- htmx: copied from the locked npm dependency; license shipped with it.
- Model icons: cobalt adaptations of the original Hugging Face model/team
  artwork. Original marks belong to their creators; these are Mei-specific
  adaptations, not official alternate logos. Source links are in
  `design/model-logo-sources.json`; generation prompts are kept alongside it.

The website design/content lives here; durable scope and decisions live in Kiem:
`kiem://note/700e0747-dcbb-4132-b216-34ae5bde0594`.
