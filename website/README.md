# Mei website

Mobile-first static website. Hono renders JSX full pages and HTML fragments during
the build; htmx switches the explainer diagram without a full reload. Each
comparison also has a complete static page, so the site works without JavaScript.
No application server or model downloads run in production. The shared page
template loads Simple Analytics asynchronously from its CDN, with
`data-hostname="mei.computer"` for privacy-first page-view analytics.
The shared footer uses Simple Analytics' Text link badge preset, labelled
“100% privacy-first analytics”, for the Free plan attribution.

`/how-mei-fits/` explains the system diagram, compares Mei with llama-server,
and shows generic-client and CoCore connection examples. Full-page comparison
links work without JavaScript; htmx swaps only the diagram when available.
The page uses shared navigation and metadata components and can grow into
additional explanations later.

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

`src/data.ts` defines the current stable release links, pinned model downloads,
and serving examples. The home page covers installation, client setup, supported
models, and the latest release, including Inkling's context and feature limits.

When updating the release, verify the published tag, archive and checksum, then
update the model revisions, examples and qualification claims together. Keep
claims scoped to the tested model, runtime and hardware; do not present unfinished
qualifications as completed results.

Historical benchmark evidence remains in `public/data/` in the repository.
The build excludes that directory from the published site. The old comparison
pages, chart fragments and `/data/results.json` are no longer generated.

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
