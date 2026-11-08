const github = "https://github.com/tijs/mei";

export function SiteHead({
  base,
  path = "",
  title,
  description,
}: {
  base: string;
  path?: string;
  title: string;
  description: string;
}) {
  const siteURL = new URL(
    process.env.SITE_URL ?? `https://tijs.github.io${base}`,
  );
  const url = new URL(path, siteURL).href;
  return (
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{title}</title>
      <meta name="description" content={description} />
      <meta name="theme-color" content="#fbf4e9" />
      <link rel="canonical" href={url} />
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:type" content="website" />
      <meta property="og:url" content={url} />
      <meta
        property="og:image"
        content={`${siteURL.href}assets/mei-coast.jpg`}
      />
      <link rel="icon" href={`${base}assets/mei-logo.png`} type="image/png" />
      <link
        rel="preload"
        href={`${base}assets/Anton-Regular.ttf`}
        as="font"
        type="font/ttf"
        crossorigin="anonymous"
      />
      <link rel="stylesheet" href={`${base}assets/site.css`} />
      <meta
        name="htmx-config"
        content='{"allowEval":false,"allowScriptTags":false,"selfRequestsOnly":true}'
      />
      <script src={`${base}assets/htmx.min.js`} defer></script>
      <script src={`${base}assets/site.js`} defer></script>
    </head>
  );
}

export function Brand({ base }: { base: string }) {
  return (
    <a class="brand" href={`${base}#top`} aria-label="Mei home">
      <svg class="brand-seal" viewBox="238 134 758 745" aria-hidden="true">
        <image href={`${base}assets/mei-logo.png`} width="1254" height="1254" />
      </svg>
      <span>mei</span>
    </a>
  );
}

export function SiteHeader({
  base,
  explainer = false,
}: {
  base: string;
  explainer?: boolean;
}) {
  const Links = () => (
    <>
      <a
        href={`${base}how-mei-fits/`}
        aria-current={explainer ? "page" : undefined}
      >
        How Mei fits
      </a>
      <a href={`${base}#models`}>Models</a>
      <a href={`${base}#results`}>Results</a>
      <a href={`${base}#get-started`}>Get started</a>
      <a href={github}>GitHub ↗</a>
    </>
  );
  return (
    <header class="site-header wrap">
      <Brand base={base} />
      <nav class="desktop-nav" aria-label="Main navigation">
        <Links />
      </nav>
      <details class="mobile-menu">
        <summary>
          Menu <span aria-hidden="true">☰</span>
        </summary>
        <nav aria-label="Mobile navigation">
          <Links />
        </nav>
      </details>
    </header>
  );
}

export function SiteFooter({ base }: { base: string }) {
  return (
    <footer class="site-footer wrap">
      <Brand base={base} />
      <p>A Swift/MLX model server for Apple Silicon.</p>
      <nav aria-label="Footer">
        <a href={`${base}how-mei-fits/`}>How Mei fits</a>
        <a href={github}>GitHub ↗</a>
        <a href={`${github}#readme`}>Documentation ↗</a>
        <a href="https://github.com/ml-explore/mlx">MLX ↗</a>
        <a href="https://github.com/tijs/vmlx-swift">vMLX ↗</a>
      </nav>
      <small>Open source · MIT · Model weights have their own licenses.</small>
    </footer>
  );
}

export function Code({ children, label }: { children: string; label: string }) {
  return (
    <div class="code-block">
      <div class="code-top">
        <span>{label}</span>
        <button
          id={`copy-${label.toLowerCase().replaceAll(" ", "-")}`}
          type="button"
          class="copy-button"
          data-copy
          aria-label={`Copy ${label}`}
        >
          Copy
        </button>
      </div>
      <pre>
        <code>{children}</code>
      </pre>
    </div>
  );
}
