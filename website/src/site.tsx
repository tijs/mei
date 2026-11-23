import { Hono } from "hono";
import { models, modelCommands, release } from "./data.js";
import { Code, SiteFooter, SiteHead, SiteHeader } from "./components.js";
import { ExplainerPage, SystemDiagram } from "./explainer.js";

export function basePath(value = process.env.BASE_PATH ?? "/mei/") {
  if (!/^\/(?:[a-zA-Z0-9_-]+\/)*$/.test(value))
    throw new Error(
      "BASE_PATH must be / or a slash-delimited path such as /mei/",
    );
  return value;
}

const github = "https://github.com/tijs/mei";

function Icon({ name }: { name: "tools" | "chat" | "code" | "arrow" }) {
  const paths = {
    tools:
      "M14.7 6.3a5.5 5.5 0 0 0-7 7L2.5 18.5a2.1 2.1 0 0 0 3 3l5.2-5.2a5.5 5.5 0 0 0 7-7l-3.8 3.8-3-3 3.8-3.8Z",
    chat: "M20 11.5a8 8 0 0 1-8 8c-1.5 0-3-.4-4.2-1.1L3 20l1.4-4.6A8 8 0 1 1 20 11.5ZM8 11h.01M12 11h.01M16 11h.01",
    code: "m8 5-6 7 6 7m8-14 6 7-6 7M14 3l-4 18",
    arrow: "M4 12h16m-6-6 6 6-6 6",
  };
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}

function Page({ base }: { base: string }) {
  return (
    <html lang="en">
      <SiteHead
        base={base}
        title="Mei — Run language models on your Mac"
        description="Mei is a native Swift model server for Apple Silicon, powered by MLX. Connect your apps through an OpenAI-compatible API. No Python environment required."
      />
      <body id="top">
        <a class="skip-link" href="#main">
          Skip to content
        </a>
        <SiteHeader base={base} />
        <main id="main">
          <section class="hero wrap" aria-labelledby="hero-title">
            <img
              class="hero-art"
              src={`${base}assets/mei-coast.jpg`}
              width="1536"
              height="1024"
              alt="A hand lifting a small, friendly Mac toward a blue seaside sky, printed in cobalt and pink."
              fetchpriority="high"
            />
            <div class="hero-copy">
              <h1 id="hero-title">
                Run language models
                <br />
                on your Mac.
              </h1>
              <p class="hero-description">
                Mei runs language models on Apple Silicon and makes them
                available to your apps through an OpenAI-compatible API.
              </p>
              <p class="hero-detail">
                Native Swift, powered by MLX. No Python environment required.
                Tested on an M1 Max with 32 GB of memory.
              </p>
              <a class="button" href="#get-started">
                Get started <Icon name="arrow" />
              </a>
              <span class="hero-footnote">
                <a href={release.url}>Mei {release.version}</a> · Apple Silicon · macOS 15+ · MIT licensed
              </span>
            </div>
          </section>
          <section class="section wrap work" aria-labelledby="work-title">
            <div class="section-intro">
              <h2 id="work-title">What you can use it for</h2>
              <p>
                Mei runs the model on your Mac. Your app provides the tools and
                carries out the actions the model requests.
              </p>
              <a class="explainer-link" href={`${base}how-mei-fits/`}>
                See how your app, Mei and the model fit together →
              </a>
            </div>
            <div class="work-list">
              <article>
                <Icon name="tools" />
                <div>
                  <h3>Tool-using agents</h3>
                  <p>
                    Let a model request actions through your app’s APIs, read
                    the results, and decide what to do next.
                  </p>
                </div>
              </article>
              <article>
                <Icon name="chat" />
                <div>
                  <h3>Local assistants</h3>
                  <p>
                    Build assistants that search information supplied by your
                    app, summarize it, or extract data from text.
                  </p>
                </div>
              </article>
              <article>
                <Icon name="code" />
                <div>
                  <h3>Light coding</h3>
                  <p>
                    Write small scripts, process data, or make changes to a few
                    lines of code.
                  </p>
                </div>
              </article>
            </div>
            <aside class="expectation">
              <strong>What to expect.</strong> The models we recommend are best
              suited to assistants, tool calls and small coding tasks. They
              still make mistakes. Give them only the actions they need and
              check important changes. We do not recommend this setup for
              independently building or maintaining large software projects.
            </aside>
          </section>
          <section
            class="section wrap"
            id="models"
            aria-labelledby="models-title"
          >
            <div class="section-heading">
              <div>
                <h2 id="models-title">Recommended models</h2>
              </div>
              <p>
                Mei supports Inkling 2 Mini alongside Groot, Qwen and Ornith.
                Choose a model that fits your Mac and the work you want to do.
              </p>
            </div>
            <p class="repack-explanation">
              The Groot, Qwen and Ornith downloads below are repacks of
              existing, quantized MLX models. A repack
              changes the packaged files, for example by removing unused
              components or adjusting the file layout. The repacking step leaves
              the retained language-model weights unchanged. Each Mei
              configuration selects a fixed version and includes the settings we
              tested with it.
            </p>
            <div class="model-grid">
              {models.map((m) => (
                <article class="model" id={m.id}>
                  <div class="model-heading">
                    <img
                      class="model-logo"
                      src={`${base}assets/${"logo" in m ? m.logo : `model-${m.id}.png`}`}
                      alt=""
                      width="64"
                      height="64"
                      loading="lazy"
                      decoding="async"
                    />
                    <h3>{m.name}</h3>
                  </div>
                  <p class="model-size">{m.fullName}</p>
                  <p class="model-description">{m.description}</p>
                  <p class="model-preparation">{m.preparation}</p>
                  <div class="model-links">
                    <a href={m.modelCard}>Original model card ↗</a>
                    <a href={`https://huggingface.co/${m.repo}`}>
                      Preparation notes ↗
                    </a>
                    <a
                      href={`https://huggingface.co/${m.repo}/tree/${m.revision}`}
                    >
                      Files used by Mei ↗
                    </a>
                    {"guide" in m && (
                      <>
                        <a href={m.guide}>Inkling setup and API example ↗</a>
                        <a href={`${github}#use-mei-with-pi`}>Pi client setup ↗</a>
                      </>
                    )}
                  </div>
                  <details class="profile">
                    <summary>Download and run {m.name}</summary>
                    <p>{m.automaticSettings}</p>
                    <Code label={`${m.name} model commands`}>{modelCommands(m)}</Code>
                  </details>
                </article>
              ))}
            </div>
            <p class="model-note">
              These are mixture-of-experts models. Each token uses only part
              of the model, but Mei still needs memory for the full set of
              weights, the prompt and generated text.
            </p>
          </section>
          <section class="section wrap" id="results" aria-labelledby="release-title">
            <div class="section-heading">
              <div><h2 id="release-title">Latest release: Mei {release.version}</h2></div>
              <p>The curated models now share a 26,624-token default on a 32 GB Mac. <a href="#context">Choose a context limit ↗</a></p>
            </div>
            <p>
              Run <code>mei pull inkling-2-mini-mei-q4</code> to opt in.
              Continue using <code>inkling-2-mini</code> for the original weights. The profile uses a 26,624-token context on a
              32 GB M1 Max. Use <code>--context-cap</code> to override it. Q4 scored
              20/25 on the task suite, compared with 22/25 for the original release.
            </p>
            <p><a href={release.url}>Read the full release notes ↗</a></p>
          </section>
          <section class="section wrap engine">
            <div>
              <h2>How Mei runs models</h2>
            </div>
            <div class="engine-copy">
              <p>
                Mei is written in Swift and uses MLX through vMLX to run models
                on Apple Silicon. Each server process loads one model and
                responds to HTTP requests from your app.
              </p>
              <p>
                It runs as a compiled executable, so you don’t need to install
                Python or manage a Python environment to use it.
              </p>
              <p>
                It can stream generated text and return tool calls. Supported
                features depend on the model: Groot, Qwen and Ornith can also
                produce JSON using supported schema rules and reuse calculations
                from earlier requests. Inkling uses its separate text profile.
                Mei reads long prompts in chunks to limit memory use.
              </p>
              <p>
                We test on an M1 Max with 32 GB of memory. Other Apple Silicon
                Macs can run Mei too. The model you can run and its speed depend
                on your Mac’s memory, the length of the prompt and the work you
                ask it to do.
              </p>
              <p>
                <a href={`${base}how-mei-fits/`}>
                  Read more about how Mei fits into your app →
                </a>
              </p>
              <a href={`${github}/blob/main/docs/OPENAI-COMPATIBILITY.md`}>
                Supported API features and limits ↗
              </a>
            </div>
          </section>
          <section
            class="section wrap getting-started"
            id="get-started"
            aria-labelledby="start-title"
          >
            <div class="section-heading">
              <div>
                <h2 id="start-title">Install and run Mei</h2>
              </div>
              <p>
                Mei is a command-line server. Install it, download a model, then
                connect your own app or send an API request. Mei does not
                include a chat interface.
              </p>
            </div>
            <div class="steps">
              <article>
                <span class="step-number">01</span>
                <h3>Install Mei</h3>
                <p>
                  This installs stable <a href={release.url}>Mei {release.version}</a>.
                  You need an Apple Silicon Mac running macOS 15 or later.
                </p>
                <Code label="Install Mei">{"brew install tijs/tap/mei"}</Code>
                <p class="small">
                  Manual install: <a href={release.archive}>Apple Silicon archive</a>{" "}
                  and <a href={`${release.archive}.sha256`}>checksum</a>.
                </p>
              </article>
              <article>
                <span class="step-number">02</span>
                <h3>Download and start Groot</h3>
                <p>
                  These commands use Groot, our suggested first model for the 32
                  GB M1 Max. Install the Hugging Face download tool first. If
                  you already have the <code>hf</code> command, skip that first
                  block.
                </p>
                <Code label="Install the download helper">
                  {"brew install uv\nuv tool install huggingface_hub"}
                </Code>
                <Code label="Download and run Groot">
                  {modelCommands(models[0])}
                </Code>
                <p class="small">
                  Mei downloads a fixed version and loads its validated settings automatically.{" "}
                  <code>mei pull groot-19b-a3b-text --dry-run</code> previews
                  what it would download. Each server runs one model; stop it
                  with Ctrl-C before switching models.
                </p>
              </article>
              <article>
                <span class="step-number">03</span>
                <h3>Send a request</h3>
                <p>
                  Set your OpenAI-compatible client’s base URL to{" "}
                  <code>http://127.0.0.1:8024/v1</code> and use the served model
                  ID <code>groot-19b-a3b-text</code>.
                </p>
                <Code label="Make a first request">
                  {
                    'curl http://127.0.0.1:8024/v1/chat/completions \\\n  -H \'Content-Type: application/json\' \\\n  -d \'{\n    "model": "groot-19b-a3b-text",\n    "messages": [{\n      "role": "user",\n      "content": "Hello, Mei."\n    }]\n  }\''
                  }
                </Code>
                <div class="integration">
                  <span class="integration-dot" aria-hidden="true" />
                  <div>
                    <h4>Use Mei with CoCore</h4>
                    <p>
                      CoCore supports Mei as an external model server. When it
                      connects, it checks whether the selected model can handle
                      tool calls and structured responses.
                    </p>
                    <a href={`${github}/blob/main/docs/COCORE.md`}>
                      CoCore setup instructions ↗
                    </a>
                  </div>
                </div>
              </article>
            </div>
            <section id="context" aria-labelledby="context-title">
              <h3 id="context-title">Choose a context limit</h3>
              <p>Restart Mei with <code>--context-cap</code> to override the default. Set your client’s context window to the same value.</p>
              <Code label="Override the context limit">{"mei --model groot-19b-a3b-text --context-cap 65536"}</Code>
              <div class="table-scroll">
                <table>
                  <caption>Example context caps for a single model</caption>
                  <thead><tr><th scope="col">Mac unified memory</th><th scope="col">Tokens</th><th scope="col">When to use it</th></tr></thead>
                  <tbody>
                    <tr><th scope="row">24 GB</th><td>8,192</td><td>Try if the weights fit. Inkling uses this below 32 GB.</td></tr>
                    <tr><th scope="row">32 GB</th><td>26,624</td><td>Curated default. A greedy raw Inkling Q4 completion processed 26,368 input tokens in 134 seconds on an M1 Max.</td></tr>
                    <tr><th scope="row">64 GB or more</th><td>65,536</td><td>Try a long prompt and check Memory Pressure in Activity Monitor.</td></tr>
                  </tbody>
                </table>
              </div>
              <p>Inkling shares the limit between prompt and reply. Groot, Qwen and Ornith apply it to the prompt and use a separate reply limit. Model weights and cache size affect how much fits. SenseNova Loop supports up to 4,096 tokens.</p>
            </section>
            <div class="start-links">
              <a class="button" href={release.url}>
                Get Mei {release.version} <Icon name="arrow" />
              </a>
              <a class="text-link" href={`${github}#quickstart`}>
                Read the full quickstart ↗
              </a>
            </div>
          </section>
        </main>
        <SiteFooter base={base} />
        <div class="sr-only" role="status" id="copy-status" />
      </body>
    </html>
  );
}

export function createSite(base = basePath()) {
  const app = new Hono();
  app.get("/", (c) => c.html(<Page base={base} />));
  app.get("/how-mei-fits/", (c) => c.html(<ExplainerPage base={base} />));
  app.get("/how-mei-fits/llama-server.html", (c) =>
    c.html(<ExplainerPage base={base} system="llama" />),
  );
  app.get("/fragments/system-mei.html", (c) =>
    c.html(<SystemDiagram base={base} system="mei" />),
  );
  app.get("/fragments/system-llama.html", (c) =>
    c.html(<SystemDiagram base={base} system="llama" />),
  );
  app.get("/404.html", (c) =>
    c.html(
      <html lang="en">
        <head>
          <meta name="viewport" content="width=device-width,initial-scale=1" />
          <title>Page not found — Mei</title>
          <link rel="stylesheet" href={`${base}assets/site.css`} />
        </head>
        <body>
          <main class="wrap section">
            <h1>Page not found</h1>
            <p>This page doesn’t exist.</p>
            <a class="button" href={base}>
              Back to Mei
            </a>
          </main>
        </body>
      </html>,
    ),
  );
  return app;
}

export const outputRoutes = [
  "/",
  "/how-mei-fits/",
  "/how-mei-fits/llama-server.html",
  "/fragments/system-mei.html",
  "/fragments/system-llama.html",
  "/404.html",
];
