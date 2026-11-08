import { Hono } from "hono";
import { evidence, labels, methods, models, views, type View } from "./data.js";

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

function Brand({ base }: { base: string }) {
  // Viewport crops the exact approved raster, without redrawing the seal.
  return (
    <a class="brand" href={`${base}#top`} aria-label="Mei home">
      <svg class="brand-seal" viewBox="238 134 758 745" aria-hidden="true">
        <image href={`${base}assets/mei-logo.png`} width="1254" height="1254" />
      </svg>
      <span>mei</span>
    </a>
  );
}

function Bar({
  name,
  value,
  max,
  text,
  second = false,
}: {
  name: string;
  value: number;
  max: number;
  text: string;
  second?: boolean;
}) {
  return (
    <div class={`bar-row${second ? " secondary" : ""}`}>
      <span class="bar-name">{name}</span>
      <div class="bar-track" aria-hidden="true">
        <span class="bar-fill" style={`width:${(value / max) * 100}%`} />
      </div>
      <span class="bar-value">{text}</span>
    </div>
  );
}

export function Results({ view, base }: { view: View; base: string }) {
  const titles = {
    quality: "Tasks completed successfully",
    time: "Average time per task",
    memory: "Increase in wired memory",
    decode: "Text generation speed",
  };
  return (
    <div
      id="results-panel"
      class="results-panel"
      aria-live="polite"
      aria-atomic="true"
    >
      <nav class="chart-nav" aria-label="Choose a measurement">
        {views.map((v) => (
          <a
            id={`view-${v}`}
            href={`${base}results/${v}.html#results`}
            hx-get={`${base}fragments/results-${v}.html`}
            hx-target="#results-panel"
            hx-swap="outerHTML"
            hx-sync="#results-panel:replace"
            aria-current={v === view ? "page" : undefined}
          >
            {labels[v]}
          </a>
        ))}
      </nav>
      <div class="chart-heading">
        <h3>{titles[view]}</h3>
        <p>
          {view === "quality"
            ? "Passed tasks · more is better"
            : view === "time"
              ? "Seconds per task · less is better"
              : view === "memory"
                ? "Peak wired increase · GiB · less is better"
                : "Generated tokens per second · more is better"}
        </p>
      </div>
      {view === "quality" ? (
        <>
          <h4 class="plot-label">
            Assistant tasks <span>out of 24 attempts</span>
          </h4>
          <div class="plot">
            {models.map((m) => (
              <Bar
                name={m.name}
                value={m.household}
                max={24}
                text={`${m.household}/24`}
              />
            ))}
          </div>
          <h4 class="plot-label">
            Python tasks <span>out of 6 attempts</span>
          </h4>
          <div class="plot">
            {models.map((m) => (
              <Bar
                name={m.name}
                value={m.python}
                max={6}
                text={`${m.python}/6`}
                second
              />
            ))}
          </div>
        </>
      ) : view === "time" ? (
        <div class="plot single">
          {models.map((m) => (
            <Bar
              name={m.name}
              value={m.taskSeconds}
              max={50}
              text={`${m.taskSeconds.toFixed(2)} s`}
            />
          ))}
        </div>
      ) : view === "memory" ? (
        <div class="plot single">
          {models.map((m) => (
            <Bar
              name={m.name}
              value={m.wiredGiB}
              max={24}
              text={`${m.wiredGiB.toFixed(2)}`}
            />
          ))}
        </div>
      ) : (
        <>
          <h4 class="plot-label">
            Short prompt <span>17 input tokens</span>
          </h4>
          <div class="plot">
            {models.map((m) => (
              <Bar
                name={m.name}
                value={m.shortDecode}
                max={75}
                text={`${m.shortDecode.toFixed(1)}`}
              />
            ))}
          </div>
          <h4 class="plot-label">
            Long prompt <span>30,000 input tokens</span>
          </h4>
          <div class="plot">
            {models.map((m) => (
              <Bar
                name={m.name}
                value={m.longDecode}
                max={75}
                text={`${m.longDecode.toFixed(1)}`}
                second
              />
            ))}
          </div>
        </>
      )}
      <p class="chart-method">{methods[view]}</p>
      <a
        class="text-link"
        href={view === "decode" ? evidence.contextSource : evidence.source}
      >
        Test setup and full results ↗
      </a>
    </div>
  );
}

function Code({ children, label }: { children: string; label: string }) {
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

function Page({ view, base }: { view: View; base: string }) {
  const url = new URL(process.env.SITE_URL ?? `https://tijs.github.io${base}`)
    .href;
  return (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Mei — Run language models on your Mac</title>
        <meta
          name="description"
          content="Mei runs language models on Apple Silicon using MLX. Connect your apps through an OpenAI-compatible API. Includes tested model settings and published results."
        />
        <meta name="theme-color" content="#fbf4e9" />
        <link rel="canonical" href={url} />
        <meta
          property="og:title"
          content="Mei — Run language models on your Mac"
        />
        <meta
          property="og:description"
          content="Run language models on Apple Silicon using MLX and connect your apps through an OpenAI-compatible API."
        />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={url} />
        <meta property="og:image" content={`${url}assets/mei-coast.jpg`} />
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
      <body id="top">
        <a class="skip-link" href="#main">
          Skip to content
        </a>
        <header class="site-header wrap">
          <Brand base={base} />
          <nav class="desktop-nav" aria-label="Main navigation">
            <a href={`${base}#models`}>Models</a>
            <a href={`${base}#results`}>Results</a>
            <a href={`${base}#get-started`}>Get started</a>
            <a href={github}>GitHub ↗</a>
          </nav>
          <details class="mobile-menu">
            <summary>
              Menu <span aria-hidden="true">☰</span>
            </summary>
            <nav aria-label="Mobile navigation">
              <a href={`${base}#models`}>Models</a>
              <a href={`${base}#results`}>Results</a>
              <a href={`${base}#get-started`}>Get started</a>
              <a href={github}>GitHub ↗</a>
            </nav>
          </details>
        </header>
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
                Uses MLX. Tested on an M1 Max with 32 GB of memory.
              </p>
              <a class="button" href="#get-started">
                Get started <Icon name="arrow" />
              </a>
              <span class="hero-footnote">
                Apple Silicon · macOS 15+ · MIT licensed
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
                Mei includes tested settings for these three models. The
                descriptions below explain the differences we found in the same
                set of tests.
              </p>
            </div>
            <div class="model-grid">
              {models.map((m) => (
                <article class="model" id={m.id}>
                  <div class="model-heading">
                    <img
                      class="model-logo"
                      src={`${base}assets/model-${m.id}.png`}
                      alt=""
                      width="64"
                      height="64"
                      loading="lazy"
                      decoding="async"
                    />
                    <h3>{m.name}</h3>
                  </div>
                  <p class="model-size">{m.fullName}</p>
                  <p>{m.description}</p>
                  <div class="model-fact">
                    <strong>{m.household}/24</strong>
                    <span>assistant tasks passed</span>
                  </div>
                  <div class="model-links">
                    <a href={m.modelCard}>Original model card ↗</a>
                    <a
                      href={`https://huggingface.co/${m.repo}/tree/${m.revision}`}
                    >
                      Files used by Mei ↗
                    </a>
                  </div>
                  <details class="profile">
                    <summary>Configuration name</summary>
                    <code>{m.profile}</code>
                  </details>
                </article>
              ))}
            </div>
            <p class="model-note">
              All three are mixture-of-experts models: they use about 3 billion
              parameters for each token, but the full set of weights must be
              stored. Weight storage in these tests was about 11.45 GiB for
              Groot and 18.17 GiB for Qwen and Ornith. Running a model also
              needs memory for the prompt, generated text and calculations.
            </p>
            <div class="ongoing">
              <h3>We keep testing new models</h3>
              <p>
                We use test results to choose models and adjust their settings.
                In the latest tests, Gemma passed the same 22 of 24 assistant
                tasks as Groot, but passed 3 of 6 Python tasks and averaged
                72.27 seconds per task. Our current recommendations remain
                Groot, Qwen and Ornith.
              </p>
              <a href={evidence.evaluationSource}>
                Read the latest model tests ↗
              </a>
            </div>
          </section>
          <section class="results" id="results" aria-labelledby="results-title">
            <div class="wrap">
              <div class="results-intro">
                <h2 id="results-title">Test results</h2>
                <p>
                  Compare task results, time per task, memory measurements and
                  text generation speed for the three models.
                </p>
                <p class="test-stamp">
                  M1 Max · 32 GB · Mei 0.11.0
                  <br />
                  Tested 7 November 2026 · two runs per model
                </p>
              </div>
              <Results view={view} base={base} />
              <details class="methodology">
                <summary>How we tested</summary>
                <p>
                  We used test suite {evidence.method} with random seeds 17 and
                  29. Each model ran 12 simulated household-assistant tasks and
                  3 small Python tasks per run. We gave both runs equal weight
                  when calculating averages. Each model used its own sampling
                  and cache settings. Two runs are too few to show that one
                  model will consistently outperform another or to predict how
                  it will work in your app.
                </p>
                <div class="table-scroll">
                  <table>
                    <caption>Results from two runs per model</caption>
                    <thead>
                      <tr>
                        <th scope="col">Model</th>
                        <th scope="col">Assistant</th>
                        <th scope="col">Python</th>
                        <th scope="col">Task time</th>
                        <th scope="col">Wired increase</th>
                      </tr>
                    </thead>
                    <tbody>
                      {models.map((m) => (
                        <tr>
                          <th scope="row">{m.name}</th>
                          <td>{m.household}/24</td>
                          <td>{m.python}/6</td>
                          <td>{m.taskSeconds.toFixed(2)} s</td>
                          <td>{m.wiredGiB.toFixed(2)} GiB</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  The memory figures show changes in the Mac’s wired memory, not
                  total memory used by each model. Task times include failures.
                  Text generation speed was tested separately and does not show
                  whether longer prompts produced correct answers. These charts
                  show the three recommended Mei models; the full report
                  includes other models and engines.
                </p>
                <a href={`${base}data/results.json`}>
                  Download the chart data ↗
                </a>
              </details>
            </div>
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
                It can stream generated text, return tool calls and produce JSON
                using supported schema rules. It reads long prompts in chunks to
                limit memory use and can reuse calculations from earlier
                requests.
              </p>
              <p>
                We test on an M1 Max with 32 GB of memory. Other Apple Silicon
                Macs can run Mei too. The model you can run and its speed depend
                on your Mac’s memory, the length of the prompt and the work you
                ask it to do.
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
                  This installs the stable release. You need an Apple Silicon
                  Mac running macOS 15 or later.
                </p>
                <Code label="Install Mei">{"brew install tijs/tap/mei"}</Code>
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
                  {
                    "mei pull groot-19b-a3b-text\n\nmei \\\n  --model-dir ~/.cache/mei/models/Groot-19B-A3B-261023-rco-role-4.5bpw-textonly \\\n  --model-profile groot-19b-a3b-text \\\n  --served-model-id groot"
                  }
                </Code>
                <p class="small">
                  Mei downloads a fixed version of the model.{" "}
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
                  ID <code>groot</code>.
                </p>
                <Code label="Make a first request">
                  {
                    'curl http://127.0.0.1:8024/v1/chat/completions \\\n  -H \'Content-Type: application/json\' \\\n  -d \'{\n    "model": "groot",\n    "messages": [{\n      "role": "user",\n      "content": "Hello, Mei."\n    }]\n  }\''
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
            <div class="start-links">
              <a class="button" href={`${github}/releases/latest`}>
                Get the latest release <Icon name="arrow" />
              </a>
              <a class="text-link" href={`${github}#quickstart`}>
                Read the full quickstart ↗
              </a>
            </div>
          </section>
        </main>
        <footer class="site-footer wrap">
          <Brand base={base} />
          <p>A Swift/MLX model server for Apple Silicon.</p>
          <nav aria-label="Footer">
            <a href={github}>GitHub ↗</a>
            <a href={`${github}#readme`}>Documentation ↗</a>
            <a href="https://github.com/ml-explore/mlx">MLX ↗</a>
            <a href="https://github.com/tijs/vmlx-swift">vMLX ↗</a>
          </nav>
          <small>
            Open source · MIT · Model weights have their own licenses.
          </small>
        </footer>
        <div class="sr-only" role="status" id="copy-status" />
      </body>
    </html>
  );
}

export function createSite(base = basePath()) {
  const app = new Hono();
  app.get("/", (c) => c.html(<Page view="quality" base={base} />));
  for (const view of views) {
    app.get(`/results/${view}.html`, (c) =>
      c.html(<Page view={view} base={base} />),
    );
    app.get(`/fragments/results-${view}.html`, (c) =>
      c.html(<Results view={view} base={base} />),
    );
  }
  app.get("/data/results.json", (c) =>
    c.json({ evidence, models, definitions: methods }),
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
  ...views.flatMap((v) => [
    `/results/${v}.html`,
    `/fragments/results-${v}.html`,
  ]),
  "/data/results.json",
  "/404.html",
];
