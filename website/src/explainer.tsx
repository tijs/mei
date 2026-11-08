import { Code, SiteFooter, SiteHead, SiteHeader } from "./components.js";

export type System = "mei" | "llama";

export function SystemDiagram({
  base,
  system,
}: {
  base: string;
  system: System;
}) {
  const mei = system === "mei";
  return (
    <div id="system-diagram" class="system-diagram" aria-live="polite">
      <nav class="system-switcher" aria-label="Compare model servers">
        <a
          id="system-mei"
          href={`${base}how-mei-fits/#system`}
          hx-get={`${base}fragments/system-mei.html`}
          hx-target="#system-diagram"
          hx-swap="outerHTML"
          hx-sync="#system-diagram:replace"
          aria-current={mei ? "page" : undefined}
        >
          Mei
        </a>
        <a
          id="system-llama"
          href={`${base}how-mei-fits/llama-server.html#system`}
          hx-get={`${base}fragments/system-llama.html`}
          hx-target="#system-diagram"
          hx-swap="outerHTML"
          hx-sync="#system-diagram:replace"
          aria-current={!mei ? "page" : undefined}
        >
          llama-server
        </a>
      </nav>
      <div class="system-mac">
        <span class="system-device-label">Your Mac</span>
        <div class="system-app-row">
          <div class="system-box">
            <h3>Your app or agent harness</h3>
            <p>CoCore or an app you build.</p>
            <p>
              Keeps the conversation, sends requests and runs the tools. An
              agent harness is the software that manages this loop.
            </p>
          </div>
          <div class="system-box system-tools">
            <h3>Your tools</h3>
            <p>For example, read a file or run a calculation.</p>
            <p>The app runs these actions. The model asks for them.</p>
          </div>
        </div>
        <div class="system-api">
          <span aria-hidden="true">⇅</span>
          <p>
            Messages and tool descriptions go to the server.
            <br />
            Generated text or tool-call requests come back.
            <br />
            <strong>OpenAI-compatible HTTP API</strong>
          </p>
        </div>
        <div class="system-runtime-row">
          <div
            class={`system-process ${mei ? "system-process-mei" : "system-process-llama"}`}
          >
            <div class="system-process-head">
              <h3>{mei ? "Mei" : "llama-server"}</h3>
              <span>One running server process</span>
              <p>
                {mei
                  ? "Provides the API, selects model settings and manages requests and response rules."
                  : "Provides an HTTP API for llama.cpp. It also includes a web chat interface."}
              </p>
            </div>
            <details class="system-inside" open>
              <summary>Inside this process</summary>
              <div class="system-layer">
                <h4>{mei ? "vMLX runtime" : "llama.cpp runtime"}</h4>
                {mei && <p>Mei’s pinned fork</p>}
                <p>
                  Loads the model, runs text generation and manages generation
                  caches.
                </p>
              </div>
              <div class="system-layer">
                <h4>{mei ? "MLX" : "ggml and hardware backends"}</h4>
                <p>
                  {mei
                    ? "Performs array calculations on Apple Silicon’s CPU and GPU."
                    : "Performs the calculations. Metal is one available backend on Apple Silicon."}
                </p>
              </div>
              <p class="system-library-note">
                These libraries run inside the server. They are not separate
                servers to start.
              </p>
            </details>
          </div>
          <div class="system-box system-model">
            <h3>Downloaded model files</h3>
            <p>
              {mei
                ? "For example, Mei’s prepared MLX version of Groot."
                : "A compatible model in GGUF format."}
            </p>
            <p class="system-load">
              <span aria-hidden="true">←</span> Loaded by the runtime
            </p>
            <p>Model weights are data on disk, not another running app.</p>
          </div>
        </div>
        <p class="system-hardware">Apple Silicon CPU and GPU · shared memory</p>
      </div>
      <p class="system-caption">
        {mei
          ? "You install and start Mei. Its vMLX and MLX dependencies are included. Your app calls Mei over HTTP; it does not call those libraries directly."
          : "llama-server fills the same role here: a model server that a separate app can call. It uses llama.cpp and GGUF model files, while Mei uses vMLX, MLX and compatible MLX model files."}
      </p>
    </div>
  );
}

export function ExplainerPage({
  base,
  system = "mei",
}: {
  base: string;
  system?: System;
}) {
  return (
    <html lang="en">
      <SiteHead
        base={base}
        path="how-mei-fits/"
        title="Where Mei fits in an AI system — Mei"
        description="How your app, Mei, vMLX, MLX and model files work together. Compare with llama-server and connect an OpenAI-compatible client or CoCore."
      />
      <body id="top">
        <a class="skip-link" href="#main">
          Skip to content
        </a>
        <SiteHeader base={base} explainer />
        <main id="main" class="explainer wrap">
          <header class="explainer-intro">
            <h1>Where Mei fits</h1>
            <p>
              Mei runs a language model and serves requests from your app. Your
              app handles the conversation and tools. The diagram shows where
              each part runs on a Mac.
            </p>
          </header>
          <section
            class="explainer-section"
            id="system"
            aria-label="System diagram"
          >
            <SystemDiagram base={base} system={system} />
          </section>
          <section
            class="explainer-section explainer-prose"
            aria-labelledby="layers-title"
          >
            <h2 id="layers-title">What Mei adds</h2>
            <p>
              Mei provides the HTTP server, model configurations, request limits
              and response handling. It uses vMLX to load models and generate
              text, and MLX to perform the underlying calculations.
            </p>
            <p>
              Mei uses a maintained fork of vMLX. Some of the changes made for
              Mei live in that fork, including work on generation and caching.
              The work is spread across the server and its dependencies; you
              receive them together when you install Mei.
            </p>
            <p>
              The model files contain the learned weights and other data the
              runtime needs. Downloading a model does not start a server. You
              start Mei with a model directory, then connect your app to Mei’s
              API.
            </p>
          </section>
          <section
            class="explainer-section explainer-prose"
            aria-labelledby="tools-title"
          >
            <h2 id="tools-title">Who runs a tool call?</h2>
            <p>
              The model can ask for a tool. Your app decides whether to run it
              and returns the result. Mei does not read your files or run shell
              commands just because the model requests them.
            </p>
            <ol class="tool-call-steps">
              <li>You ask your assistant to summarize a file.</li>
              <li>
                The app sends the conversation and a description of its{" "}
                <code>read_file</code> tool to Mei.
              </li>
              <li>
                The model requests <code>read_file</code>. Mei returns that
                request to the app.
              </li>
              <li>
                The app reads the file and sends its contents back in another
                request.
              </li>
              <li>Mei runs the model again and returns the summary.</li>
            </ol>
            <p>
              The app can repeat this loop for more steps, stop it, or ask you
              to approve an action.
            </p>
          </section>
          <section
            class="explainer-section explainer-prose"
            aria-labelledby="comparison-title"
          >
            <h2 id="comparison-title">How this compares with llama-server</h2>
            <p>
              <a href="https://github.com/ggml-org/llama.cpp/tree/master/tools/server">
                llama-server
              </a>{" "}
              also provides an API that an app or agent harness can call. It
              runs models through llama.cpp and includes a web chat interface.
            </p>
            <p>
              Mei focuses on Apple Silicon and uses the MLX stack. llama.cpp
              supports several hardware backends and uses GGUF model files.
              Choose model files that match the server you are running.
            </p>
            <p>
              An app may support both servers through its OpenAI-compatible
              provider settings. The exact API features and model behavior still
              need to match what the app expects.
            </p>
          </section>
          <section
            class="explainer-section explainer-prose"
            id="connect"
            aria-labelledby="connect-title"
          >
            <h2 id="connect-title">Connect your app to Mei</h2>
            <p>
              Start Mei with the model ID <code>groot</code>, as shown in the{" "}
              <a href={`${base}#get-started`}>installation instructions</a>.
              Then configure the model provider in your app.
            </p>
            <div class="connection-example">
              <h3>OpenAI-compatible client</h3>
              <p>
                If the app supports a custom OpenAI-compatible endpoint, use
                these settings:
              </p>
              <Code label="OpenAI-compatible client settings">
                {"Base URL: http://127.0.0.1:8024/v1\nModel ID: groot"}
              </Code>
              <p>
                The model ID must match the ID Mei serves. The app also needs to
                support the features you use, such as streaming or tool calls.
              </p>
            </div>
            <div class="connection-example">
              <h3>CoCore</h3>
              <p>
                CoCore’s engine map takes the server root, without{" "}
                <code>/v1</code>. Add this entry to{" "}
                <code>~/.cocore/engine-map</code>:
              </p>
              <Code label="CoCore engine map">
                {"groot = http://127.0.0.1:8024"}
              </Code>
              <p>
                CoCore checks the model’s capabilities when it connects. See the{" "}
                <a href="https://github.com/tijs/mei/blob/main/docs/COCORE.md">
                  CoCore setup instructions
                </a>{" "}
                for the complete configuration.
              </p>
            </div>
          </section>
          <section
            class="explainer-section explainer-prose explainer-sources"
            aria-labelledby="sources-title"
          >
            <h2 id="sources-title">Further reading</h2>
            <ul>
              <li>
                <a href="https://github.com/tijs/mei/blob/main/docs/ARCHITECTURE.md">
                  Mei’s architecture
                </a>
              </li>
              <li>
                <a href="https://github.com/tijs/mei/blob/main/docs/VMLX-FORK.md">
                  Mei’s vMLX fork and dependency versions
                </a>
              </li>
              <li>
                <a href="https://github.com/ml-explore/mlx">MLX</a>
              </li>
              <li>
                <a href="https://github.com/ggml-org/llama.cpp/tree/master/tools/server">
                  llama-server documentation
                </a>
              </li>
            </ul>
            <a class="button" href={`${base}#get-started`}>
              Install and run Mei →
            </a>
          </section>
        </main>
        <SiteFooter base={base} />
        <div class="sr-only" role="status" id="copy-status" />
      </body>
    </html>
  );
}
