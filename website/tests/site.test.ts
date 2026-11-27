import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { createSite, basePath, outputRoutes } from "../src/site.js";
import { models, release } from "../src/data.js";

test("current release matches the binary version and model examples use automatic settings", async () => {
  const config = readFileSync(new URL("../../Sources/MeiCore/ServerConfig.swift", import.meta.url), "utf8");
  assert.equal(config.match(/static let version = "([^"]+)"/)?.[1], release.version);
  const html = await (await createSite("/").request("https://test.local/")).text();
  assert.ok(html.includes(`href="${release.url}"`));
  assert.ok(html.includes(`href="${release.archive}"`));
  assert.ok(html.includes(`href="${release.archive}.sha256"`));
  for (const model of models) {
    const card = html.split(`<article class="model" id="${model.id}">`)[1]?.split("</article>")[0];
    assert.ok(card, `missing shared model card for ${model.id}`);
    assert.ok(card.includes(`Download and run ${model.name}`));
    assert.ok(card.includes(`mei pull ${model.profile}\nmei --model ${model.profile}`));
    assert.ok(card.includes(model.automaticSettings));
  }
  assert.ok(!html.includes("--model-profile"));
  assert.ok(html.includes("/assets/model-inkling-cobalt.png"));
  assert.ok(readFileSync(new URL("../public/assets/model-inkling-cobalt.png", import.meta.url)).length > 0);
  assert.ok(html.includes("--context-cap auto"));
  assert.ok(html.includes("mei --model groot-19b-a3b-text --context-cap 65536"));
  assert.ok(html.includes("<caption>Example context caps for a single model</caption>"));
  assert.ok(html.includes("inkling-2-mini"));
  assert.ok(html.includes("mei pull inkling-2-mini-mei-q4"));
  assert.ok(html.includes("22/25 passed"));
  assert.ok(html.includes("--cache-mode memory"));
  for (const old of ["0.11.0", "0.11.2", "Previous Mei", "results/quality.html"])
    assert.ok(!html.includes(old), `old release detail: ${old}`);
  const ids = [...html.matchAll(/ id="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length, "copy controls and sections must have unique IDs");
});

test("retired comparison routes no longer serve old release details", async () => {
  const site = createSite("/");
  for (const path of ["/results/quality.html", "/fragments/results-memory.html", "/data/results.json"])
    assert.equal((await site.request(`https://test.local${path}`)).status, 404);
});

test("all public routes render and internal navigation resolves for both hosting paths", async () => {
  for (const base of ["/mei/", "/"]) {
    const site = createSite(base);
    for (const route of outputRoutes) {
      const response = await site.request(`https://test.local${route}`);
      assert.equal(response.status, 200, route);
      if (route.endsWith(".json")) continue;
      const html = await response.text();
      for (const [, href] of html.matchAll(/href="([^"]+)"/g)) {
        if (!href.startsWith(base) || href.includes("assets/")) continue;
        const [path, fragment] = href.slice(base.length).split("#");
        const destination = await site.request(`https://test.local/${path}`);
        assert.equal(destination.status, 200, href);
        if (fragment) assert.ok((await destination.text()).includes(`id="${fragment}"`), href);
      }
    }

  }
});

test("invalid build base paths fail instead of emitting broken or unsafe URLs", () => {
  for (const value of ["mei", "/mei", "//host/", "/a/../", "/<script>/"])
    assert.throws(() => basePath(value));
});

test("explainer comparison has full-page fallbacks and separate client/CoCore settings", async () => {
  for (const base of ["/", "/mei/"]) {
    const site = createSite(base);
    const home = await (await site.request("https://test.local/")).text();
    assert.ok(home.includes(`href="${base}how-mei-fits/"`));
    const mei = await (
      await site.request("https://test.local/how-mei-fits/")
    ).text();
    const llama = await (
      await site.request("https://test.local/how-mei-fits/llama-server.html")
    ).text();
    assert.ok(mei.includes('id="system-diagram"'));
    assert.ok(mei.includes("http://127.0.0.1:8024/v1"));
    assert.ok(mei.includes("groot-19b-a3b-text = http://127.0.0.1:8024</code>"));
    assert.ok(!mei.includes("Osaurus"));
    assert.ok(llama.includes("A compatible model in GGUF format."));
    for (const html of [mei, llama]) {
      for (const [, href] of html.matchAll(
        /(?:href|hx-get)="([^"]*(?:how-mei-fits\/|fragments\/system-)[^"]*)"/g,
      )) {
        if (href.startsWith("https://")) continue;
        const path = "/" + href.slice(base.length).split("#")[0];
        assert.equal(
          (await site.request(`https://test.local${path}`)).status,
          200,
          href,
        );
      }
    }
  }
});
