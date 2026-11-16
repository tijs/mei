import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { createSite, basePath, outputRoutes } from "../src/site.js";
import { evidence, models, release } from "../src/data.js";
import measured from "../public/data/mei-0.11.2-20261116/measurements.json" with { type: "json" };

test("current release matches the binary version; model examples use automatic settings without relabelling evidence", async () => {
  const config = readFileSync(new URL("../../Sources/MeiCore/ServerConfig.swift", import.meta.url), "utf8");
  assert.equal(config.match(/static let version = "([^"]+)"/)?.[1], release.version);
  const html = await (await createSite("/").request("https://test.local/")).text();
  assert.ok(html.includes(`href="${release.url}"`));
  assert.ok(html.includes(`href="${release.archive}"`));
  assert.ok(html.includes(`href="${release.archive}.sha256"`));
  for (const model of models) {
    assert.ok(html.includes(`mei pull ${model.profile}`));
    assert.ok(html.includes(`models/${model.repo.split("/").at(-1)}`));
    assert.ok(html.includes(`--served-model-id ${model.id}`));
    assert.ok(html.includes(model.automaticSettings));
  }
  assert.ok(!html.includes("--model-profile"));
  assert.equal(evidence.runtime, "Mei 0.11.2");
  const ids = [...html.matchAll(/ id="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length, "copy controls and sections must have unique IDs");
});

test("all four charts use the complete audited cohort and preserve historical values separately", async () => {
  assert.equal(measured.taskObservations.length, 90);
  assert.equal(measured.decodeObservations.length, 60);
  assert.equal(evidence.date, "2026-11-16");
  assert.equal(evidence.dateLabel, "16 November 2026");
  const historical = JSON.parse(readFileSync(new URL("../public/data/mei-0.11.0-20261107/results.json", import.meta.url), "utf8"));
  assert.equal(historical.evidence.runtime, "Mei 0.11.0");
  assert.equal(historical.models.find((m: { id: string }) => m.id === "groot").household, 22);
  const speedLabels = { groot: "Groot", qwen: "Base Qwen3.6", ornith: "Ornith1.5" };
  for (const model of models) {
    const rows = measured.taskObservations.filter((r) => r.model === model.id);
    assert.equal(rows.length, 30);
    assert.equal(new Set(rows.map((r) => `${r.seed}/${r.track}/${r.taskId}`)).size, 30);
    for (const seed of [17, 29]) {
      assert.equal(rows.filter((r) => r.seed === seed && r.track === "household").length, 12);
      assert.equal(rows.filter((r) => r.seed === seed && r.track === "python").length, 3);
    }
    assert.equal(rows.filter((r) => r.track === "household" && r.passed).length, model.household);
    assert.equal(rows.filter((r) => r.track === "python" && r.passed).length, model.python);
    // Failed attempts remain in the task-time average.
    assert.ok(Math.abs(rows.reduce((sum, r) => sum + r.wallSeconds, 0) / 30 - model.taskSeconds) < 1e-9);
    const runs = measured.taskRuns.filter((r) => r.model === model.id);
    assert.equal(runs.length, 2);
    assert.deepEqual(runs.map((r) => r.seed).sort((a, b) => a - b), [17, 29]);
    assert.ok(Math.abs(runs.reduce((sum, r) => sum + r.wiredIncreaseGiB, 0) / 2 - model.wiredGiB) < 1e-9);
    for (const context of ["short", "long"] as const) {
      const speed = measured.decodeObservations.filter((r) => r.model === speedLabels[model.id] && r.context === context);
      assert.equal(speed.length, 10);
      assert.equal(new Set(speed.map((r) => r.process_id)).size, 2);
      for (const process of new Set(speed.map((r) => r.process_id))) {
        assert.deepEqual(speed.filter((r) => r.process_id === process).map((r) => r.repeat).sort((a, b) => a - b), [1, 2, 3, 4, 5]);
      }
      assert.ok(speed.every((r) => r.completion_tokens === 128 && r.prompt_tokens === (context === "short" ? 17 : 30000)));
      assert.ok(Math.abs(speed.reduce((sum, r) => sum + r.decode_tps, 0) / 10 - (context === "short" ? model.shortDecode : model.longDecode)) < 1e-9);
    }
  }
  assert.ok(measured.taskObservations.some((r) => !r.passed), "do not select only successful attempts");
  const site = createSite("/");
  const download = await (await site.request("https://test.local/data/results.json")).json();
  assert.deepEqual(download.models, models.map((m) => ({ ...m })));
  assert.equal(download.evidence.runtime, evidence.runtime);
  assert.equal(download.evidence.date, evidence.date);
  for (const view of ["quality", "time", "memory", "decode"]) {
    const html = await (await site.request(`https://test.local/results/${view}.html`)).text();
    assert.ok(html.includes("Mei 0.11.2") && html.includes("16 November 2026"));
    const expected = view === "quality"
      ? [...models.map((m) => [m.name, `${m.household}/24`]), ...models.map((m) => [m.name, `${m.python}/6`])]
      : view === "time" ? models.map((m) => [m.name, `${m.taskSeconds.toFixed(2)} s`])
      : view === "memory" ? models.map((m) => [m.name, m.wiredGiB.toFixed(2)])
      : [...models.map((m) => [m.name, m.shortDecode.toFixed(1)]), ...models.map((m) => [m.name, m.longDecode.toFixed(1)])];
    const fragment = await (await site.request(`https://test.local/fragments/results-${view}.html`)).text();
    for (const rendered of [html, fragment]) {
      const start = rendered.indexOf('id="results-panel"');
      const end = rendered.indexOf('<p class="chart-method">', start);
      assert.ok(start >= 0 && end > start);
      const panel = rendered.slice(start, end);
      const bars = [...panel.matchAll(/<span class="bar-name">([^<]+)<\/span>[\s\S]*?<span class="bar-value">([^<]+)<\/span>/g)].map((m) => [m[1], m[2]]);
      assert.deepEqual(bars, expected, `${view}: chart labels and values must match in full pages and fragments`);
    }
  }
});

test("all public routes render; chart links have complete-page and fragment destinations for both hosting paths", async () => {
  for (const base of ["/mei/", "/"]) {
    const site = createSite(base);
    for (const route of outputRoutes) {
      const response = await site.request(`https://test.local${route}`);
      assert.equal(response.status, 200, route);
      if (route.endsWith(".json")) continue;
      const html = await response.text();
      const chartUrls = [
        ...html.matchAll(
          /(?:hx-get|href)="([^"]*(?:results\/|fragments\/)[^"]*)"/g,
        ),
      ];
      for (const [, url] of chartUrls) {
        if (url.startsWith("https://")) continue;
        assert.ok(url.startsWith(base), `${base}: ${url}`);
        const route = "/" + url.slice(base.length).split("#")[0];
        assert.equal(
          (await site.request(`https://test.local${route}`)).status,
          200,
          route,
        );
      }
    }
    const fragment = await (
      await site.request("https://test.local/fragments/results-memory.html")
    ).text();
    assert.ok(fragment.includes('id="results-panel"'));
    assert.ok(fragment.includes("not total model memory"));
    assert.ok(!fragment.includes("<html"));
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
    assert.ok(mei.includes("groot = http://127.0.0.1:8024</code>"));
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
