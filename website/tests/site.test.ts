import assert from "node:assert/strict";
import { test } from "node:test";
import { createSite, basePath, outputRoutes } from "../src/site.js";

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
