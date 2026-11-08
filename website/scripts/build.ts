import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { createSite, outputRoutes } from "../src/site.js";

await rm("dist", { recursive: true, force: true });
await mkdir("dist/assets", { recursive: true });
await cp("public", "dist", { recursive: true });
await cp("node_modules/htmx.org/dist/htmx.min.js", "dist/assets/htmx.min.js");
await cp("node_modules/htmx.org/LICENSE", "dist/assets/HTMX-LICENSE.txt");
const app = createSite();
for (const route of outputRoutes) {
  const response = await app.request(`http://build.local${route}`);
  if (!response.ok)
    throw new Error(`Failed to render ${route}: ${response.status}`);
  const output = `dist${route === "/" ? "/index.html" : route}`;
  await mkdir(output.slice(0, output.lastIndexOf("/")), { recursive: true });
  let body = await response.text();
  if (!route.includes("/fragments/") && !route.endsWith(".json"))
    body = `<!doctype html>\n${body}`;
  await writeFile(output, body);
}
await writeFile("dist/.nojekyll", "");
console.log(`Rendered ${outputRoutes.length} Hono routes to static files.`);
