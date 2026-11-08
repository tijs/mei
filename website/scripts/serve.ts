import { serve } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono } from 'hono';
import { basePath } from '../src/site.js';

const base = basePath();
const app = new Hono();
if (base !== '/') app.get('/', c => c.redirect(base));
app.use('*', serveStatic({ root: './dist', rewriteRequestPath: path => path.startsWith(base) ? '/' + path.slice(base.length) : '/__missing__' }));
app.notFound(c => c.text('Not found', 404));
serve({ fetch: app.fetch, hostname: '127.0.0.1', port: Number(process.env.PORT ?? 4173) }, info => console.log(`Mei website: http://127.0.0.1:${info.port}${base}`));
