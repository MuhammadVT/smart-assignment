Vendored from [Leaflet](https://leafletjs.com) 1.9.4 (`leaflet.js`, `leaflet.css`,
`images/`), licensed under BSD-2-Clause (see `LICENSE`).

Bundled locally rather than loaded from a CDN so the proximity map (see
`smart_assignment/webapp/static/app.js`) works fully offline, with no external
network dependency or third-party script trust decision at page-load time.

To upgrade: download the `leaflet` npm package, copy `dist/leaflet.js`,
`dist/leaflet.css`, and `dist/images/*` here, and update `LICENSE`/this file's
version note. `leaflet.js`'s trailing `//# sourceMappingURL=...` comment is
stripped since the corresponding `.map` file isn't vendored.
