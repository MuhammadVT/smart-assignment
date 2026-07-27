# Kids' app hub

A kid-friendly launcher that organizes the family's apps by **whose they are**.
Tap a face — Ibrahim, Fatimah, Yusuf, or *Everyone* — and that child's apps fill
the screen.

Self-contained on purpose: one HTML file, no CDN, no build step, no fonts to
download. It opens straight from disk, loads instantly on a tablet, and there is
nothing to break when the wifi is slow.

```
kids-hub/
  index.html   the whole UI (styles, markup, and the app list all in one file)
  serve.py     stdlib static server -- binds $PORT, no dependencies
  Dockerfile   what Railway builds
```

## Adding one of the kids' apps

Everything editable lives in a single `KIDS` array near the top of `index.html`,
under the comment banner that says so. Adding an app is one line:

```js
{
  id: "ibrahim",
  name: "Ibrahim",
  full: "Ibrahim Muhammad",
  emoji: "🦖",                       // shown on the profile picker
  colors: ["#2f8fff", "#00d1c1"],    // this child's theme
  apps: [
    { name: "Dino Math",  desc: "Practice times tables", url: "https://…", icon: "🦕" },
    { name: "Story Time", desc: "Read along out loud",   url: "https://…", icon: "📖" },
  ],
},
```

A child can have any number of apps (or none — the card then invites you to add
one). Apps open in a new tab so the hub stays one tab away, and the *Everyone*
view badges each tile with whose app it is.

To add a fourth child, copy a whole block and give them a new `id` and colors.

## Deploying on Railway

This directory is a **separate service** from the repo's Python project, so the
one setting that matters is the root directory:

1. Railway → your project → **New** → **GitHub Repo** → `MuhammadVT/smart-assignment`.
2. Open the new service → **Settings** → **Source** → set **Root Directory** to
   `kids-hub`. Without this, Railway builds the repo root (the Sysco
   slot-assignment service) instead of the hub.
3. Railway finds the `Dockerfile` and builds it. No start command to configure —
   the `CMD` handles it, and `$PORT` is injected automatically.
4. **Settings → Networking → Generate Domain** to get a public URL.

Redeploys are automatic on push to the default branch, so adding an app is: edit
`index.html`, commit, push.

### The three kids' apps

Each child's app should be **its own Railway service** (its own root directory or
its own repo), not part of this one. That keeps a crash or a redeploy in one
child's app from taking the other two down with it — and the hub only needs the
resulting URL pasted into `index.html`.

## Running it locally

```bash
python3 kids-hub/serve.py     # then open http://localhost:3000
```

Or just double-click `index.html` — it works straight from the filesystem too.

## Accessibility and device notes

- Touch targets are sized for small hands on a tablet; the profile pictures are
  ~120px and every app tile is at least 96px tall.
- Keyboard navigable, with visible focus rings, and the profile picker uses real
  buttons with `aria-pressed` rather than clickable `div`s.
- Light and dark themes both supported, following the device setting.
- Respects `prefers-reduced-motion`.
- The last-picked child is remembered in `localStorage`, so a kid who always uses
  the same tablet lands on their own apps.
