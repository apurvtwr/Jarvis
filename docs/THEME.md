# Jarvis visual theme — TRON

**Status: binding.** Every UI change (desktop web UI, dialogs, icons, launcher
art, future clients) must follow this document. If a request seems to need a
different look, extend this theme — do not introduce a second one. Changes to
the theme itself must be made here first, then in `web/theme.css`.

## The idea

Jarvis looks like a system running *inside the Grid*: near-black space, thin
glowing light-lines, precise geometry, and energy that means something.

* **Cyan is Jarvis.** Primary actions, focus, active state, the atom mark.
* **Orange is the opposing program.** Used *only* for danger: deleting,
  denying, warnings, errors, permission requests, offline backends.
* Everything else is quiet: dark panels, dim blue-grey text, 1px traces.

## Where things live

| Concern            | File               | Rule |
|--------------------|--------------------|------|
| Colors, fonts, radii, glow, component styles | `web/theme.css` | Only place CSS lives. Components use `var(--token)`; no raw hex outside `:root`. |
| Icons              | `web/icons.js`     | Only icon source. Add glyphs to `ICONS`; render with `icon()` / `iconButton()` / `data-icon`. |
| Markup             | `web/index.html`   | Structure only. No `style=` attributes, no `<style>` blocks. |
| Behaviour          | `web/app.js`       | Uses `openDialog()` / `showError()` — never `alert`, `confirm`, `prompt`. |
| App/launcher icon  | `packaging/jarvis.svg` (+ PNGs) | Cyan atom on dark. |

## Palette (tokens in `web/theme.css`)

| Token | Value | Use |
|---|---|---|
| `--bg-0` / `--bg-1` / `--bg-2` | `#050b11` / `#07131a` / `#0a151d` | Page, bars, deep panels |
| `--panel` / `--panel-raised` | `#0b1a23` / `#0d2831` | Cards, user messages |
| `--line` / `--line-strong` / `--line-hot` | `#17333f` / `#28505b` / `#4dbcca` | Borders: idle / control / hover |
| `--cyan` / `--cyan-bright` / `--cyan-soft` / `--cyan-deep` | `#45e4f4` / `#adffff` / `#75c9d7` / `#1698b0` | Energy, primary |
| `--orange` / `--orange-soft` | `#ff9e3d` / `#ffc58a` | Danger only |
| `--text` / `--text-bright` / `--text-dim` / `--text-faint` | `#dcecf3` / `#ecffff` / `#8ba6b3` / `#5f8794` | Copy |

Need a new shade? Add a token to `:root` and document it here.

## Typography

* UI text: **DM Sans**. Display (brand, headings, titles): **Space Grotesk**.
* Labels/eyebrows: 10–12px, UPPERCASE, letter-spacing `.12–.2em`, bold.
* Brand wordmark: `JARVIS`, Space Grotesk 700, letter-spacing `.24em`.

## Shape, light, motion

* Corners are tight: `--radius` (6px) for controls, `--radius-lg` (10px) for cards and dialogs. No pills, no big bubbles.
* Borders are 1px light-lines. Glow (`--glow`, `box-shadow` with cyan alpha) appears on hover/focus/active — not at rest.
* Accent traces: a 1–2px horizontal gradient line `transparent → cyan → transparent` on top of dialogs and assistant messages.
* Background of content areas: the faint 64px grid (`--grid`).
* Motion is short (`--ease`, 180ms) and functional. The spinning atom is the only "processing" indicator. Honour `prefers-reduced-motion`.
* Dark only. **No light mode**, no white surfaces, no pastel colors, no gradients other than cyan/orange energy and dark panel fades.

## Buttons and icons

* **Every action that can be an icon is an icon-only button** (`iconButton(name, label, onClick, variant)`), with a mandatory label that becomes `aria-label` and the Tron tooltip. Text buttons are not used.
* Variants: default (outlined), `primary` (cyan fill: the main action of a region — send, new chat, confirm), `danger` (orange: delete, deny), `ghost` (borderless, for row/hover actions), `lg` (44px).
* Icons: 24×24 viewBox, stroke-only, `currentColor`, stroke-width 1.6, round caps, geometric. **No emoji, no unicode glyphs as icons, no icon fonts, no third-party icon packs.**
* Destructive actions always confirm through a `danger` dialog stating exactly what is lost.
* Row actions in the sidebar appear on hover/focus only.

## Components checklist (for any new UI)

1. Uses only tokens from `theme.css`.
2. Actions are `iconButton`s with labels; danger is orange; primary is cyan.
3. Dialogs go through `openDialog` (eyebrow `JARVIS / AREA`, icon, heading, description).
4. Works in the collapsed (64px, icons-only) sidebar.
5. Keyboard reachable, visible cyan focus ring.
6. Looks like it belongs on the Grid. If in doubt: darker, thinner, more precise.
