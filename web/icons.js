/* JARVIS icon registry — the ONLY source of icons in the UI.
 * Rules (see docs/THEME.md): 24x24 viewBox, stroke-only, currentColor,
 * stroke-width 1.6, square-ish geometry. No emoji, no icon fonts, no
 * third-party icon packs. Add new glyphs here and reference them by name. */
const ICONS = {
  'chat':        '<path d="M4 5h16v11H9l-5 4z"/>',
  'chat-plus':   '<path d="M4 5h16v11H9l-5 4z"/><path d="M12 7.5v6M9 10.5h6"/>',
  'folder':      '<path d="M3 6.5h6l2 2h10V18H3z"/>',
  'folder-plus': '<path d="M3 6.5h6l2 2h10V18H3z"/><path d="M12 10.5v5M9.5 13h5"/>',
  'folder-open': '<path d="M3 18V6.5h6l2 2h8V11"/><path d="M3 18l3-7h15l-3 7z"/>',
  'move':        '<path d="M3 6.5h6l2 2h10V18H3z"/><path d="M8.5 13.5h7M13 11l2.5 2.5L13 16"/>',
  'inbox':       '<path d="M3 13l3-8h12l3 8v6H3z"/><path d="M3 13h5l1.5 2.5h5L16 13h5"/>',
  'sidebar':     '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M9 4v16"/><path d="M5.5 8h1.5M5.5 11h1.5"/>',
  'pencil':      '<path d="M4 20l1-4.5L15.5 5l3.5 3.5L8.5 19z"/><path d="M13.5 7l3.5 3.5"/>',
  'trash':       '<path d="M4 7h16M9.5 7V4h5v3M6 7l1 13h10l1-13"/><path d="M10 11v5.5M14 11v5.5"/>',
  'attach':      '<path d="M19.5 11.5l-7.8 7.8a4.8 4.8 0 01-6.8-6.8l8.2-8.2a3.2 3.2 0 014.5 4.5l-8 8a1.6 1.6 0 01-2.3-2.3l7.4-7.4"/>',
  'send':        '<path d="M4 12L20 4l-6 16-2.8-7.2z"/><path d="M11.2 12.8L20 4"/>',
  'check':       '<path d="M5 12.5l4.5 4.5L19 7"/>',
  'close':       '<path d="M6 6l12 12M18 6L6 18"/>',
  'sliders':     '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><rect x="13" y="5" width="4" height="4"/><rect x="7" y="15" width="4" height="4"/>',
  'chevron':     '<path d="M9 6l6 6-6 6"/>',
  'alert':       '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.2v.3"/>',
  'cpu':         '<rect x="6.5" y="6.5" width="11" height="11"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21"/>',
  'shield':      '<path d="M12 3l8 3v6c0 4.5-3.4 7.8-8 9-4.6-1.2-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
};

function icon(name, size = 18) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('class', 'icon');
  svg.innerHTML = ICONS[name] || ICONS['alert'];
  return svg;
}

/* Icon-only button. `label` is mandatory: it becomes aria-label and the tooltip. */
function iconButton(name, label, onClick, variant = '') {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'icon-btn' + (variant ? ' ' + variant : '');
  b.setAttribute('aria-label', label);
  b.dataset.tip = label;
  b.append(icon(name));
  if (onClick) b.addEventListener('click', e => { e.stopPropagation(); onClick(e); });
  return b;
}

/* Hydrate static markup: <button data-icon="name" data-tip="Label"></button> */
function hydrateIcons(root = document) {
  for (const node of root.querySelectorAll('[data-icon]')) {
    if (node.querySelector('svg.icon')) continue;
    node.prepend(icon(node.dataset.icon, Number(node.dataset.iconSize) || 18));
    if (node.dataset.tip && !node.getAttribute('aria-label')) node.setAttribute('aria-label', node.dataset.tip);
  }
}
