/* JARVIS desktop client. Styling: web/theme.css. Icons: web/icons.js. Theme rules: docs/THEME.md */
let current = null, pending = null, busy = false, activeRequest = null, displayName = 'You';
let projects = [], chats = [], lastSignature = '', polling = false, dragging = null;
const $ = id => document.getElementById(id);

/* ---------- utilities ---------- */
async function api(path, method = 'GET', data) {
  let r;
  try { r = await fetch('/api' + path, { method, headers: { 'Content-Type': 'application/json' }, body: data === undefined ? undefined : JSON.stringify(data) }); }
  catch (e) { throw Error('Cannot reach the Jarvis service. Check that it is running and try again.'); }
  const raw = await r.text(); let obj;
  try { obj = raw ? JSON.parse(raw) : {}; } catch { obj = {}; }
  if (!r.ok) {
    if (r.status === 404 && obj.detail === 'Not Found') throw Error('The running Jarvis service is out of date. Install this update and restart the service, then try again.');
    const detail = typeof obj.detail === 'string' ? obj.detail : Array.isArray(obj.detail) ? 'Please check the supplied information.' : null;
    throw Error(detail && detail !== 'Not Found' ? detail : r.status >= 500 ? 'Jarvis encountered a server error. Please try again.' : 'The request could not be completed.');
  }
  return obj;
}
function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }
function atomNode(processing = true) {
  const a = el('span', 'atom' + (processing ? ' processing' : '')); a.setAttribute('aria-hidden', 'true');
  a.innerHTML = '<span class="orbit"></span><span class="orbit two"></span><span class="core"></span><span class="electron"></span><span class="electron two"></span>';
  return a;
}
function store(key, value) { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch { return null; } }
const chatById = id => chats.find(c => c.id === id);
const projectById = id => projects.find(p => p.id === id);
async function withCurrent(fn) {
  if (!current) return showError('No conversation selected', 'Create or select a chat first.');
  const c = chatById(current) || (await api('/chats/' + current)).conversation;
  fn(c);
}

/* ---------- tooltip ---------- */
(() => {
  const tip = $('tooltip'); let target = null;
  document.addEventListener('mouseover', e => {
    const t = e.target.closest('[data-tip]');
    if (t === target) return; target = t;
    if (!t) return tip.classList.remove('show');
    tip.textContent = t.dataset.tip;
    const r = t.getBoundingClientRect(); tip.classList.add('show');
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let top = r.bottom + 8; if (top + h > innerHeight - 4) top = r.top - h - 8;
    tip.style.top = top + 'px';
    tip.style.left = Math.max(6, Math.min(innerWidth - w - 6, r.left + r.width / 2 - w / 2)) + 'px';
  });
  for (const ev of ['mousedown', 'scroll', 'keydown']) document.addEventListener(ev, () => { tip.classList.remove('show'); target = null; }, true);
})();

/* ---------- sidebar ---------- */
function toggleSidebar() { document.body.classList.toggle('sidebar-collapsed'); store('jarvis-sidebar-collapsed', document.body.classList.contains('sidebar-collapsed')); }
if (store('jarvis-sidebar-collapsed') === 'true' || (store('jarvis-sidebar-collapsed') === null && innerWidth < 560)) document.body.classList.add('sidebar-collapsed');

function closedGroups() { try { return new Set(JSON.parse(store('jarvis-closed-groups') || '[]')); } catch { return new Set(); } }
function toggleGroup(key) { const s = closedGroups(); s.has(key) ? s.delete(key) : s.add(key); store('jarvis-closed-groups', JSON.stringify([...s])); renderSidebar(); }

async function refreshChats() {
  [chats, projects] = await Promise.all([api('/chats'), api('/projects')]);
  renderSidebar();
}

function chatRow(c) {
  const row = el('div', 'chat-row' + (c.id === current ? ' active' : ''));
  row.draggable = true; row.tabIndex = 0; row.title = c.title;
  row.append(icon('chat', 16), el('span', 'chat-title', c.title));
  const actions = el('div', 'row-actions');
  actions.append(
    iconButton('pencil', 'Rename', () => renameChat(c), 'ghost'),
    iconButton('move', 'Move to project', () => moveChat(c), 'ghost'),
    iconButton('trash', 'Delete', () => deleteChat(c), 'ghost danger'));
  row.append(actions);
  row.onclick = () => openChat(c.id);
  row.onkeydown = e => { if (e.key === 'Enter') openChat(c.id); if (e.key === 'Delete') deleteChat(c); if (e.key === 'F2') renameChat(c); };
  row.ondragstart = e => { dragging = c.id; row.classList.add('dragging'); e.dataTransfer.setData('text/plain', c.id); e.dataTransfer.effectAllowed = 'move'; };
  row.ondragend = () => { dragging = null; row.classList.remove('dragging'); document.querySelectorAll('.drop-target').forEach(n => n.classList.remove('drop-target')); };
  return row;
}

function group({ key, name, iconName, items, actions = [], projectId = null, onHeadDblClick }) {
  const closed = closedGroups().has(key);
  const g = el('section', 'group' + (closed ? '' : ' open'));
  const head = el('div', 'group-head');
  head.append(icon('chevron', 14), icon(iconName, 16), el('span', 'name', name), el('span', 'count', String(items.length)));
  head.firstChild.classList.add('chev');
  const bar = el('div', 'row-actions'); bar.append(...actions); head.append(bar);
  head.onclick = () => toggleGroup(key);
  if (onHeadDblClick) head.ondblclick = e => { e.preventDefault(); onHeadDblClick(); };
  const body = el('div', 'group-body');
  if (items.length) body.append(...items.map(chatRow)); else body.append(el('div', 'group-empty', projectId ? 'No chats yet — drop one here.' : 'Nothing unfiled.'));
  g.append(head, body);
  // Drag a chat onto a group to move it there.
  g.ondragover = e => { if (!dragging) return; e.preventDefault(); g.classList.add('drop-target'); };
  g.ondragleave = e => { if (!g.contains(e.relatedTarget)) g.classList.remove('drop-target'); };
  g.ondrop = async e => {
    e.preventDefault(); g.classList.remove('drop-target');
    const id = e.dataTransfer.getData('text/plain'); const c = chatById(id);
    if (c && (c.project_id || null) !== projectId) await moveTo(c, projectId);
  };
  return g;
}

function renderSidebar() {
  const root = $('chats'); root.replaceChildren();
  for (const p of projects) {
    root.append(group({
      key: p.id, name: p.name, iconName: 'folder', projectId: p.id,
      items: chats.filter(c => c.project_id === p.id),
      actions: [
        iconButton('chat-plus', 'New chat in ' + p.name, () => newChat(p.id), 'ghost'),
        iconButton('sliders', 'Shared context', () => editProject(p), 'ghost'),
        iconButton('pencil', 'Rename project', () => renameProject(p), 'ghost'),
        iconButton('trash', 'Delete project', () => deleteProject(p), 'ghost danger')],
    }));
  }
  root.append(group({ key: '__unfiled', name: 'Unfiled', iconName: 'inbox', items: chats.filter(c => !c.project_id || !projectById(c.project_id)) }));
}

/* ---------- dialogs ---------- */
let dialogClose = null;
function closeDialog() { $('dialog').hidden = true; dialogClose = null; }
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('dialog').hidden) closeDialog(); });
$('dialog').addEventListener('mousedown', e => { if (e.target === $('dialog')) closeDialog(); });

/* openDialog: themed replacement for prompt/confirm/alert.
   field: {label, value, multiline, allowEmpty}; body: custom node; submit: {icon, label} or null. */
function openDialog({ eyebrow = 'JARVIS / WORKSPACE', eyebrowIcon = 'chat', heading, description = '', field = null, body = null, danger = false, submit = { icon: 'check', label: 'Save' }, onSubmit }) {
  $('dialog').hidden = false;
  $('dialog-panel').className = 'dialog-panel' + (danger ? ' danger' : '');
  const eb = $('dialog-eyebrow'); eb.replaceChildren(icon(eyebrowIcon, 14), document.createTextNode(eyebrow));
  $('dialog-heading').textContent = heading; $('dialog-description').textContent = description;
  $('dialog-validation').textContent = '';
  const wrap = $('dialog-body'); wrap.replaceChildren();
  let input = null;
  if (field) {
    const id = 'dialog-field';
    const label = el('label', '', field.label || ''); label.htmlFor = id;
    input = field.multiline ? el('textarea') : el('input');
    input.id = id; input.value = field.value || ''; input.autocomplete = 'off';
    if (field.multiline) input.rows = 6; else input.maxLength = 120;
    if (field.placeholder) input.placeholder = field.placeholder;
    wrap.append(label, input);
  }
  if (body) wrap.append(body);
  const actions = $('dialog-actions'); actions.replaceChildren();
  actions.append(iconButton('close', submit ? 'Cancel' : 'Close', closeDialog, submit ? '' : 'primary'));
  let submitBtn = null;
  if (submit) { submitBtn = iconButton(submit.icon, submit.label, null, danger ? 'danger' : 'primary'); submitBtn.type = 'submit'; actions.append(submitBtn); }
  $('dialog-form').onsubmit = async e => {
    e.preventDefault(); if (!submit) return closeDialog();
    const val = input ? input.value.trim() : null;
    if (input && !val && !field.allowEmpty) { $('dialog-validation').textContent = 'This field cannot be empty.'; input.focus(); return; }
    submitBtn.disabled = true;
    try { await onSubmit(val); closeDialog(); }
    catch (err) { $('dialog-validation').textContent = err.message; submitBtn.disabled = false; }
  };
  hydrateIcons($('dialog'));
  (input || submitBtn || actions.firstChild).focus(); if (input && !field.multiline) input.select();
}
function showError(heading, message) {
  openDialog({ eyebrow: 'JARVIS / SYSTEM NOTICE', eyebrowIcon: 'alert', heading, description: message, danger: true, submit: null });
}

/* ---------- projects ---------- */
function newProject(thenMove = null) {
  openDialog({ eyebrow: 'JARVIS / PROJECTS', eyebrowIcon: 'folder-plus', heading: 'Create a project',
    description: 'Chats inside a project share its instructions, folders and recent context.',
    field: { label: 'PROJECT NAME' }, submit: { icon: 'folder-plus', label: 'Create project' },
    onSubmit: async name => {
      const p = await api('/projects', 'POST', { name }); await refreshChats();
      if (thenMove) await moveTo(thenMove, p.id); else await newChat(p.id);
    } });
}
function renameProject(p) {
  openDialog({ eyebrow: 'JARVIS / PROJECTS', eyebrowIcon: 'folder', heading: 'Rename project', field: { label: 'PROJECT NAME', value: p.name },
    onSubmit: async name => { await api('/projects/' + p.id, 'PATCH', { name }); await refreshChats(); renderHeader(); } });
}
function editProject(p) {
  openDialog({ eyebrow: 'JARVIS / PROJECTS', eyebrowIcon: 'sliders', heading: p.name + ' · shared context',
    description: 'Instructions every chat in this project receives. Chats here also see recent turns from each other.',
    field: { label: 'SHARED INSTRUCTIONS', value: p.context || '', multiline: true, allowEmpty: true, placeholder: 'e.g. We use ROS2 Jazzy; prefer C++17; target is a Jetson Orin.' },
    onSubmit: async context => { await api('/projects/' + p.id, 'PATCH', { context }); await refreshChats(); } });
}
function deleteProject(p) {
  const n = chats.filter(c => c.project_id === p.id).length;
  openDialog({ eyebrow: 'JARVIS / PROJECTS', eyebrowIcon: 'trash', danger: true, heading: 'Delete project “' + p.name + '”?',
    description: (n ? `Its ${n} chat${n > 1 ? 's are' : ' is'} kept and moved to Unfiled. ` : '') + 'The project’s shared instructions and folder grants are removed.',
    submit: { icon: 'trash', label: 'Delete project' },
    onSubmit: async () => { await api('/projects/' + p.id, 'DELETE'); await refreshChats(); renderHeader(); await refreshDetail(true); } });
}

/* ---------- chats ---------- */
async function newChat(project_id = null) {
  try { const c = await api('/chats', 'POST', { title: 'New chat', project_id }); await openChat(c.id); $('prompt').focus(); }
  catch (e) { showError('Request unsuccessful', e.message); }
}
async function openChat(id) { current = id; lastSignature = ''; store('jarvis-current', id); renderSidebar(); renderHeader(); await refreshDetail(true); }
function renameChat(c) {
  openDialog({ eyebrow: 'JARVIS / CHATS', eyebrowIcon: 'pencil', heading: 'Rename chat', field: { label: 'CHAT NAME', value: c.title },
    onSubmit: async title => { await api('/chats/' + c.id, 'PATCH', { title }); await refreshChats(); renderHeader(); } });
}
async function moveTo(c, projectId) {
  try { await api('/chats/' + c.id + '/project', 'PUT', { project_id: projectId }); await refreshChats(); if (c.id === current) { renderHeader(); await refreshDetail(true); } }
  catch (e) { showError('Unable to move chat', e.message); }
}
function moveChat(c) {
  const list = el('div', 'choice-list');
  const choice = (iconName, label, pid) => {
    const b = el('button', 'choice' + ((c.project_id || null) === pid ? ' current' : '')); b.type = 'button';
    b.append(icon(iconName, 16), el('span', '', label));
    if ((c.project_id || null) === pid) b.append(el('span', 'hint', 'current'));
    b.onclick = async () => { closeDialog(); if ((c.project_id || null) !== pid) await moveTo(c, pid); };
    return b;
  };
  for (const p of projects) list.append(choice('folder', p.name, p.id));
  list.append(choice('inbox', 'Unfiled (no project)', null));
  const create = el('button', 'choice'); create.type = 'button'; create.append(icon('folder-plus', 16), el('span', '', 'New project…'));
  create.onclick = () => newProject(c); list.append(create);
  openDialog({ eyebrow: 'JARVIS / CHATS', eyebrowIcon: 'move', heading: 'Move “' + c.title + '”',
    description: 'The chat will share context with the other chats in the project you pick.', body: list, submit: null });
}
function deleteChat(c) {
  openDialog({ eyebrow: 'JARVIS / CHATS', eyebrowIcon: 'trash', danger: true, heading: 'Delete “' + c.title + '”?',
    description: 'This permanently deletes the chat, its messages and its folder grants. This cannot be undone.',
    submit: { icon: 'trash', label: 'Delete permanently' },
    onSubmit: async () => {
      await api('/chats/' + c.id, 'DELETE');
      await refreshChats();
      if (current === c.id) { current = null; const next = chats[0]; if (next) await openChat(next.id); else showEmpty(); }
    } });
}

/* ---------- conversation pane ---------- */
function showEmpty() {
  current = null; renderHeader(); pending = null; showApproval();
  const pane = $('messages'); pane.replaceChildren(emptyState('Create a chat to get started.'));
}
function emptyState(text) {
  const e = el('div', 'empty-state'); e.append(atomNode(false), el('h1', '', 'What can I help you build?'), el('p', '', text)); return e;
}
function renderHeader() {
  const c = current && chatById(current); const crumb = $('crumb'); crumb.replaceChildren();
  if (!c) { $('title').textContent = 'Command center'; return; }
  const p = c.project_id && projectById(c.project_id);
  crumb.append(icon(p ? 'folder' : 'inbox', 13), document.createTextNode(p ? p.name : 'Unfiled'));
  $('title').textContent = c.title;
}
function showApproval() {
  const box = $('approval'); box.style.display = pending && !busy ? 'block' : 'none';
  if (pending && !busy) { const args = JSON.parse(pending.arguments); $('approval-title').textContent = pending.tool; $('approval-args').textContent = JSON.stringify(args.args, null, 2); }
}
function messageBox(role, content, model) {
  const box = el('article', 'message ' + role); const meta = el('div', 'meta', role === 'assistant' ? 'JARVIS' : role === 'user' ? displayName : role);
  if (role === 'assistant' && model) meta.append(el('span', 'model-tag', model));
  box.append(meta, el('div', '', content)); return box;
}
function showThinking() {
  if (!activeRequest || activeRequest.id !== current || $('thinking')) return;
  const box = el('article', 'message assistant'); box.id = 'thinking'; box.append(el('div', 'meta', 'JARVIS'));
  const content = el('div', 'thinking'); content.append(atomNode(), el('span', '', 'Processing your request…')); box.append(content);
  $('messages').append(box);
}
const atBottom = node => node.scrollHeight - node.scrollTop - node.clientHeight < 110;
async function refreshDetail(force = false) {
  if (!current) return;
  const id = current, d = await api('/chats/' + id); if (current !== id) return;
  const sig = JSON.stringify([d.messages, d.pending?.id, d.conversation, d.folders]);
  const pane = $('messages'); const stick = force || atBottom(pane); const oldTop = pane.scrollTop;
  if (force || sig !== lastSignature) {
    lastSignature = sig; pane.replaceChildren();
    if (!d.messages.length && !(activeRequest?.id === id)) pane.append(emptyState('Ask a question, inspect a project, or assign a task.'));
    for (const m of d.messages) pane.append(messageBox(m.role, m.content, m.model));
    if (activeRequest?.id === id && activeRequest.message && !d.messages.some(m => m.role === 'user' && m.content === activeRequest.message)) pane.append(messageBox('user', activeRequest.message));
    if (busy && activeRequest?.id === id) showThinking();
    pane.scrollTop = stick ? pane.scrollHeight : oldTop;
  }
  const known = chatById(id);
  if (!known || known.title !== d.conversation.title || known.project_id !== d.conversation.project_id) { Object.assign(known || {}, d.conversation); if (!known) await refreshChats(); else renderSidebar(); renderHeader(); }
  $('model').value = d.conversation.model_override || 'auto';
  const f = $('folders'); f.replaceChildren();
  if (d.folders.length) { f.append(icon('folder-open', 14), document.createTextNode(d.folders.join('  ·  '))); }
  pending = d.pending; showApproval();
}

async function loadConfig() {
  const data = await api('/config'); displayName = data.display_name || 'You';
  const sel = $('model'); sel.replaceChildren(new Option('Auto router', 'auto'));
  const byProvider = {};
  for (const m of data.available_models) (byProvider[m.split('/')[0]] ||= []).push(m);
  for (const [prov, models] of Object.entries(byProvider)) {
    const og = document.createElement('optgroup'); og.label = prov.toUpperCase();
    for (const m of models) og.append(new Option(m.slice(prov.length + 1), m));
    sel.append(og);
  }
  const status = $('system-status'); const down = (data.providers || []).filter(p => !p.ok);
  status.querySelector('.pulse').classList.toggle('off', down.length > 0);
  status.querySelector('span').textContent = down.length ? `${down.length} BACKEND${down.length > 1 ? 'S' : ''} OFFLINE` : 'AI BACKENDS · ONLINE';
  status.dataset.tip = (data.providers || []).map(p => `${p.name} (${p.type}): ${p.ok ? p.models.length + ' models' : 'offline'}`).join(' · ') || 'No backends configured';
}
$('model').addEventListener('change', async () => {
  if (current) try { await api('/chats/' + current + '/model', 'PUT', { model: $('model').value }); } catch (e) { showError('Request unsuccessful', e.message); }
});

/* Native picker: GTK dialog inside the desktop app, zenity via the service in browser mode.
   Resolves to an array of absolute paths ([] if cancelled); throws if no picker is available. */
async function pickPaths(kind) {
  if (window.jarvisDesktop?.native) return (await window.jarvisDesktop.pick(kind, true))?.paths || [];
  return (await api('/pick', 'POST', { kind, multiple: true })).paths || [];
}

function grantTarget() {
  const c = chatById(current); const p = c?.project_id && projectById(c.project_id);
  return { p, endpoint: p ? '/projects/' + p.id + '/folders' : '/chats/' + current + '/folders' };
}

async function attachPaths(paths) {
  const { endpoint } = grantTarget(); const failed = [];
  for (const path of paths) { try { await api(endpoint, 'POST', { path }); } catch (e) { failed.push(path + ': ' + e.message); } }
  await refreshDetail(true);
  if (failed.length) showError('Some items could not be attached', failed.join('\n'));
}

function grantFolder() {
  if (!current) return showError('No conversation selected', 'Create a chat first, then attach files or folders.');
  const { p } = grantTarget();
  const list = el('div', 'choice-list');
  const choice = (iconName, label, hint, onPick) => {
    const b = el('button', 'choice'); b.type = 'button';
    b.append(icon(iconName, 16), el('span', '', label), el('span', 'hint', hint));
    b.onclick = onPick; list.append(b);
  };
  const viaPicker = kind => async () => {
    closeDialog();
    let paths;
    try { paths = await pickPaths(kind); }
    catch (e) { return typePath(e.message); }
    if (paths.length) await attachPaths(paths);
  };
  choice('folder-open', 'Folder', 'browse', viaPicker('folder'));
  choice('file', 'Files', 'browse', viaPicker('file'));
  choice('pencil', 'Type a path', 'manual', () => typePath());
  openDialog({ eyebrow: 'JARVIS / WORKSPACE', eyebrowIcon: 'attach', heading: 'Attach to this chat',
    description: p ? `Attached items are shared with every chat in “${p.name}”.` : 'Only this chat gets access to what you attach.',
    body: list, submit: null });
}

function typePath(reason = '') {
  openDialog({ eyebrow: 'JARVIS / WORKSPACE', eyebrowIcon: 'attach', heading: 'Attach by path',
    description: (reason ? reason + '. ' : '') + 'Enter the absolute path of a file or folder.',
    field: { label: 'ABSOLUTE PATH', placeholder: '/home/you/projects/robot' }, submit: { icon: 'attach', label: 'Attach' },
    onSubmit: async path => { await api(grantTarget().endpoint, 'POST', { path }); await refreshDetail(true); } });
}

async function send() {
  if (!current) return showError('No conversation selected', 'Create a chat first.');
  if (busy || pending) return showError('Approval required', 'Resolve the pending permission request before sending another message.');
  const message = $('prompt').value.trim(); if (!message) return;
  const id = current; busy = true; activeRequest = { id, message }; $('send').disabled = true;
  $('prompt').value = ''; $('status').textContent = 'Jarvis is processing…';
  const pane = $('messages'); pane.querySelector('.empty-state')?.remove();
  pane.append(messageBox('user', message)); showThinking(); pane.scrollTop = pane.scrollHeight;
  try { const r = await api('/chats/' + id + '/messages', 'POST', { message, model: $('model').value }); if (id === current) $('status').textContent = r.status === 'approval' ? 'Waiting for approval' : ''; }
  catch (e) { showError('Request unsuccessful', e.message); $('status').textContent = ''; }
  finally { busy = false; activeRequest = null; $('send').disabled = false; if (id === current) await refreshDetail(true); }
}
async function resolveApproval(accept) {
  if (!pending || busy) return;
  const id = current, aid = pending.id; busy = true; activeRequest = { id, message: null }; pending = null; showApproval();
  $('status').textContent = accept ? 'Executing approved operation…' : 'Continuing after denial…'; showThinking();
  try { const r = await api('/chats/' + id + '/approvals/' + aid, 'POST', { accept }); if (id === current) $('status').textContent = r.status === 'approval' ? 'Waiting for approval' : ''; }
  catch (e) { showError('Request unsuccessful', e.message); }
  finally { busy = false; activeRequest = null; if (id === current) await refreshDetail(true); }
}
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });

/* ---------- boot ---------- */
hydrateIcons();
loadConfig().catch(e => { $('status').textContent = e.message; })
  .then(refreshChats)
  .then(async () => {
    const saved = store('jarvis-current');
    const target = chatById(saved) ? saved : chats[0]?.id;
    if (target) await openChat(target); else showEmpty();
  })
  .catch(e => { $('status').textContent = e.message; });
let tick = 0;
setInterval(async () => {
  if (busy || polling || dragging || !$('dialog').hidden) return;
  polling = true;
  try { if (++tick % 2 === 0) { await refreshChats(); if (current && !chatById(current)) { const next = chats[0]; next ? await openChat(next.id) : showEmpty(); } } if (current) await refreshDetail(false); } catch (e) { } finally { polling = false; }
}, 4000);
