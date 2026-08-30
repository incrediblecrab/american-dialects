'use strict';

const $ = (id) => document.getElementById(id);

const state = { session: null, n: null, asked: 0, question: null, busy: false };

async function api(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

function show(name) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.remove('is-active'));
  $('screen-' + name).classList.add('is-active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function progress() {
  $('progress-fill').style.width = (100 * state.asked / state.n) + '%';
}

// ------------------------------------------------------------------- quiz

function renderQuestion(q) {
  state.question = q;
  $('step').textContent = `Question ${state.asked + 1} of ${state.n}`;
  $('question').textContent = q.text;

  const box = $('choices');
  box.innerHTML = '';
  q.choices.forEach((c) => {
    const b = document.createElement('button');
    b.className = 'choice';
    b.textContent = c.text;
    b.addEventListener('click', () => answer(c.id, b));
    box.appendChild(b);
  });

  // Re-trigger the entrance animation so each question arrives the same way.
  const screen = $('screen-quiz');
  screen.classList.remove('is-active');
  void screen.offsetWidth;
  screen.classList.add('is-active');
}

async function answer(choiceId, el) {
  if (state.busy) return;
  state.busy = true;
  if (el) el.classList.add('is-picked');

  const r = await api('/api/answer', {
    session: state.session,
    question: state.question.id,
    choice: choiceId,
  });
  state.asked = r.asked;
  progress();

  if (r.done || !r.question) {
    finish(r.guess);
  } else {
    renderQuestion(r.question);
  }
  state.busy = false;
}

// ----------------------------------------------------------------- result

function list(el, rows, label) {
  el.innerHTML = '';
  rows.forEach((row) => {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = label(row);
    const pct = document.createElement('span');
    pct.className = 'pct';
    pct.textContent = (100 * row.p).toFixed(1) + '%';
    li.append(name, pct);
    el.appendChild(li);
  });
}

function area(km2) {
  if (km2 >= 1e6) return (km2 / 1e6).toFixed(1) + ' million km²';
  if (km2 >= 1e4) return Math.round(km2 / 1e3) + ',000 km²';
  return Math.round(km2).toLocaleString() + ' km²';
}

function finish(g) {
  if (!g) {
    $('place').textContent = 'Not enough to go on';
    $('confidence').textContent = 'Answer at least one question and try again.';
    show('result');
    return;
  }

  const top = g.places[0];
  $('place').textContent = top ? `${top.name}, ${top.state}` : g.state;

  const bits = [`From ${g.answered} answer${g.answered === 1 ? '' : 's'}`];
  if (top) bits.push(`a ${(100 * top.p).toFixed(1)}% bet on that metro area`);
  bits.push(`80% of the belief inside ${area(g.km2)}`);
  $('confidence').textContent = bits.join(', ') + '.';

  list($('places'), g.places, (r) => `${r.name}, ${r.state}`);
  list($('states'), g.states, (r) => r.state);

  $('map').src = `/api/map/${state.session}.png?t=${Date.now()}`;
  document.querySelector('.map-wrap').hidden = false;
  $('truth-saved').hidden = true;
  $('truth-input').value = '';
  show('result');
}

// A lost session (server restarted mid-game) should degrade quietly rather
// than leaving a broken image on the page.
$('map').addEventListener('error', () => {
  document.querySelector('.map-wrap').hidden = true;
});

// ------------------------------------------------------------------ flow

async function start() {
  state.busy = true;
  const r = await api('/api/start', {});
  state.session = r.session;
  state.n = r.n;
  state.asked = 0;
  progress();
  show('quiz');
  renderQuestion(r.question);
  state.busy = false;
}

$('start').addEventListener('click', start);
$('again').addEventListener('click', start);

$('skip').addEventListener('click', async () => {
  if (state.busy) return;
  state.busy = true;
  const r = await api('/api/answer', {
    session: state.session,
    question: state.question.id,
    choice: null,
  });
  state.asked = r.asked;
  progress();
  if (r.done || !r.question) finish(r.guess);
  else renderQuestion(r.question);
  state.busy = false;
});

$('finish').addEventListener('click', async () => {
  if (state.busy) return;
  state.busy = true;
  const r = await api('/api/finish', { session: state.session });
  $('progress-fill').style.width = '100%';
  finish(r.guess);
  state.busy = false;
});

$('truth-save').addEventListener('click', async () => {
  const truth = $('truth-input').value.trim();
  if (!truth) return;
  await api('/api/log', { session: state.session, truth });
  $('truth-saved').hidden = false;
});

$('truth-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('truth-save').click();
});

// Number keys pick an answer, because a party trick should be fast.
document.addEventListener('keydown', (e) => {
  if (!$('screen-quiz').classList.contains('is-active')) return;
  const n = parseInt(e.key, 10);
  if (!Number.isNaN(n) && n >= 1) {
    const b = $('choices').children[n - 1];
    if (b) b.click();
  }
});
