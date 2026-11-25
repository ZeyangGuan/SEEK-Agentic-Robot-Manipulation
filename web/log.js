'use strict';

(() => {
  const root = document.querySelector('[data-log]');
  if (!root) return;

  const make = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = text;
    return element;
  };
  const num = (value, digits = 1) => (value === null || value === undefined ? '–' : value.toFixed(digits));
  const pct = (value) => (value === null || value === undefined ? '–' : `${value.toFixed(0)}%`);
  const pct1 = (value) => (value === null || value === undefined ? '–' : `${value.toFixed(1)}%`);
  const variantName = (variant) => (variant === 'random' ? 'randomized' : 'standard');

  const cellTitle = (episode) => {
    const where = `seed ${episode.seed}, ${variantName(episode.variant)} layout ${episode.layout}`;
    if (episode.status === 'done') {
      return `Episode ${episode.episode}: score ${num(episode.score, 0)}, ${episode.requests} model requests, ${where}`;
    }
    const state = { running: 'running', queued: 'queued', check: 'held for inspection' }[episode.status];
    return `Episode ${episode.episode}: ${state}, ${where}`;
  };

  const cell = (episode) => {
    const box = make('span', `log-cell is-${episode.status}`);
    box.title = cellTitle(episode);
    if (episode.status === 'done') {
      box.textContent = num(episode.score, 0);
      if (episode.score >= 99.999) box.classList.add('is-full');
      else if (episode.score <= 0) box.classList.add('is-zero');
      else box.style.setProperty('--s', (episode.score / 100).toFixed(3));
    } else if (episode.status === 'check') {
      box.textContent = '!';
    }
    if (episode.variant === 'random') box.classList.add('is-random');
    return box;
  };

  const stat = (value, small, label) => {
    const box = make('div', 'stat');
    const big = make('b', null, value);
    if (small) big.append(make('small', null, small));
    box.append(big, make('span', null, label));
    return box;
  };

  const episodeTable = (task) => {
    const table = make('table', 'log-eps');
    const head = make('tr');
    ['Episode', 'Seed', 'Layout', 'Score', 'Model requests', 'Steps'].forEach((label) => head.append(make('th', null, label)));
    const thead = make('thead');
    thead.append(head);
    const tbody = make('tbody');
    task.episodes.forEach((episode) => {
      const row = make('tr', `is-${episode.status}`);
      const layout = `${variantName(episode.variant)} ${episode.layout}` + (episode.replaced_layout ? ` (replaces ${episode.replaced_layout})` : '');
      const score = episode.status === 'done' ? num(episode.score, 0) : { running: 'running', queued: 'queued', check: 'held' }[episode.status];
      [episode.episode, episode.seed, layout, score, episode.requests ?? '–', episode.steps ?? '–'].forEach((value) => row.append(make('td', null, value)));
      tbody.append(row);
    });
    table.append(thead, tbody);
    return table;
  };

  const taskRow = (task, complete, other) => {
    const details = make('details', 'log-row');
    details.dataset.category = task.category;
    const summary = make('summary');
    const name = make('span', 'log-name');
    name.append(make('b', null, task.name), make('i', null, task.category));
    const cells = make('span', 'log-cells');
    task.episodes.forEach((episode) => cells.append(cell(episode)));
    const scored = complete ? '' : `${task.done}/${task.total} scored · `;
    const versus = other ? ` · Astra ${num(other.score)}` : '';
    summary.append(
      name,
      make('span', 'log-meta', task.done
        ? `${scored}${pct(task.success)} success · ${num(task.requests)} requests${versus}`
        : (task.episodes.some((episode) => episode.status === 'running') ? 'Running' : 'Queued')),
      make('span', 'log-num log-done', `${task.done}/${task.total}`),
      make('span', 'log-num log-score', num(task.score)),
      make('span', 'log-num log-astra', other ? num(other.score) : '–'),
      make('span', 'log-num log-success', pct(task.success)),
      make('span', 'log-num log-req', num(task.requests)),
      cells,
    );
    const body = make('div', 'log-body');
    body.append(episodeTable(task));
    details.append(summary, body);
    return details;
  };

  const categoryRow = (category, tasks, other) => {
    const details = make('details', 'log-row log-cat-row');
    details.dataset.category = category.name;
    const summary = make('summary');
    const name = make('span', 'log-name');
    name.append(make('b', null, category.name));
    const versus = other ? ` · Astra ${num(other.score)}` : '';
    summary.append(
      name,
      make('span', 'log-meta', `${category.tasks} tasks · ${pct1(category.success)} success · ${num(category.requests)} requests${versus}`),
      make('span', 'log-num log-done', `${category.done}/${category.total}`),
      make('span', 'log-num log-score', num(category.score)),
      make('span', 'log-num log-astra', other ? num(other.score) : '–'),
      make('span', 'log-num log-success', pct1(category.success)),
      make('span', 'log-num log-req', num(category.requests)),
      make('span', 'log-cat-hint', `${category.tasks} tasks · ${category.total} episodes`),
    );
    const body = make('div', 'log-tasks');
    body.append(...tasks);
    details.append(summary, body);
    return details;
  };

  const versusBars = (seek, astra) => {
    const bars = make('div', 'log-vs');
    [['seek', seek], ['astra', astra]].forEach(([who, value]) => {
      const bar = make('span', `log-vs-bar is-${who}`);
      const fill = make('i');
      fill.style.setProperty('--p', (value / 100).toFixed(4));
      bar.append(fill);
      bars.append(bar);
    });
    return bars;
  };

  const categoryTile = (category, complete, other) => {
    const tile = make('div', 'tile log-cat');
    tile.dataset.category = category.name;
    const top = make('p', 'log-cat-name', category.name);
    const score = make('b', 'log-cat-score', num(category.score));
    if (other) score.append(make('small', null, ` / ${num(other.score)}`));
    tile.append(top, score);
    if (other) {
      tile.append(
        make('p', 'log-cat-meta', `${pct1(category.success)} / ${pct1(other.success)} success`),
        make('p', 'log-cat-meta log-cat-meta-next', `${num(category.requests)} / ${num(other.requests)} requests`),
      );
    } else {
      tile.append(make('p', 'log-cat-meta', `${pct1(category.success)} success · ${num(category.requests)} requests`));
    }
    if (other) tile.append(versusBars(category.score, other.score));
    if (!complete) {
      const bar = make('div', 'log-bar');
      const fill = make('i');
      fill.style.setProperty('--p', (category.done / category.total).toFixed(4));
      bar.append(fill);
      tile.append(bar, make('p', 'log-cat-count', `${category.done} / ${category.total} episodes · ${category.tasks} tasks`));
    }
    return tile;
  };

  const versusPlot = (data, compare) => {
    const plot = document.querySelector('[data-versus-plot]');
    if (!plot || !compare) return;
    const direct = new Map(compare.tasks.map((task) => [task.name, task]));
    const groupsOf = data.summary.categories.map((category) => ({
      name: category.name,
      tasks: data.tasks
        .filter((task) => task.category === category.name && direct.has(task.name))
        .map((task) => ({ task, other: direct.get(task.name) }))
        .sort((a, b) => (b.task.score - b.other.score) - (a.task.score - a.other.score) || b.task.score - a.task.score),
    }));
    const requestTop = Math.max(50, Math.ceil(Math.max(...groupsOf.flatMap((g) => g.tasks.map(({ task, other }) => Math.max(task.requests, other.requests)))) / 50) * 50);
    const panels = [
      { key: 'score', title: 'Task score', top: 100, step: 25, better: (s, a) => s >= a, unit: '' },
      { key: 'requests', title: 'Model requests per episode', top: requestTop, step: 50, better: (s, a) => s <= a, unit: ' requests' },
    ];
    const children = [];
    panels.forEach((panel, p) => {
      const head = make('p', 'vs-title', panel.title);
      const axis = make('div', 'vs-axis');
      const area = make('div', 'vs-area');
      for (let level = 0; level <= panel.top; level += panel.step) {
        const y = `${(level / panel.top) * 100}%`;
        const label = make('span', null, String(level));
        label.style.setProperty('--y', y);
        axis.append(label);
        const line = make('i', 'vs-gl');
        line.style.setProperty('--y', y);
        area.append(line);
      }
      const groups = make('div', 'vs-groups');
      let index = 0;
      groupsOf.forEach((g) => {
        const group = make('div', 'vs-group');
        group.style.flexGrow = String(g.tasks.length);
        g.tasks.forEach(({ task, other }) => {
          const s = task[panel.key];
          const a = other[panel.key];
          const col = make('div', `vs-col${panel.better(s, a) ? '' : ' is-down'}`);
          col.title = `${task.name}: SEEK ${num(s)}${panel.unit}, gpt-6.2-astra acting directly ${num(a)}${panel.unit}`;
          const at = (value) => `${(value / panel.top) * 100}%`;
          col.style.setProperty('--a', at(a));
          col.style.setProperty('--s', at(s));
          col.style.setProperty('--lo', at(Math.min(a, s)));
          col.style.setProperty('--len', at(Math.abs(s - a)));
          col.style.setProperty('--i', String(index));
          index += 1;
          col.append(make('i', 'vs-line'), make('i', 'vs-dot vs-astra'), make('i', 'vs-dot vs-seek'));
          group.append(col);
        });
        groups.append(group);
      });
      area.append(groups);
      const body = make('div', 'vs-body');
      body.append(area);
      if (p === panels.length - 1) {
        const cats = make('div', 'vs-cats');
        groupsOf.forEach((g) => {
          const label = make('span', null, g.name);
          label.style.flexGrow = String(g.tasks.length);
          cats.append(label);
        });
        body.append(cats);
      }
      const row = make('div', `vs-panel vs-${panel.key}`);
      row.append(head, axis, body);
      children.push(row);
    });
    plot.replaceChildren(...children);
  };

  const render = ([data, compare]) => {
    const average = data.summary.average;
    const progress = data.progress;
    const complete = data.state === 'complete';
    const otherCategory = new Map((compare ? compare.summary.categories : []).map((category) => [category.name, category]));
    const otherTask = new Map((compare ? compare.tasks : []).map((task) => [task.name, task]));
    const vs = compare ? compare.summary.average : null;
    const versus = (value, format) => (vs ? ` / ${format(value)}` : null);
    root.classList.toggle('is-complete', complete);
    root.classList.toggle('has-versus', Boolean(compare));
    root.querySelectorAll('[data-log-live]').forEach((element) => { element.hidden = complete; });
    root.querySelectorAll('[data-log-versus]').forEach((element) => { element.hidden = !compare; });
    root.querySelector('[data-log-stats]').replaceChildren(
      complete
        ? stat(String(progress.total), null, 'episodes')
        : stat(String(progress.done), ` / ${progress.total}`, progress.running ? `episodes scored, ${progress.running} running now` : 'episodes scored'),
      stat(num(average.score), vs && versus(vs.score, num), 'average task score'),
      stat(pct1(average.success), vs && versus(vs.success, pct1), 'of episodes reach the full score'),
      stat(num(average.requests), vs && versus(vs.requests, num), 'model requests per episode'),
    );
    root.querySelector('[data-log-cats]').replaceChildren(...data.summary.categories.map((category) => categoryTile(category, complete, otherCategory.get(category.name))));
    const rows = root.querySelector('[data-log-rows]');
    rows.replaceChildren(...data.summary.categories.map((category) => categoryRow(
      category,
      data.tasks.filter((task) => task.category === category.name).map((task) => taskRow(task, complete, otherTask.get(task.name))),
      otherCategory.get(category.name),
    )));

    const filters = root.querySelector('[data-log-filters]');
    const names = ['All', ...data.summary.categories.map((category) => category.name)];
    filters.replaceChildren(...names.map((label, index) => {
      const button = make('button', 'log-pill', label);
      button.type = 'button';
      button.setAttribute('aria-pressed', index === 0 ? 'true' : 'false');
      button.addEventListener('click', () => {
        filters.querySelectorAll('button').forEach((other) => other.setAttribute('aria-pressed', other === button ? 'true' : 'false'));
        rows.querySelectorAll(':scope > .log-cat-row').forEach((group) => {
          group.hidden = label !== 'All' && group.dataset.category !== label;
          if (label !== 'All' && group.dataset.category === label) group.open = true;
        });
      });
      return button;
    }));

    const stamp = new Date(data.updated);
    const time = root.querySelector('[data-log-time]');
    time.dateTime = data.updated;
    time.textContent = stamp.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
    root.querySelector('[data-log-state]').textContent = complete ? 'Complete' : 'In progress';
    root.classList.add('is-ready');
    versusPlot(data, compare);
  };

  const load = (src) => fetch(src, { cache: 'no-store' }).then((response) => {
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  });

  Promise.all([load(root.dataset.src), root.dataset.compare ? load(root.dataset.compare).catch(() => null) : null])
    .then(render)
    .catch(() => {
      root.querySelector('[data-log-rows]').replaceChildren(make('p', 'log-empty', 'The evaluation log could not be loaded.'));
    });
})();
