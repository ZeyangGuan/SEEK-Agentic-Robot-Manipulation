(() => {
  if (!('IntersectionObserver' in window)) return;
  document.documentElement.classList.add('motion');
  const INLINE = new Set(['A', 'B', 'BR', 'CODE', 'EM', 'I', 'KBD', 'S', 'SMALL', 'SPAN', 'STRONG', 'SUB', 'SUP', 'U']);
  const tag = () => {
    document.querySelectorAll('.reveal').forEach((el) => {
      const kids = Array.from(el.children);
      if (kids.length && kids.every((c) => !INLINE.has(c.tagName))) {
        el.classList.add('stagger');
        kids.forEach((c, i) => c.style.setProperty('--i', Math.min(i, 8)));
      }
      const prev = el.previousElementSibling;
      if (prev && prev.classList.contains('reveal')) el.style.setProperty('--d', '0.16s');
    });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', tag);
  else tag();
})();

(() => {
  const bind = () => {
    document.querySelectorAll('[data-chapters]').forEach((group) => {
      const video = group.querySelector('video');
      const tabs = Array.from(group.querySelectorAll('[role="tab"]'));
      if (!video || !tabs.length) return;
      const starts = tabs.map((tab) => parseFloat(tab.dataset.t));
      const mark = (k) => tabs.forEach((tab, i) => {
        tab.setAttribute('aria-selected', String(i === k));
        tab.tabIndex = i === k ? 0 : -1;
      });
      const go = (k) => {
        mark(k);
        const seek = () => {
          video.currentTime = starts[k] + 0.05;
          const p = video.play();
          if (p && p.catch) p.catch(() => {});
        };
        if (video.readyState >= 1) seek();
        else {
          video.preload = 'auto';
          video.addEventListener('loadedmetadata', seek, { once: true });
          video.load();
        }
      };
      tabs.forEach((tab, k) => {
        tab.addEventListener('click', () => go(k));
        tab.addEventListener('keydown', (e) => {
          if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
          e.preventDefault();
          const j = (k + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
          tabs[j].focus();
          go(j);
        });
      });
      video.addEventListener('timeupdate', () => {
        let k = 0;
        starts.forEach((s, i) => { if (video.currentTime + 0.1 >= s) k = i; });
        if (tabs[k].getAttribute('aria-selected') !== 'true') mark(k);
      });
    });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
