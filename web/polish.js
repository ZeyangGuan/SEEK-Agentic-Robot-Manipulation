// Smoother section reveals: each .reveal block brings its children in one after another, with a short rise and a blur
// that clears, and the right column of a split follows its left column. When a block is revealed is still decided in
// app.js (it adds .in); this file only tags the blocks. Without IntersectionObserver nothing changes. Like the rise it
// replaces (style.css), it runs whatever the reduced-motion setting.
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
