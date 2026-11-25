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
  if (!('IntersectionObserver' in window) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const parse = (text) => {
    const m = /^\s*(-?\d+(?:\.(\d+))?)(.*)$/.exec(text);
    return m ? { value: parseFloat(m[1]), digits: m[2] ? m[2].length : 0, suffix: m[3] } : null;
  };
  const bind = () => {
    document.querySelectorAll('.stat b').forEach((b) => {
      const from = b.querySelector('.from');
      const node = b.lastChild;
      if (!from || !node || node.nodeType !== Node.TEXT_NODE) return;
      const start = parse(from.textContent);
      const end = parse(node.textContent);
      if (!start || !end) return;
      const show = (value) => { node.textContent = `${value.toFixed(end.digits)}${end.suffix}`; };
      show(start.value);
      const stat = b.closest('.stat') || b;
      const observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        const delay = 450 + (parseFloat(getComputedStyle(stat).getPropertyValue('--i')) || 0) * 90;
        const duration = 1500;
        let t0 = null;
        const step = (now) => {
          if (t0 === null) t0 = now + delay;
          const t = Math.min(Math.max((now - t0) / duration, 0), 1);
          const k = 1 - Math.pow(1 - t, 3);
          show(start.value + (end.value - start.value) * k);
          if (t < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
      }, { threshold: 0.4 });
      observer.observe(stat);
    });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
