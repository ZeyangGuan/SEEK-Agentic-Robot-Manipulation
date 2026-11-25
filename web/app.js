'use strict';

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const onScroll = [];
const onResize = [];
let scrollQueued = false;

const runScroll = () => {
  scrollQueued = false;
  onScroll.forEach((handler) => handler());
};

const queueScroll = () => {
  if (scrollQueued) return;
  scrollQueued = true;
  window.requestAnimationFrame(runScroll);
};

const pinProgress = (section) => {
  const box = section.getBoundingClientRect();
  const distance = box.height - window.innerHeight;
  return distance > 0 ? clamp(-box.top / distance, 0, 1) : 0;
};

const HERO_TRAVEL = 320;
const hero = document.querySelector('[data-hero]');
if (hero && window.matchMedia('(min-height: 480px)').matches) {
  const stage = hero.querySelector('.hero-stage');
  const [leftCanvas, rightCanvas] = hero.querySelectorAll('canvas');
  const leftContext = leftCanvas.getContext('2d');
  const rightContext = rightCanvas.getContext('2d');
  const width = leftCanvas.width;
  const height = leftCanvas.height;
  const count = Number(hero.dataset.frames);
  const split = Number(hero.dataset.split);
  const endRight = Number(hero.dataset.t1);
  const endLeft = Number(hero.dataset.t2);
  const clocks = { left: hero.querySelector('[data-clock="left"]'), right: hero.querySelector('[data-clock="right"]') };
  const outcomes = { left: hero.querySelector('[data-out="left"]'), right: hero.querySelector('[data-out="right"]') };
  const frames = new Array(count);
  let target = 0;
  let current = 0;
  let drawnLeft = -1;
  let drawnRight = -1;
  let shown = -1;
  let running = false;

  const nearest = (index, last) => {
    for (let i = index; i >= 0; i -= 1) if (frames[i]) return i;
    for (let i = index + 1; i <= last; i += 1) if (frames[i]) return i;
    return -1;
  };

  const stamp = (seconds) => {
    const whole = Math.floor(seconds);
    return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
  };

  const render = (index) => {
    const left = nearest(index, count - 1);
    if (left >= 0 && left !== drawnLeft) {
      leftContext.drawImage(frames[left], 0, 0, width, height, 0, 0, width, height);
      drawnLeft = left;
    }
    const right = nearest(Math.min(index, split), split);
    if (right >= 0 && right !== drawnRight) {
      rightContext.drawImage(frames[right], width, 0, width, height, 0, 0, width, height);
      drawnRight = right;
    }
    if (index === shown) return;
    shown = index;
    const seconds = index <= split
      ? (index * endRight) / split
      : endRight + ((index - split) * (endLeft - endRight)) / (count - 1 - split);
    clocks.left.textContent = stamp(seconds);
    clocks.right.textContent = stamp(Math.min(seconds, endRight));
    outcomes.right.classList.toggle('on', index >= split);
    outcomes.left.classList.toggle('on', index >= count - 1);
  };

  const tick = () => {
    current += (target - current) * 0.22;
    if (Math.abs(target - current) < 0.05) current = target;
    render(Math.round(current));
    if (current === target) {
      running = false;
      return;
    }
    window.requestAnimationFrame(tick);
  };

  const kick = () => {
    if (running) return;
    running = true;
    window.requestAnimationFrame(tick);
  };

  const load = (index) => new Promise((resolve) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => {
      frames[index] = image;
      if (Math.abs(index - Math.round(current)) < 3) {
        drawnLeft = -1;
        drawnRight = -1;
        render(Math.round(current));
      }
      resolve();
    };
    image.onerror = resolve;
    image.src = `${hero.dataset.src}${String(index).padStart(3, '0')}.webp`;
  });

  const order = [];
  for (const stride of [16, 4, 1]) {
    for (let i = 0; i < count; i += stride) if (!order.includes(i)) order.push(i);
  }
  if (!order.includes(count - 1)) order.splice(1, 0, count - 1);
  let next = 0;
  const pump = () => {
    if (next >= order.length) return;
    const index = order[next];
    next += 1;
    load(index).then(pump);
  };
  load(order[next++]).then(() => {
    for (let i = 0; i < 6; i += 1) pump();
  });

  document.documentElement.classList.add('scrub-on');
  onScroll.push(() => {
    const travel = pinProgress(hero) * HERO_TRAVEL;
    const ease = (from, span) => {
      const t = clamp((travel - from) / span, 0, 1);
      return t * t * (3 - 2 * t);
    };
    const fade = ease(11, 75);
    const cardIn = ease(66, 31);
    const cardOut = ease(128, 35);
    const set = (name, value) => stage.style.setProperty(name, value.toFixed(3));
    set('--a', fade);
    set('--card-in', cardIn);
    set('--card-out', cardOut);
    set('--card', cardIn * (1 - cardOut));
    set('--b', cardOut);
    set('--light', 0.17 + 0.08 * fade + 0.75 * cardOut);
    set('--colour', 0.7 + 0.3 * cardOut);
    set('--zoom', 1.07 - 0.025 * fade - 0.045 * cardOut);
    stage.classList.toggle('untitled', fade >= 1);
    stage.classList.toggle('playing', cardOut >= 1);
    target = clamp((travel - 158) / 140, 0, 1) * (count - 1);
    kick();
  });
}

const method = document.querySelector('[data-method]');
if (method && window.matchMedia('(min-height: 600px)').matches) {
  const steps = Array.from(method.querySelectorAll('.steps li'));
  const view = method.querySelector('.viewport');
  const image = view.querySelector('img');
  const ratio = Number(image.getAttribute('width')) / Number(image.getAttribute('height'));
  const parse = (text) => text.split(',').map((value) => Number(value) / 100);
  const boxes = steps.map((step) => ({ wide: parse(step.dataset.box), narrow: parse(step.dataset.boxNarrow || step.dataset.box) }));
  const focus = document.createElement('span');
  focus.className = 'focus';
  focus.setAttribute('aria-hidden', 'true');
  view.appendChild(focus);
  let active = -1;

  const place = (size, room, centre) => (size <= room ? (room - size) / 2 : clamp(room / 2 - centre, room - size, 0));

  const frame = (index) => {
    const viewWidth = view.clientWidth;
    const [x0, y0, x1, y1] = viewWidth < 520 ? boxes[index].narrow : boxes[index].wide;
    const viewHeight = view.clientHeight;
    const imageHeight = viewWidth / ratio;
    const scale = Math.min(1 / (x1 - x0), viewHeight / ((y1 - y0) * imageHeight)) * 0.94;
    const shiftX = place(scale * viewWidth, viewWidth, (scale * (x0 + x1) * viewWidth) / 2);
    const shiftY = place(scale * imageHeight, viewHeight, (scale * (y0 + y1) * imageHeight) / 2);
    image.style.width = `${viewWidth}px`;
    image.style.transform = `translate(${shiftX.toFixed(1)}px, ${shiftY.toFixed(1)}px) scale(${scale.toFixed(4)})`;
    focus.style.left = `${(shiftX + scale * x0 * viewWidth).toFixed(1)}px`;
    focus.style.top = `${(shiftY + scale * y0 * imageHeight).toFixed(1)}px`;
    focus.style.width = `${(scale * (x1 - x0) * viewWidth).toFixed(1)}px`;
    focus.style.height = `${(scale * (y1 - y0) * imageHeight).toFixed(1)}px`;
    focus.classList.toggle('whole', index === boxes.length - 1);
  };

  const setStep = (index) => {
    if (index === active) return;
    active = index;
    steps.forEach((step, i) => step.classList.toggle('on', i === index));
    frame(index);
  };

  method.classList.add('pin');
  method.style.height = `${100 + steps.length * 58}vh`;
  view.addEventListener('click', (event) => event.preventDefault());
  steps.forEach((step, index) => {
    step.addEventListener('click', () => {
      const box = method.getBoundingClientRect();
      const distance = box.height - window.innerHeight;
      window.scrollTo({ top: window.scrollY + box.top + ((index + 0.5) / steps.length) * distance });
    });
  });
  onScroll.push(() => setStep(Math.min(steps.length - 1, Math.floor(pinProgress(method) * steps.length))));
  onResize.push(() => frame(Math.max(active, 0)));
  setStep(0);
}

document.querySelectorAll('[data-tabs]').forEach((group) => {
  const tabs = Array.from(group.querySelectorAll('[role="tab"]'));
  const video = group.querySelector('video');
  const goalOut = group.querySelector('[data-goal-out]');
  const labelOut = group.querySelector('[data-label-out]');
  const playButton = group.querySelector('[data-pp]');
  const track = group.querySelector('[data-track]');
  const fill = track ? track.querySelector('i') : null;
  const auto = group.hasAttribute('data-auto');
  let visible = false;
  let heldByUser = false;

  const play = () => {
    const attempt = video.play();
    if (attempt) attempt.catch(() => {});
  };

  const select = (tab, focus) => {
    tabs.forEach((other) => {
      const on = other === tab;
      other.setAttribute('aria-selected', String(on));
      other.tabIndex = on ? 0 : -1;
    });
    if (focus) tab.focus();
    const list = tab.parentElement;
    list.scrollLeft = tab.offsetLeft - list.offsetLeft - (list.clientWidth - tab.offsetWidth) / 2;
    video.poster = tab.dataset.poster;
    video.src = tab.dataset.src;
    if (goalOut) goalOut.textContent = tab.dataset.goal;
    if (labelOut) labelOut.textContent = tab.dataset.label;
    if (fill) fill.style.transform = 'scaleX(0)';
    if (auto && visible && !heldByUser) play();
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      if (tab.getAttribute('aria-selected') === 'true') return;
      if (auto) heldByUser = false;
      select(tab, false);
    });
    tab.addEventListener('keydown', (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      select(tabs[(index + step + tabs.length) % tabs.length], true);
    });
  });

  if (!auto) return;

  video.addEventListener('ended', () => {
    const current = tabs.findIndex((tab) => tab.getAttribute('aria-selected') === 'true');
    select(tabs[(current + 1) % tabs.length], false);
  });

  const syncButton = () => {
    if (!playButton) return;
    if (video.paused) {
      playButton.setAttribute('data-paused', '');
      playButton.setAttribute('aria-label', 'Play');
    } else {
      playButton.removeAttribute('data-paused');
      playButton.setAttribute('aria-label', 'Pause');
    }
  };
  video.addEventListener('play', syncButton);
  video.addEventListener('pause', syncButton);
  syncButton();

  if (playButton) {
    playButton.addEventListener('click', () => {
      if (video.paused) {
        heldByUser = false;
        play();
      } else {
        heldByUser = true;
        video.pause();
      }
    });
  }

  if (track && fill) {
    const draw = () => {
      if (video.duration) fill.style.transform = `scaleX(${video.currentTime / video.duration})`;
      if (!video.paused) window.requestAnimationFrame(draw);
    };
    video.addEventListener('play', () => window.requestAnimationFrame(draw));
    video.addEventListener('seeked', draw);
    track.addEventListener('click', (event) => {
      if (!video.duration) return;
      const box = track.getBoundingClientRect();
      video.currentTime = clamp((event.clientX - box.left) / box.width, 0, 1) * video.duration;
    });
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible && !heldByUser) play();
      if (!visible) video.pause();
    }, { threshold: 0.3 }).observe(video);
  } else {
    visible = true;
    play();
  }
});

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('in');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -6% 0px' });
  document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));

  const navLinks = Array.from(document.querySelectorAll('.nav-links a'));
  const byId = new Map(navLinks.map((link) => [link.getAttribute('href').slice(1), link]));
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((link) => link.removeAttribute('aria-current'));
      const link = byId.get(entry.target.id);
      if (link) link.setAttribute('aria-current', 'true');
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('main section[id]').forEach((section) => spy.observe(section));
}

const copyButton = document.getElementById('copy-bibtex');
const bibtex = document.getElementById('bibtex-text');
if (copyButton && bibtex) {
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent);
      copyButton.textContent = 'Copied';
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents(bibtex);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = 'Selected';
    }
    window.setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 1600);
  });
}

window.addEventListener('scroll', queueScroll, { passive: true });
window.addEventListener('resize', () => {
  onResize.forEach((handler) => handler());
  queueScroll();
});
runScroll();
