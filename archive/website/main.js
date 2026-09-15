import { createManifold } from './manifold.js';

const $ = (selector) => document.querySelector(selector);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const hero = createManifold($('#hero-manifold'), { mode: 'hero', shape: 'torus' });
const progressInput = $('#transport-progress');
const playButton = $('#play-transport');
let playing = false;
let currentShape = 'sphere';

function renderProgress(progress) {
  const value = Math.max(0, Math.min(1, progress));
  progressInput.value = String(Math.round(value * 1000));
  progressInput.style.setProperty('--progress', `${value * 100}%`);
  progressInput.setAttribute('aria-valuetext', `${Math.round(value * 100)} percent, ${value < 0.01 ? 'source distribution' : value > 0.99 ? 'target distribution' : 'intermediate distribution'}`);
  $('#progress-value').textContent = `t = ${value.toFixed(2)}`;
  $('#large-time').textContent = value.toFixed(2);
}

const demo = createManifold($('#demo-manifold'), { mode: 'demo', shape: currentShape, onProgress: renderProgress });
demo.setProgress(0);
demo.setPlaying(false);

function setPlaying(next) {
  playing = next;
  demo.setPlaying(next);
  playButton.setAttribute('aria-pressed', String(next));
  playButton.setAttribute('aria-label', next ? 'Pause transport animation' : 'Play transport animation');
  playButton.querySelector('span').textContent = next ? 'Pause' : 'Play';
  playButton.querySelector('path').setAttribute('d', next ? 'M5 4h4v12H5zM12 4h4v12h-4z' : 'm7 4 9 6-9 6z');
}

playButton.addEventListener('click', () => setPlaying(!playing));
progressInput.addEventListener('input', () => {
  setPlaying(false);
  const progress = Number(progressInput.value) / 1000;
  demo.setProgress(progress);
  renderProgress(progress);
});
$('#reset-transport').addEventListener('click', () => {
  setPlaying(false);
  demo.setProgress(0);
  renderProgress(0);
});
document.querySelectorAll('[data-shape]').forEach(button => {
  button.addEventListener('click', () => {
    currentShape = button.dataset.shape;
    document.querySelectorAll('[data-shape]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('is-active', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    demo.setShape(currentShape);
    $('#demo-manifold').setAttribute('aria-label', `Interactive schematic of particles moving from a uniform to a concentrated distribution on a ${currentShape}. Drag to rotate the view.`);
  });
});

const experiments = {
  sphere: { image: 'assets/figure-sphere.png', tag: 'SPHERICAL DISTRIBUTIONS / S²', title: 'Learning distributions on the sphere.', description: 'Source and target densities, the learned transport, and its pushforward on the sphere. Original results from Figure 2 of the paper.', alt: 'Figure 2 sphere panels: source density, target density, transport map, and learned pushforward.' },
  torus: { image: 'assets/figure-torus.png', tag: 'TORUS DISTRIBUTIONS / T²', title: 'Transport around a periodic world.', description: 'The same framework models a distribution on the torus, respecting its periodic geometry. Original results from Figure 2 of the paper.', alt: 'Figure 2 torus panels: source density, target density, transport map, and learned pushforward.' },
  drift: { image: 'assets/figure-drift.png', tag: 'CONTINENTAL DRIFT / S²', title: 'A journey across a changing planet.', description: 'An application of Riemannian transport to continental drift on the sphere. Original visualization from Figure 1 of the paper.', alt: 'Figure 1: continental-drift visualization on the sphere from the original paper.' }
};
let currentExperiment = 'sphere';
const tabs = [...document.querySelectorAll('[data-experiment]')];
function selectExperiment(key, focus = false) {
  currentExperiment = key;
  const entry = experiments[key];
  tabs.forEach(button => {
    const selected = button.dataset.experiment === key;
    button.classList.toggle('is-active', selected);
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  $('#experiment-panel').setAttribute('aria-labelledby', `tab-${key}`);
  $('#figure-stage-labels').hidden = key !== 'sphere';
  $('#experiment-image').src = entry.image;
  $('#experiment-image').alt = entry.alt;
  $('#experiment-tag').textContent = entry.tag;
  $('#experiment-title').textContent = entry.title;
  $('#experiment-description').textContent = entry.description;
}
tabs.forEach((button, index) => {
  button.addEventListener('click', () => selectExperiment(button.dataset.experiment));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      selectExperiment(tabs[next].dataset.experiment, true);
    }
  });
});
selectExperiment('sphere');

const dialog = $('#figure-dialog');
$('#expand-figure').addEventListener('click', () => {
  const entry = experiments[currentExperiment];
  $('#expanded-image').src = entry.image;
  $('#expanded-image').alt = entry.alt;
  $('#expanded-caption').textContent = entry.description;
  dialog.showModal();
});
$('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});

let copyTimeout;
$('#copy-citation').addEventListener('click', async () => {
  const button = $('#copy-citation');
  try {
    await navigator.clipboard.writeText($('#citation-text').textContent.trim());
    button.querySelector('span').textContent = 'Copied!';
    $('#copy-status').textContent = 'BibTeX citation copied to clipboard.';
    clearTimeout(copyTimeout);
    copyTimeout = setTimeout(() => { button.querySelector('span').textContent = 'Copy BibTeX'; }, 2200);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('#citation-text'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.querySelector('span').textContent = 'Select & copy';
    $('#copy-status').textContent = 'Citation selected. Use your browser’s copy command.';
  }
});

if (!reducedMotion.matches && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); } });
  }, { threshold: 0.08 });
  document.querySelectorAll('.intro-grid, .principles, .section-heading, .method-layout, .method-steps, .resource-links, .citation').forEach(element => { element.classList.add('reveal'); observer.observe(element); });
}

window.addEventListener('pagehide', event => {
  if (event.persisted) return;
  hero.destroy();
  demo.destroy();
});
