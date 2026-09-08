# RNOT project page

A self-contained, local research page for **Riemannian Neural Optimal Transport**. No build step, JavaScript dependencies, remote font requests, analytics, or deployment service is needed.

From this directory, start the preview:

```sh
python3 -m http.server 4173 --bind 127.0.0.1
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173). Use a local HTTP server: the browser loads `main.js` and `manifold.js` as JavaScript modules.

## Page features

- An animated particle torus, with mouse/touch and keyboard rotation.
- A transport explorer with sphere/torus selection, play/pause, scrubbing, and reset.
- Original figures, keyboard-accessible gallery tabs, and an expandable image dialog.
- The paper abstract, method explanation, local manuscript/code downloads, and a copyable citation.
- Responsive layouts and reduced-motion support. Animations suspend when their canvas is offscreen or the tab is hidden.

`index.html` contains the paper content, `styles.css` the visual design, `main.js` the page interactions, and `manifold.js` the procedural illustrations. All required assets are served locally.

## Assets and interpretation

The Canvas illustrations are **schematic explanations**, not trained model outputs. The gallery uses original artwork from the camera-ready paper. PDF/figure provenance, metadata, and captions are in [assets/figure-provenance.json](assets/figure-provenance.json) and [assets/ASSET_NOTES.md](assets/ASSET_NOTES.md).

`assets/paper.pdf` is the unchanged local camera-ready manuscript. `assets/riemannian-neural-ot-code.zip` is a Git archive of `finalMLGH` commit `99bacc7`, containing the experiment library, pinned reproduction setup, and verification report. It excludes the website itself. This download reflects the recorded partial reproduction findings; the page does not claim that every published experiment has been rerun.

The local font files are Inter Display, Space Grotesk, and JetBrains Mono. Their SIL Open Font License texts are included in `assets/fonts/`.

This is a local preview. No hosting configuration or deployment action is included.
