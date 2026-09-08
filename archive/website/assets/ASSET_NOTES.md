# Original paper assets for the local research page

Primary plates, rendered from the camera-ready PDF without changing artwork:

- `sphere-torus-transport.png`: Figure 2, page 8, 2934×864. Best main visual: four stages across a sphere and a torus.
- `continental-drift.png`: Figure 1, page 7, 2934×492. Real geophysical application, from150 million years ago to the present.
- `dimension-scaling.png`: Figure 3, page 9, 2934×882. Keep plot labels and legend visible; the y-axis is KL +1 on a log scale.

All three have white backgrounds and generous resolution for a white figure plate on the dark page. They include their original panel/axis labels; external captions can stay concise. Each is genuine published-paper artwork, not a fresh experiment result.

Individual original panels are available as `sphere-{source,target,transport,pushforward}.png` and `torus-{source,target,transport,pushforward}.png`. These were copied byte-for-byte from the existing LaTeX figure directory. `continental-transport-map.png` is a high-resolution rendering of the existing vector transport-map PDF, useful as a single visual card.

`paper-metadata.json` contains the exact author order/affiliations, verbatim abstract, a concise grounded method summary, and paper links. `citation.bib` uses the stable arXiv identifier. The camera-ready front matter states ICML2026/PMLR306, but official proceedings page numbers and DOI were not independently retrieved, so none are invented.

`figure-provenance.json` records original source hashes, PDF pages/crop rectangles, dimensions and suggested captions. Main plates were visually checked after rendering: labels, plots and colorful surface/trajectory content remain intact.
