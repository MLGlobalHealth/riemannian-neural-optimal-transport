# Local browser verification

Verified on 8 September 2026 using an isolated Chrome session. The project page has not been deployed.

- Desktop and mobile layouts were visually inspected; no horizontal overflow at 320, 390, 768, or 1440 pixels.
- Play/pause, keyboard scrubbing, reset, and sphere/torus switching work.
- Canvas rotation is available by dragging or arrow keys. Vertical touch swipes still scroll the page.
- Gallery tabs support arrow keys, Home, End, and wrapping; the expanded figure closes with Escape and returns focus.
- Copy BibTeX writes the exact displayed citation.
- PDF and ZIP downloads were checked against local file hashes.
- Reduced-motion preference stops automatic hero animation.
- All original figure/PDF hashes match their provenance records.
- The browser reported no JavaScript errors or failed asset responses.

The in-app browser connection was unavailable, so visual and interaction checks used standalone Playwright with installed Chrome and a fresh ephemeral profile. No browser tooling is required to run the page.
