# Sahaya Vijay Jeyaraj — Personal Homepage

A single-page academic homepage for Dr. Sahaya Vijay Jeyaraj, Postdoctoral Researcher at the
National Institute for Materials Science (NIMS), Japan. Built as a self-contained static site
(one HTML file, embedded CSS, vector figures) so it can be hosted anywhere with no build step.

**Live site:** `https://<sahayavijayjeyaraj>.github.io`

## Sections

- **Home** — hero banner with name, role, and quick links
- **About** — biography and research background
- **Research Interests** — six areas, each with an explanatory scientific figure
- **Publications** — 15 peer-reviewed papers, plus live Citations / h-index from Google Scholar
- **Contact** — email, postal address, and collaboration note

## File structure

```
personal_homepage/
├── index.html            # the entire site (HTML + CSS in one file)
├── scholar-stats.json    # citation count, h-index, publication count
├── README.md             # this file
└── images/
    ├── Profile.jpg           # your profile photo  (replace with your own)
    ├── Img_Home_Page.png     # hero banner image   (replace with your own)
    ├── rc1_graph.svg         # figure: Chemical Graph Theory
    ├── rc2_autocatalysis.svg # figure: Autocatalysis & the Origin of Life
    ├── rc3_modelling.svg     # figure: Mathematical & Computational Modelling
    ├── rc4_eis.svg           # figure: Electrochemical Impedance Spectroscopy
    ├── rc5_camkii.svg        # figure: Synaptic Memory & the CaMKII Switch
    └── rc6_compute.svg       # figure: Scientific Computing & Open Tools
```

Keep this structure intact. `index.html` references the images and `scholar-stats.json`
with relative paths, so moving or renaming files will break the page.

## Updating content

- **Text, publications, contact details** — edit `index.html` directly.
- **Profile photo & banner** — replace `images/Profile.jpg` and `images/Img_Home_Page.png`
  with your own files, keeping the same names.
- **Citations / h-index** — edit `scholar-stats.json`:

  ```json
  {
    "publications": 15,
    "citations": 89,
    "hindex": 4
  }
  ```

  The page reads these values on load. Note: the fetch only runs over a real web server
  (e.g. GitHub Pages), not when opening the file locally with a double-click.

## Deploy to GitHub Pages

1. Create a **public** repository named `<your-username>.github.io`.
2. Upload `index.html`, `scholar-stats.json`, `README.md`, and the whole `images/` folder,
   keeping the structure above. Commit the changes.
3. Go to **Settings → Pages**, set *Source* to **Deploy from a branch**,
   branch **main**, folder **/ (root)**, and Save.
4. Wait ~1 minute, then open `https://<your-username>.github.io`.

To host under a different path instead, name the repo anything (e.g. `homepage`);
the site will then live at `https://<your-username>.github.io/homepage/`.

## Tech notes

- No frameworks, no dependencies, no build step — plain HTML/CSS.
- Research figures are hand-drawn **SVG** (scalable, sharp at any size).
- Fully responsive: cards and the hero banner adapt to phone, tablet, and desktop.

---
© 2026 Dr. Sahaya Vijay Jeyaraj · National Institute for Materials Science (NIMS), Japan
