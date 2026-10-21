# web/

`web/` holds the site generators and the static assets they ship. `build.py` runs the gates and builds one model's site; `assemble.py` combines every live model into the whole static site. The other modules are helpers the two entry points import (homepage, comparison, benchmarks, payload, discovery pages). `web/assets/` holds the CSS and JS the generators read: most of it is inlined into the generated pages, and `home.css` and `home.js` are copied beside the homepage.

- Build one model: `QCL_ROOT=models/<slug> python3 web/build.py` (every live model: `for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 web/build.py; done`).
- Assemble the site: `python3 web/assemble.py _site`.

Output goes to `models/<slug>/site/` and `_site/`. Both are git-ignored.
