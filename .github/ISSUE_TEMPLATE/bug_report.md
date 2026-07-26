---
name: Bug report
about: Report something that isn't working as documented
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug and what you expected instead.

**Minimal reproducer**
The smallest script that shows the problem (geometry, the `run_*` call and its
config dict). A failing case with concrete numbers is worth far more than a
description.

```python
from cavsim2d import Study, EllipticalCavity
# ...
```

**Traceback / output**
Paste the full error, or the wrong numbers you got vs. the ones you expected.

**Environment**
- cavsim2d version:
- Python version:
- OS:
- ngsolve / gmsh versions (if the failure is in a solve):
