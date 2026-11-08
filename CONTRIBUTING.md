# Contributing

Pie is in alpha, developed by Vijay Sriram at aggnr. During the alpha it
takes no pull requests: its packages still change with Crow between
releases, and every line is kept under one hand so that the licence can
still change. A pull request is closed with this note by a workflow, so
nobody waits on one.

What is welcome, in Issues:

- **A bug**, with the program that shows it and what each mode said —
  `crow run -i file.co` and `crow run file.co`. Pie's answers are held to
  be the same in both, and to the library each package follows (PyTorch
  in float64, scipy, scikit-learn, networkx, faiss), so a difference from
  either is a bug by itself.
- **A question** the READMEs, the book (`docs/book`) or
  `docs/pie-design.md` do not answer.
- **What you built with it.**

Working from a checkout: `./test.sh [crow]` runs every package's tests
and examples, the parity programs and the book's examples in both modes,
and `docs/pie-design.md` says where the line with Crow lies.
