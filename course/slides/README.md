# Lecture slides

[Course home](../../README.md)

One deck per session, built on the Imperial College London template in [template/](template/). Open the `.pptx` files in PowerPoint to present or edit by hand.

To change a deck at its source, edit `src/NN_slug.md` and rebuild:

```bash
python -m pip install -r requirements-slides.txt
python scripts/build_slides.py --only 11_steering   # or no argument for all decks
```

The source format is described at the top of [scripts/build_slides.py](../../scripts/build_slides.py). Figures are drawn by `src/NN_slug_figures.py` with the shared style in `src/figstyle.py`, and saved to `figures/`. The build stops if text does not fit its box at 16 pt; split the slide. A hand edit made in PowerPoint is lost when the deck is rebuilt, so choose one way of working per deck.
