# Classroom data and provenance

All data in the new core notebooks are authored or generated for this course. They are small mechanism demonstrations, not natural-language, speech, or vision benchmarks.

- Tokenization strings: short authored examples; language coverage is illustrative, not linguistically validated evaluation data.
- Campus rules: fictional documents, with no claim about any AIMS campus policy.
- Sentiment TSV fixtures: synthetic English sentences labelled positive/negative; they are not AfriSenti records. Use them to smoke-test the file format and pipeline only.
- Charts, geometric patterns, and videos: generated arrays with known labels. Fixed rendering assumptions deliberately expose shortcut learning and limited generalization.
- Waveform: a synthetic tone and noise, not a human recording.
- Classification probabilities, preference pairs, and judge choices: constructed numerical cases, not outputs measured from named models.

All random core data use NumPy's generator with seed 42. Each notebook records the generating code inline. The SVG illustrations in this directory visualize authored examples and are not external media. Any project adding real data must supply its own provenance/access statement and preserve source-disjoint splits.
