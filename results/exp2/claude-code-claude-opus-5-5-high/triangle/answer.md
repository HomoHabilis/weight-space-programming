**Observations:**

- **Output biases (b3).** The whole one-pixel frame is strongly negative: row 0 is −0.33 to −0.52, and columns 0 and 11 are about −0.35 to −0.54. Row 1 is also about −0.35 to −0.49, except at columns 5–6 (−0.143, −0.105). Rows 2–10, columns 2–9, sit near 0. So the background is black, and a white object lives inside the frame. Its tip sits at the top centre (row 1, columns 5–6).
- **W3 thin diagonal strokes.** Several output units paint thin slanted lines that start near the top centre and run down to the sides:
  - Unit 13 runs down-left: px17 (row 1, col 5) 0.129, px40 0.328, px51 0.282, px63 0.368, px74 0.450, px86 0.457, px97 (row 8, col 1) 0.351.
  - Unit 20 is its mirror, running down-right: px18 (row 1, col 6) 0.350, px31 0.299, px43 0.464, px56 0.387, px68 0.318, px81 0.419, px93/94 0.23, px106 (row 8, col 10) 0.301.
  - Units 27 and 31 have the same shape with the top shifted left or right: px28 0.517, px39 0.500 (unit 27); px44 0.556, px69 0.619 (unit 31). Unit 8 is another right edge.
  - The slope is always about 1 column per 2 rows.
- **W3 full-width horizontal lines in the lower half only.**
  - Unit 21: row 10, values 0.237, 0.403, 0.361, … 0.353.
  - Unit 41: row 9, 0.248, 0.455, 0.371, …
  - Unit 37: row 7, 0.190, 0.306, 0.405, 0.465, 0.394.
  - Unit 44: row 10, columns 2–7, 0.31–0.49.
  - Units 11 and 12 move a bottom edge: unit 12 has row 6 at +0.2–0.3 and row 8 at −0.21 to −0.33; unit 11 has row 9 at −0.36 to −0.42.
  - No horizontal lines appear in rows 1–3, so nothing has a flat top edge.
- **Solid 2-D blobs, not just lines.**
  - Unit 35 is a solid positive trapezoid that widens downward: rows 6–9, columns 2–9, about 0.1–0.34.
  - Units 19, 26 and 39 are solid negative blobs over the centre.
  - In W1, unit 2 is positive on a solid bottom-left block: px98–100, 110–112 and 121–125 at about 0.16–0.26.
- **W1 detectors.**
  - Units 4 and 5 read slanted strips on the upper left (px27 0.314, px39 0.382, px50 0.405, px74 0.398).
  - Units 14, 20 and 28 read the matching strips on the right (px32 0.396, px57 0.421, px81 0.514).
  - Unit 19 reads the tip column (px29 0.629).
  - Unit 0 reads the left edge plus row 8 (px98 0.474), and also has positive interior weights (px65 0.107, px102 0.177).
  - Unit 13 is negative on rows 8–9 (−0.30 to −0.36) but positive on rows 5–7 of the interior (about +0.1). That pattern detects "filled down to row 7, empty below".

**Reasoning:**

- Two edges slope outward from a top point and close with a horizontal bottom edge. That is an upright triangle, with the apex up and the base down.
- I ruled out downward-pointing triangles because there are no top horizontal lines.
- I ruled out squares and houses because there are no vertical edges.
- I ruled out the letter "A" because the bottom lines span the full width rather than being two leg ends.
- The shape is most likely filled rather than an outline:
  - There are solid blobs in both W1 and W3.
  - Unit 13 wants the interior white just above an empty bottom.
  - The horizontal "base" units add a row (21, 37, 41) without removing the row above. That is how a filled shape grows; an outline whose base moves would have to erase the old line.
- Variation between samples: the apex column shifts between about columns 4 and 7, the base sits on rows 6 to 10, and the width changes accordingly.

**Images:** One solid white, upward-pointing, roughly isosceles triangle on a black background, always inside the black border:
- The apex is at the top centre, around row 1–2 and columns 5–6.
- The sides fan out at about 1 column per 2 rows.
- The flat base is usually low, around rows 8–10, and spans up to columns 1–10.
- Between samples, size and horizontal position vary by a couple of pixels; a smaller triangle has its base at about row 6–7.

**Example:**
```
............
.....##.....
....####....
....####....
...######...
...######...
..########..
..########..
.##########.
.##########.
............
............
```

**Confidence:** 55%