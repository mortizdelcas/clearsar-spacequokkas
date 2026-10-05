# The rescorer, in plain language

After several detectors have already proposed boxes and those boxes have been merged, we still have a problem: **the confidence number on a box is not always honest**.

A box can look very sure and still be wrong. Another box can look doubtful and still be real RFI. The rescorer is a small second opinion. It does **not** draw a new box. It only asks: *how likely is this existing box to be a real interference mark?* Then it may change the score.

---

## The job in one sentence

Look at each fused box, guess whether it is a true detection, and rewrite shaky scores so the ranking is more trustworthy.

---

## What it sees

For every merged box the rescorer gets two things:

1. **A small picture** — a 96 by 96 crop of the quicklook, centred on the box, with a bit of the surroundings. This answers “does this *look* like RFI?”
2. **A short fact sheet** about the box:
   - how confident the merge already was
   - how big and how skinny the box is, and where it sits in the image
   - whether other boxes sit on top of it
   - **who voted for it** — Cascade, Co-DETR, D-FINE, D-FINE with a flip

That last part matters. A box that four models agreed on is a different story from a box that only one model invented.

---

## How it thinks

Think of two specialists talking, then a judge.

**The vision specialist** is a tiny convolutional network. It looks at the crop in four shrinking steps (the image is halved each time), paying more attention to the channels that matter. It then summarises the whole crop in two ways: the average look, and the strongest look. Together that is “what does this patch look like?”

**The fact-sheet specialist** is a small multilayer network. It reads the 22 numbers (score, shape, neighbours, who voted) and turns them into a compact summary.

**The judge** concatenates both summaries and outputs one number: the probability that this box is a true positive — meaning it really overlaps a labelled RFI region.

We train three copies of this network with different random seeds and **average their guesses**, so one lucky or unlucky training run cannot dominate.

---

## What it changes (and what it does not)

It only touches **scores**, never corners.

- If the merged score is already high (0.87 or above), we leave it alone. Those boxes are usually well calibrated.
- If the merged score is still in the mid range, we replace it with a mix: **10% the old score, 90% the rescorer’s “this is real” probability**.

So a mid-confidence fake tends to get pushed down, and a mid-confidence real tends to get pushed up. The box stays where it was. The next step (integer-grid snap) is the one that may move corners.

---

## Why it is so small

This is not another detector. The detectors already did the hard search. The rescorer only has to judge a shortlist of boxes, one crop at a time. A network that small can run over every fused box in seconds.

On our validation set it added about **+0.003 mAP**. That is a small, honest gain: better ranking in the uncertain band, not a new way of finding RFI. More rescoring after that tends to saturate. The bigger remaining problem is drawing the boundary tightly, not deciding whether a box exists.

---

## Same idea, earlier in the pipeline

The Co-DETR filter is the same family of network, used differently. There it **deletes** Co-DETR boxes that look fake *before* the merge. The rescorer comes *after* the merge and only rewrites scores. One family, two jobs: throw away junk early, re-rank what survived.
