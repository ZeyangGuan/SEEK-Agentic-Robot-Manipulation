# Frozen evaluation after five evolution attempts

Complete · updated 2026-11-25 04:45:04 UTC · **340 / 340** episodes scored

SEEK evaluated with the execution knowledge of five evolution attempts per task, on layouts it has not seen. This page is regenerated from the run every six hours until all episodes are scored; the [project page](https://zeyangguan.github.io/SEEK-Agentic-Robot-Manipulation/#log) shows the same data.

| Setting | Value |
|---|---|
| Tasks | 34 RoboDojo tasks: Generalization, Precision, Long-Horizon, Memory |
| Fixed model | `gpt-6.2-astra`, reasoning effort `xhigh` |
| Knowledge | $\mathcal K_5$, five evolution attempts per task on seed 0 standard layouts |
| Evaluation layouts | seed 2, 10 episodes per task. Generalization tasks: standard layouts 1-5 and randomized layouts 1-5; other tasks: standard layouts 1-10 |
| Protocol | Every episode starts from the same snapshot, may revise procedures within the episode, and writes nothing back |

## Summary

| | Tasks | Episodes | Score | Success (%) | Model requests / episode |
|---|---:|---:|---:|---:|---:|
| **Average** | 34 | 340 / 340 | 80.1 | 75.0 | 14.1 |
| Generalization | 12 | 120 / 120 | 74.8 | 69.2 | 21.9 |
| Precision | 8 | 80 / 80 | 79.2 | 75.0 | 11.1 |
| Long-Horizon | 8 | 80 / 80 | 72.8 | 62.5 | 18.1 |
| Memory | 6 | 60 / 60 | 93.6 | 93.3 | 5.3 |

Score is the benchmark task score (0–100). A task is averaged over its scored episodes, a category over its tasks, and the average over the four categories. Success counts episodes with a full score. Model requests are those of the run that produced each score, as for the direct run; requests of attempts interrupted by infrastructure failures are not counted.

## Tasks

### Generalization

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores |
|---|---:|---:|---:|---:|---|
| Stack Bowls | 10 / 10 | 100.0 | 100.0 | 4.5 | 100 100 100 100 100 100 100 100 100 100 |
| Push T | 10 / 10 | 100.0 | 100.0 | 1.8 | 100 100 100 100 100 100 100 100 100 100 |
| Pack Objects into Box | 10 / 10 | 75.0 | 50.0 | 34.2 | 50 50 100 100 100 50 50 100 100 50 |
| Fold Clothes | 10 / 10 | 90.0 | 90.0 | 2.6 | 100 100 100 100 100 100 100 100 0 100 |
| Hang Mugs | 10 / 10 | 49.0 | 30.0 | 47.7 | 100 100 40 100 40 15 0 15 40 40 |
| Sweep Blocks | 10 / 10 | 60.0 | 60.0 | 41.4 | 100 100 0 0 100 100 0 100 100 0 |
| Pour Liquid into Cup | 10 / 10 | 60.0 | 60.0 | 10.7 | 0 0 100 100 0 100 100 100 100 0 |
| Make Toast | 10 / 10 | 60.0 | 60.0 | 38.1 | 100 100 100 100 0 100 100 0 0 0 |
| Arrange Largest Number | 10 / 10 | 87.0 | 80.0 | 8.6 | 30 100 100 100 100 100 100 100 40 100 |
| Sort Nesting Dolls by Size | 10 / 10 | 100.0 | 100.0 | 12.6 | 100 100 100 100 100 100 100 100 100 100 |
| Store Laptop and Headphones | 10 / 10 | 16.0 | 0.0 | 51.5 | 20 20 20 20 20 0 20 20 0 20 |
| Stack Blocks | 10 / 10 | 100.0 | 100.0 | 9.5 | 100 100 100 100 100 100 100 100 100 100 |

### Precision

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores |
|---|---:|---:|---:|---:|---|
| Fasten Screws | 10 / 10 | 95.0 | 90.0 | 4.2 | 100 100 100 100 100 100 100 100 50 100 |
| Plug in Charger | 10 / 10 | 90.0 | 90.0 | 14.3 | 100 100 100 0 100 100 100 100 100 100 |
| Insert Tubes | 10 / 10 | 80.0 | 70.0 | 12.6 | 20 100 100 40 100 100 40 100 100 100 |
| Pour Balls into Vase | 10 / 10 | 90.0 | 90.0 | 15.4 | 100 100 100 100 0 100 100 100 100 100 |
| Play Xylophone | 10 / 10 | 80.0 | 80.0 | 8.9 | 0 100 100 0 100 100 100 100 100 100 |
| Deposit Coin | 10 / 10 | 84.0 | 80.0 | 3.6 | 100 100 100 100 100 100 100 20 100 20 |
| Insert Key | 10 / 10 | 15.0 | 0.0 | 21.8 | 15 15 15 15 15 15 15 15 15 15 |
| Build Tower | 10 / 10 | 100.0 | 100.0 | 8.3 | 100 100 100 100 100 100 100 100 100 100 |

### Long-Horizon

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores |
|---|---:|---:|---:|---:|---|
| Fill Pen Holder | 10 / 10 | 47.5 | 20.0 | 37.9 | 25 100 25 40 25 40 40 40 40 100 |
| Classify Objects | 10 / 10 | 100.0 | 100.0 | 12.5 | 100 100 100 100 100 100 100 100 100 100 |
| Put Bottles into Dustbin | 10 / 10 | 92.5 | 90.0 | 20.3 | 100 100 100 100 100 100 25 100 100 100 |
| Play Tic-Tac-Toe | 10 / 10 | 100.0 | 100.0 | 0.0 | 100 100 100 100 100 100 100 100 100 100 |
| Fill Egg Holder | 10 / 10 | 57.5 | 40.0 | 12.5 | 90 25 10 100 10 40 0 100 100 100 |
| Organize Table | 10 / 10 | 92.5 | 70.0 | 27.1 | 100 100 100 100 75 100 75 100 75 100 |
| Make Kong | 10 / 10 | 0.0 | 0.0 | 13.1 | 0 0 0 0 0 0 0 0 0 0 |
| Play Stacking Toy | 10 / 10 | 92.0 | 80.0 | 21.6 | 100 100 100 100 60 100 100 100 60 100 |

### Memory

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores |
|---|---:|---:|---:|---:|---|
| Cover Blocks | 10 / 10 | 100.0 | 100.0 | 0.1 | 100 100 100 100 100 100 100 100 100 100 |
| Match and Pick from Conveyor | 10 / 10 | 90.0 | 90.0 | 9.9 | 100 100 100 100 100 0 100 100 100 100 |
| Swap Blocks | 10 / 10 | 100.0 | 100.0 | 3.0 | 100 100 100 100 100 100 100 100 100 100 |
| Swap T | 10 / 10 | 100.0 | 100.0 | 0.7 | 100 100 100 100 100 100 100 100 100 100 |
| Press by Number | 10 / 10 | 90.0 | 90.0 | 3.6 | 100 100 100 100 100 100 0 100 100 100 |
| Imitate Sorting Sequence | 10 / 10 | 81.5 | 80.0 | 14.5 | 100 100 100 0 100 15 100 100 100 100 |

Episode scores run from episode 1 to 10; ▶ running, · queued, ! held for inspection.

## Episodes

Seed, layout, score and model requests of every episode. The same data is in [`episodes.csv`](episodes.csv) and [`results.json`](results.json).

<details>
<summary>Stack Bowls · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 8 | 554 |
| 2 | 2 | standard | 2 | done | 100.0 | 3 | 438 |
| 3 | 2 | standard | 3 | done | 100.0 | 2 | 515 |
| 4 | 2 | standard | 4 | done | 100.0 | 2 | 469 |
| 5 | 2 | standard | 5 | done | 100.0 | 3 | 470 |
| 6 | 2 | random | 1 | done | 100.0 | 5 | 373 |
| 7 | 2 | random | 2 | done | 100.0 | 6 | 460 |
| 8 | 2 | random | 3 | done | 100.0 | 3 | 493 |
| 9 | 2 | random | 4 | done | 100.0 | 8 | 574 |
| 10 | 2 | random | 5 | done | 100.0 | 5 | 390 |

</details>

<details>
<summary>Push T · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 0 | 148 |
| 2 | 2 | standard | 2 | done | 100.0 | 2 | 195 |
| 3 | 2 | standard | 3 | done | 100.0 | 2 | 134 |
| 4 | 2 | standard | 4 | done | 100.0 | 2 | 265 |
| 5 | 2 | standard | 5 | done | 100.0 | 1 | 316 |
| 6 | 2 | random | 1 | done | 100.0 | 2 | 150 |
| 7 | 2 | random | 2 | done | 100.0 | 2 | 310 |
| 8 | 2 | random | 3 | done | 100.0 | 3 | 215 |
| 9 | 2 | random | 4 | done | 100.0 | 2 | 256 |
| 10 | 2 | random | 5 | done | 100.0 | 2 | 236 |

</details>

<details>
<summary>Pack Objects into Box · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 50.0 | 42 | 1300 |
| 2 | 2 | standard | 2 | done | 50.0 | 44 | 1300 |
| 3 | 2 | standard | 3 | done | 100.0 | 25 | 1091 |
| 4 | 2 | standard | 4 | done | 100.0 | 23 | 787 |
| 5 | 2 | standard | 5 | done | 100.0 | 16 | 692 |
| 6 | 2 | random | 1 | done | 50.0 | 40 | 1300 |
| 7 | 2 | random | 2 | done | 50.0 | 50 | 1300 |
| 8 | 2 | random | 3 | done | 100.0 | 21 | 1256 |
| 9 | 2 | random | 4 | done | 100.0 | 32 | 1232 |
| 10 | 2 | random | 5 | done | 50.0 | 49 | 1287 |

</details>

<details>
<summary>Fold Clothes · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 3 | 407 |
| 2 | 2 | standard | 2 | done | 100.0 | 2 | 392 |
| 3 | 2 | standard | 3 | done | 100.0 | 2 | 307 |
| 4 | 2 | standard | 4 | done | 100.0 | 0 | 383 |
| 5 | 2 | standard | 5 | done | 100.0 | 4 | 367 |
| 6 | 2 | random | 1 | done | 100.0 | 0 | 274 |
| 7 | 2 | random | 2 | done | 100.0 | 2 | 384 |
| 8 | 2 | random | 3 | done | 100.0 | 2 | 292 |
| 9 | 2 | random | 4 | done | 0.0 | 3 | 342 |
| 10 | 2 | random | 5 | done | 100.0 | 8 | 342 |

</details>

<details>
<summary>Hang Mugs · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 43 | 689 |
| 2 | 2 | standard | 2 | done | 100.0 | 30 | 380 |
| 3 | 2 | standard | 3 | done | 40.0 | 52 | 800 |
| 4 | 2 | standard | 4 | done | 100.0 | 27 | 391 |
| 5 | 2 | standard | 5 | done | 40.0 | 37 | 538 |
| 6 | 2 | random | 1 | done | 15.0 | 59 | 800 |
| 7 | 2 | random | 2 | done | 0.0 | 64 | 800 |
| 8 | 2 | random | 3 | done | 15.0 | 62 | 800 |
| 9 | 2 | random | 4 | done | 40.0 | 48 | 800 |
| 10 | 2 | random | 5 | done | 40.0 | 55 | 800 |

</details>

<details>
<summary>Sweep Blocks · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 13 | 581 |
| 2 | 2 | standard | 2 | done | 100.0 | 15 | 493 |
| 3 | 2 | standard | 3 | done | 0.0 | 69 | 1000 |
| 4 | 2 | standard | 4 | done | 0.0 | 47 | 1000 |
| 5 | 2 | standard | 5 | done | 100.0 | 23 | 960 |
| 6 | 2 | random | 1 | done | 100.0 | 25 | 685 |
| 7 | 2 | random | 2 | done | 0.0 | 71 | 1000 |
| 8 | 2 | random | 3 | done | 100.0 | 42 | 943 |
| 9 | 2 | random | 4 | done | 100.0 | 40 | 973 |
| 10 | 2 | random | 5 | done | 0.0 | 69 | 1000 |

</details>

<details>
<summary>Pour Liquid into Cup · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 0.0 | 15 | 301 |
| 2 | 2 | standard | 2 | done | 0.0 | 13 | 249 |
| 3 | 2 | standard | 3 | done | 100.0 | 6 | 182 |
| 4 | 2 | standard | 4 | done | 100.0 | 8 | 244 |
| 5 | 2 | standard | 5 | done | 0.0 | 12 | 230 |
| 6 | 2 | random | 1 | done | 100.0 | 6 | 202 |
| 7 | 2 | random | 2 | done | 100.0 | 6 | 166 |
| 8 | 2 | random | 3 | done | 100.0 | 4 | 174 |
| 9 | 2 | random | 4 | done | 100.0 | 13 | 216 |
| 10 | 2 | random | 5 | done | 0.0 | 24 | 400 |

</details>

<details>
<summary>Make Toast · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 34 | 1021 |
| 2 | 2 | standard | 2 | done | 100.0 | 16 | 787 |
| 3 | 2 | standard | 3 | done | 100.0 | 23 | 1051 |
| 4 | 2 | standard | 4 | done | 100.0 | 35 | 938 |
| 5 | 2 | standard | 5 | done | 0.0 | 62 | 1400 |
| 6 | 2 | random | 1 | done | 100.0 | 26 | 682 |
| 7 | 2 | random | 2 | done | 100.0 | 16 | 768 |
| 8 | 2 | random | 3 | done | 0.0 | 49 | 1395 |
| 9 | 2 | random | 4 | done | 0.0 | 66 | 1400 |
| 10 | 2 | random | 5 | done | 0.0 | 54 | 1393 |

</details>

<details>
<summary>Arrange Largest Number · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 30.0 | 36 | 1050 |
| 2 | 2 | standard | 2 | done | 100.0 | 4 | 932 |
| 3 | 2 | standard | 3 | done | 100.0 | 2 | 554 |
| 4 | 2 | standard | 4 | done | 100.0 | 2 | 639 |
| 5 | 2 | standard | 5 | done | 100.0 | 3 | 311 |
| 6 | 2 | random | 1 | done | 100.0 | 5 | 735 |
| 7 | 2 | random | 2 | done | 100.0 | 9 | 724 |
| 8 | 2 | random | 3 | done | 100.0 | 6 | 929 |
| 9 | 2 | random | 4 | done | 40.0 | 14 | 1050 |
| 10 | 2 | random | 5 | done | 100.0 | 5 | 477 |

</details>

<details>
<summary>Sort Nesting Dolls by Size · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 10 | 558 |
| 2 | 2 | standard | 2 | done | 100.0 | 11 | 690 |
| 3 | 2 | standard | 3 | done | 100.0 | 27 | 1028 |
| 4 | 2 | standard | 4 | done | 100.0 | 9 | 543 |
| 5 | 2 | standard | 5 | done | 100.0 | 8 | 508 |
| 6 | 2 | random | 1 | done | 100.0 | 17 | 874 |
| 7 | 2 | random | 2 | done | 100.0 | 11 | 644 |
| 8 | 2 | random | 3 | done | 100.0 | 8 | 470 |
| 9 | 2 | random | 4 | done | 100.0 | 7 | 581 |
| 10 | 2 | random | 5 | done | 100.0 | 18 | 857 |

</details>

<details>
<summary>Store Laptop and Headphones · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 20.0 | 42 | 750 |
| 2 | 2 | standard | 2 | done | 20.0 | 63 | 800 |
| 3 | 2 | standard | 3 | done | 20.0 | 48 | 800 |
| 4 | 2 | standard | 4 | done | 20.0 | 57 | 800 |
| 5 | 2 | standard | 5 | done | 20.0 | 55 | 800 |
| 6 | 2 | random | 1 | done | 0.0 | 56 | 800 |
| 7 | 2 | random | 2 | done | 20.0 | 39 | 800 |
| 8 | 2 | random | 3 | done | 20.0 | 35 | 622 |
| 9 | 2 | random | 4 | done | 0.0 | 70 | 800 |
| 10 | 2 | random | 5 | done | 20.0 | 50 | 800 |

</details>

<details>
<summary>Stack Blocks · Generalization · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 8 | 140 |
| 2 | 2 | standard | 2 | done | 100.0 | 7 | 235 |
| 3 | 2 | standard | 3 | done | 100.0 | 8 | 260 |
| 4 | 2 | standard | 4 | done | 100.0 | 12 | 300 |
| 5 | 2 | standard | 5 | done | 100.0 | 13 | 259 |
| 6 | 2 | random | 1 | done | 100.0 | 6 | 268 |
| 7 | 2 | random | 2 | done | 100.0 | 10 | 232 |
| 8 | 2 | random | 3 | done | 100.0 | 10 | 253 |
| 9 | 2 | random | 4 | done | 100.0 | 12 | 250 |
| 10 | 2 | random | 5 | done | 100.0 | 9 | 230 |

</details>

<details>
<summary>Fasten Screws · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 3 | 1874 |
| 2 | 2 | standard | 2 | done | 100.0 | 4 | 1888 |
| 3 | 2 | standard | 3 | done | 100.0 | 4 | 1884 |
| 4 | 2 | standard | 4 | done | 100.0 | 3 | 1875 |
| 5 | 2 | standard | 5 | done | 100.0 | 3 | 1875 |
| 6 | 2 | standard | 6 | done | 100.0 | 4 | 1875 |
| 7 | 2 | standard | 7 | done | 100.0 | 5 | 1861 |
| 8 | 2 | standard | 8 | done | 100.0 | 3 | 1860 |
| 9 | 2 | standard | 9 | done | 50.0 | 10 | 1892 |
| 10 | 2 | standard | 10 | done | 100.0 | 3 | 1893 |

</details>

<details>
<summary>Plug in Charger · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 10 | 187 |
| 2 | 2 | standard | 2 | done | 100.0 | 14 | 192 |
| 3 | 2 | standard | 3 | done | 100.0 | 25 | 364 |
| 4 | 2 | standard | 4 | done | 0.0 | 5 | 400 |
| 5 | 2 | standard | 5 | done | 100.0 | 25 | 352 |
| 6 | 2 | standard | 6 | done | 100.0 | 11 | 213 |
| 7 | 2 | standard | 7 | done | 100.0 | 9 | 181 |
| 8 | 2 | standard | 8 | done | 100.0 | 21 | 279 |
| 9 | 2 | standard | 9 | done | 100.0 | 10 | 187 |
| 10 | 2 | standard | 10 | done | 100.0 | 13 | 220 |

</details>

<details>
<summary>Insert Tubes · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 20.0 | 22 | 500 |
| 2 | 2 | standard | 2 | done | 100.0 | 3 | 389 |
| 3 | 2 | standard | 3 | done | 100.0 | 31 | 491 |
| 4 | 2 | standard | 4 | done | 40.0 | 20 | 500 |
| 5 | 2 | standard | 5 | done | 100.0 | 7 | 473 |
| 6 | 2 | standard | 6 | done | 100.0 | 15 | 483 |
| 7 | 2 | standard | 7 | done | 40.0 | 8 | 500 |
| 8 | 2 | standard | 8 | done | 100.0 | 18 | 485 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 394 |
| 10 | 2 | standard | 10 | done | 100.0 | 2 | 388 |

</details>

<details>
<summary>Pour Balls into Vase · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 10 | 247 |
| 2 | 2 | standard | 2 | done | 100.0 | 4 | 88 |
| 3 | 2 | standard | 3 | done | 100.0 | 37 | 549 |
| 4 | 2 | standard | 4 | done | 100.0 | 3 | 172 |
| 5 | 2 | standard | 5 | done | 0.0 | 31 | 600 |
| 6 | 2 | standard | 6 | done | 100.0 | 21 | 515 |
| 7 | 2 | standard | 7 | done | 100.0 | 6 | 107 |
| 8 | 2 | standard | 8 | done | 100.0 | 13 | 385 |
| 9 | 2 | standard | 9 | done | 100.0 | 14 | 330 |
| 10 | 2 | standard | 10 | done | 100.0 | 15 | 347 |

</details>

<details>
<summary>Play Xylophone · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 0.0 | 21 | 500 |
| 2 | 2 | standard | 2 | done | 100.0 | 2 | 137 |
| 3 | 2 | standard | 3 | done | 100.0 | 6 | 183 |
| 4 | 2 | standard | 4 | done | 0.0 | 32 | 500 |
| 5 | 2 | standard | 5 | done | 100.0 | 8 | 292 |
| 6 | 2 | standard | 6 | done | 100.0 | 15 | 281 |
| 7 | 2 | standard | 7 | done | 100.0 | 0 | 139 |
| 8 | 2 | standard | 8 | done | 100.0 | 2 | 127 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 145 |
| 10 | 2 | standard | 10 | done | 100.0 | 3 | 164 |

</details>

<details>
<summary>Deposit Coin · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 5 | 142 |
| 2 | 2 | standard | 2 | done | 100.0 | 2 | 119 |
| 3 | 2 | standard | 3 | done | 100.0 | 0 | 114 |
| 4 | 2 | standard | 4 | done | 100.0 | 2 | 139 |
| 5 | 2 | standard | 5 | done | 100.0 | 2 | 130 |
| 6 | 2 | standard | 6 | done | 100.0 | 0 | 130 |
| 7 | 2 | standard | 7 | done | 100.0 | 2 | 201 |
| 8 | 2 | standard | 8 | done | 20.0 | 11 | 300 |
| 9 | 2 | standard | 9 | done | 100.0 | 4 | 120 |
| 10 | 2 | standard | 10 | done | 20.0 | 8 | 300 |

</details>

<details>
<summary>Insert Key · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 15.0 | 21 | 300 |
| 2 | 2 | standard | 2 | done | 15.0 | 19 | 300 |
| 3 | 2 | standard | 3 | done | 15.0 | 39 | 300 |
| 4 | 2 | standard | 4 | done | 15.0 | 27 | 300 |
| 5 | 2 | standard | 5 | done | 15.0 | 23 | 300 |
| 6 | 2 | standard | 6 | done | 15.0 | 9 | 300 |
| 7 | 2 | standard | 7 | done | 15.0 | 16 | 300 |
| 8 | 2 | standard | 8 | done | 15.0 | 18 | 300 |
| 9 | 2 | standard | 9 | done | 15.0 | 19 | 300 |
| 10 | 2 | standard | 10 | done | 15.0 | 27 | 300 |

</details>

<details>
<summary>Build Tower · Precision · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 8 | 499 |
| 2 | 2 | standard | 2 | done | 100.0 | 4 | 572 |
| 3 | 2 | standard | 3 | done | 100.0 | 14 | 680 |
| 4 | 2 | standard | 4 | done | 100.0 | 5 | 428 |
| 5 | 2 | standard | 5 | done | 100.0 | 6 | 506 |
| 6 | 2 | standard | 6 | done | 100.0 | 8 | 432 |
| 7 | 2 | standard | 7 | done | 100.0 | 17 | 834 |
| 8 | 2 | standard | 8 | done | 100.0 | 6 | 460 |
| 9 | 2 | standard | 9 | done | 100.0 | 4 | 516 |
| 10 | 2 | standard | 10 | done | 100.0 | 11 | 620 |

</details>

<details>
<summary>Fill Pen Holder · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 25.0 | 29 | 859 |
| 2 | 2 | standard | 2 | done | 100.0 | 26 | 911 |
| 3 | 2 | standard | 3 | done | 25.0 | 45 | 870 |
| 4 | 2 | standard | 4 | done | 40.0 | 38 | 1071 |
| 5 | 2 | standard | 5 | done | 25.0 | 29 | 954 |
| 6 | 2 | standard | 6 | done | 40.0 | 49 | 1100 |
| 7 | 2 | standard | 7 | done | 40.0 | 40 | 1100 |
| 8 | 2 | standard | 8 | done | 40.0 | 38 | 1100 |
| 9 | 2 | standard | 9 | done | 40.0 | 27 | 850 |
| 10 | 2 | standard | 10 | done | 100.0 | 58 | 1010 |

</details>

<details>
<summary>Classify Objects · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 10 | 543 |
| 2 | 2 | standard | 2 | done | 100.0 | 11 | 594 |
| 3 | 2 | standard | 3 | done | 100.0 | 12 | 698 |
| 4 | 2 | standard | 4 | done | 100.0 | 11 | 517 |
| 5 | 2 | standard | 5 | done | 100.0 | 12 | 498 |
| 6 | 2 | standard | 6 | done | 100.0 | 14 | 664 |
| 7 | 2 | standard | 7 | done | 100.0 | 10 | 408 |
| 8 | 2 | standard | 8 | done | 100.0 | 10 | 510 |
| 9 | 2 | standard | 9 | done | 100.0 | 18 | 895 |
| 10 | 2 | standard | 10 | done | 100.0 | 17 | 619 |

</details>

<details>
<summary>Put Bottles into Dustbin · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 22 | 422 |
| 2 | 2 | standard | 2 | done | 100.0 | 15 | 316 |
| 3 | 2 | standard | 3 | done | 100.0 | 24 | 419 |
| 4 | 2 | standard | 4 | done | 100.0 | 21 | 378 |
| 5 | 2 | standard | 5 | done | 100.0 | 17 | 417 |
| 6 | 2 | standard | 6 | done | 100.0 | 13 | 263 |
| 7 | 2 | standard | 7 | done | 25.0 | 17 | 700 |
| 8 | 2 | standard | 8 | done | 100.0 | 28 | 485 |
| 9 | 2 | standard | 9 | done | 100.0 | 23 | 379 |
| 10 | 2 | standard | 10 | done | 100.0 | 23 | 488 |

</details>

<details>
<summary>Play Tic-Tac-Toe · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 0 | 747 |
| 2 | 2 | standard | 2 | done | 100.0 | 0 | 773 |
| 3 | 2 | standard | 3 | done | 100.0 | 0 | 772 |
| 4 | 2 | standard | 4 | done | 100.0 | 0 | 772 |
| 5 | 2 | standard | 5 | done | 100.0 | 0 | 751 |
| 6 | 2 | standard | 6 | done | 100.0 | 0 | 772 |
| 7 | 2 | standard | 7 | done | 100.0 | 0 | 747 |
| 8 | 2 | standard | 8 | done | 100.0 | 0 | 752 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 755 |
| 10 | 2 | standard | 10 | done | 100.0 | 0 | 763 |

</details>

<details>
<summary>Fill Egg Holder · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 90.0 | 17 | 700 |
| 2 | 2 | standard | 2 | done | 25.0 | 3 | 700 |
| 3 | 2 | standard | 3 | done | 10.0 | 8 | 667 |
| 4 | 2 | standard | 4 | done | 100.0 | 9 | 622 |
| 5 | 2 | standard | 5 | done | 10.0 | 17 | 700 |
| 6 | 2 | standard | 6 | done | 40.0 | 27 | 678 |
| 7 | 2 | standard | 7 | done | 0.0 | 16 | 700 |
| 8 | 2 | standard | 8 | done | 100.0 | 10 | 588 |
| 9 | 2 | standard | 9 | done | 100.0 | 13 | 641 |
| 10 | 2 | standard | 10 | done | 100.0 | 5 | 580 |

</details>

<details>
<summary>Organize Table · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 21 | 730 |
| 2 | 2 | standard | 2 | done | 100.0 | 31 | 831 |
| 3 | 2 | standard | 3 | done | 100.0 | 29 | 972 |
| 4 | 2 | standard | 4 | done | 100.0 | 42 | 954 |
| 5 | 2 | standard | 5 | done | 75.0 | 11 | 1000 |
| 6 | 2 | standard | 6 | done | 100.0 | 17 | 820 |
| 7 | 2 | standard | 7 | done | 75.0 | 39 | 1000 |
| 8 | 2 | standard | 8 | done | 100.0 | 21 | 784 |
| 9 | 2 | standard | 9 | done | 75.0 | 38 | 928 |
| 10 | 2 | standard | 10 | done | 100.0 | 22 | 862 |

</details>

<details>
<summary>Make Kong · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 0.0 | 15 | 530 |
| 2 | 2 | standard | 2 | done | 0.0 | 7 | 600 |
| 3 | 2 | standard | 3 | done | 0.0 | 19 | 525 |
| 4 | 2 | standard | 4 | done | 0.0 | 12 | 504 |
| 5 | 2 | standard | 5 | done | 0.0 | 11 | 501 |
| 6 | 2 | standard | 6 | done | 0.0 | 6 | 600 |
| 7 | 2 | standard | 7 | done | 0.0 | 16 | 340 |
| 8 | 2 | standard | 8 | done | 0.0 | 5 | 600 |
| 9 | 2 | standard | 9 | done | 0.0 | 18 | 579 |
| 10 | 2 | standard | 10 | done | 0.0 | 22 | 436 |

</details>

<details>
<summary>Play Stacking Toy · Long-Horizon · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 21 | 940 |
| 2 | 2 | standard | 2 | done | 100.0 | 16 | 1081 |
| 3 | 2 | standard | 3 | done | 100.0 | 23 | 1119 |
| 4 | 2 | standard | 4 | done | 100.0 | 36 | 1183 |
| 5 | 2 | standard | 5 | done | 60.0 | 28 | 1186 |
| 6 | 2 | standard | 6 | done | 100.0 | 22 | 1154 |
| 7 | 2 | standard | 7 | done | 100.0 | 13 | 1021 |
| 8 | 2 | standard | 8 | done | 100.0 | 11 | 1033 |
| 9 | 2 | standard | 9 | done | 60.0 | 25 | 1183 |
| 10 | 2 | standard | 10 | done | 100.0 | 21 | 1076 |

</details>

<details>
<summary>Cover Blocks · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 0 | 374 |
| 2 | 2 | standard | 2 | done | 100.0 | 0 | 377 |
| 3 | 2 | standard | 3 | done | 100.0 | 0 | 374 |
| 4 | 2 | standard | 4 | done | 100.0 | 0 | 374 |
| 5 | 2 | standard | 5 | done | 100.0 | 0 | 375 |
| 6 | 2 | standard | 6 | done | 100.0 | 1 | 388 |
| 7 | 2 | standard | 7 | done | 100.0 | 0 | 374 |
| 8 | 2 | standard | 8 | done | 100.0 | 0 | 373 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 346 |
| 10 | 2 | standard | 10 | done | 100.0 | 0 | 346 |

</details>

<details>
<summary>Match and Pick from Conveyor · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 5 | 402 |
| 2 | 2 | standard | 2 | done | 100.0 | 6 | 492 |
| 3 | 2 | standard | 3 | done | 100.0 | 6 | 449 |
| 4 | 2 | standard | 4 | done | 100.0 | 5 | 397 |
| 5 | 2 | standard | 5 | done | 100.0 | 9 | 485 |
| 6 | 2 | standard | 6 | done | 0.0 | 35 | 533 |
| 7 | 2 | standard | 7 | done | 100.0 | 10 | 448 |
| 8 | 2 | standard | 8 | done | 100.0 | 5 | 482 |
| 9 | 2 | standard | 9 | done | 100.0 | 6 | 481 |
| 10 | 2 | standard | 10 | done | 100.0 | 12 | 484 |

</details>

<details>
<summary>Swap Blocks · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 3 | 669 |
| 2 | 2 | standard | 2 | done | 100.0 | 6 | 512 |
| 3 | 2 | standard | 3 | done | 100.0 | 1 | 519 |
| 4 | 2 | standard | 4 | done | 100.0 | 0 | 507 |
| 5 | 2 | standard | 5 | done | 100.0 | 2 | 576 |
| 6 | 2 | standard | 6 | done | 100.0 | 7 | 569 |
| 7 | 2 | standard | 7 | done | 100.0 | 3 | 579 |
| 8 | 2 | standard | 8 | done | 100.0 | 3 | 509 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 570 |
| 10 | 2 | standard | 10 | done | 100.0 | 5 | 510 |

</details>

<details>
<summary>Swap T · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 2 | 185 |
| 2 | 2 | standard | 2 | done | 100.0 | 0 | 148 |
| 3 | 2 | standard | 3 | done | 100.0 | 0 | 177 |
| 4 | 2 | standard | 4 | done | 100.0 | 2 | 152 |
| 5 | 2 | standard | 5 | done | 100.0 | 0 | 141 |
| 6 | 2 | standard | 6 | done | 100.0 | 0 | 208 |
| 7 | 2 | standard | 7 | done | 100.0 | 0 | 166 |
| 8 | 2 | standard | 8 | done | 100.0 | 3 | 239 |
| 9 | 2 | standard | 9 | done | 100.0 | 0 | 149 |
| 10 | 2 | standard | 10 | done | 100.0 | 0 | 192 |

</details>

<details>
<summary>Press by Number · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 3 | 116 |
| 2 | 2 | standard | 2 | done | 100.0 | 7 | 176 |
| 3 | 2 | standard | 3 | done | 100.0 | 6 | 206 |
| 4 | 2 | standard | 4 | done | 100.0 | 3 | 177 |
| 5 | 2 | standard | 5 | done | 100.0 | 0 | 259 |
| 6 | 2 | standard | 6 | done | 100.0 | 3 | 172 |
| 7 | 2 | standard | 7 | done | 0.0 | 5 | 148 |
| 8 | 2 | standard | 8 | done | 100.0 | 0 | 191 |
| 9 | 2 | standard | 9 | done | 100.0 | 9 | 229 |
| 10 | 2 | standard | 10 | done | 100.0 | 0 | 162 |

</details>

<details>
<summary>Imitate Sorting Sequence · Memory · 10 / 10 scored</summary>

| Episode | Seed | Variant | Layout | Status | Score | Model requests | Steps |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 2 | standard | 1 | done | 100.0 | 14 | 1436 |
| 2 | 2 | standard | 2 | done | 100.0 | 18 | 1257 |
| 3 | 2 | standard | 3 | done | 100.0 | 12 | 1294 |
| 4 | 2 | standard | 4 | done | 0.0 | 7 | 869 |
| 5 | 2 | standard | 5 | done | 100.0 | 12 | 1273 |
| 6 | 2 | standard | 6 | done | 15.0 | 17 | 1600 |
| 7 | 2 | standard | 7 | done | 100.0 | 16 | 1279 |
| 8 | 2 | standard | 8 | done | 100.0 | 15 | 1328 |
| 9 | 2 | standard | 9 | done | 100.0 | 17 | 1489 |
| 10 | 2 | standard | 10 | done | 100.0 | 17 | 1441 |

</details>
