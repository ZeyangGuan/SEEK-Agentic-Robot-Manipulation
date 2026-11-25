# gpt-6.2-astra acting directly on the same episodes

Complete · updated 2026-11-25 03:18:02 UTC · **340 / 340** episodes scored

The fixed model acting directly, with no execution knowledge, on the 340 episodes of the [SEEK evaluation log](../k5-frozen-seed2/README.md). The [project page](https://zeyangguan.github.io/SEEK-Agentic-Robot-Manipulation/#results) compares the two.

| Setting | Value |
|---|---|
| Tasks | 34 RoboDojo tasks: Generalization, Precision, Long-Horizon, Memory |
| Fixed model | `gpt-6.2-astra`, reasoning effort `xhigh` |
| Harness | RoboProbe `RoboDojo_Agent_L3_Inspect_EEF`, the benchmark team's reference harness for a model acting directly |
| Knowledge | None |
| Evaluation layouts | seed 2, 10 episodes per task. Generalization tasks: standard layouts 1-5 and randomized layouts 1-5; other tasks: standard layouts 1-10 |
| Protocol | Same episodes as the SEEK log; nothing carries over between episodes |

## Summary

| | Tasks | Episodes | Score | Success (%) | Model requests / episode | SEEK score |
|---|---:|---:|---:|---:|---:|---:|
| **Average** | 34 | 340 / 340 | 49.6 | 40.7 | 63.2 | 80.1 |
| Generalization | 12 | 120 / 120 | 49.0 | 44.2 | 65.9 | 74.8 |
| Precision | 8 | 80 / 80 | 27.5 | 17.5 | 51.6 | 79.2 |
| Long-Horizon | 8 | 80 / 80 | 47.1 | 26.2 | 86.8 | 72.8 |
| Memory | 6 | 60 / 60 | 75.0 | 75.0 | 48.3 | 93.6 |

Score is the benchmark task score (0–100). A task is averaged over its episodes, a category over its tasks, and the average over the four categories. Success counts episodes with a full score. Model requests are those of the run that produced each score.

One episode, Arrange Largest Number on randomized layout 2, is scored 0: in each of its four runs the provider refused one wrist-camera image under its content policy (HTTP 400) before the episode ended. The reference harness counts a provider refusal as a model refusal. Episodes refused once and completed on a rerun keep the score of the completed run.

## Tasks

### Generalization

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores | SEEK score |
|---|---:|---:|---:|---:|---|---:|
| Stack Bowls | 10 / 10 | 83.0 | 80.0 | 66.3 | 100 100 15 100 15 100 100 100 100 100 | 100.0 |
| Push T | 10 / 10 | 100.0 | 100.0 | 40.2 | 100 100 100 100 100 100 100 100 100 100 | 100.0 |
| Pack Objects into Box | 10 / 10 | 29.5 | 10.0 | 87.5 | 0 50 50 10 0 50 25 10 0 100 | 75.0 |
| Fold Clothes | 10 / 10 | 92.0 | 90.0 | 41.0 | 100 100 100 100 100 100 100 20 100 100 | 90.0 |
| Hang Mugs | 10 / 10 | 26.5 | 10.0 | 74.8 | 40 40 0 40 0 15 15 100 0 15 | 49.0 |
| Sweep Blocks | 10 / 10 | 50.0 | 50.0 | 80.8 | 0 0 0 100 100 100 100 100 0 0 | 60.0 |
| Pour Liquid into Cup | 10 / 10 | 10.0 | 10.0 | 24.9 | 0 0 0 0 0 0 0 0 0 100 | 60.0 |
| Make Toast | 10 / 10 | 20.0 | 20.0 | 122.2 | 0 0 0 0 0 100 100 0 0 0 | 60.0 |
| Arrange Largest Number | 10 / 10 | 75.0 | 70.0 | 70.4 | 100 25 100 100 100 100 0 100 25 100 | 87.0 |
| Sort Nesting Dolls by Size | 10 / 10 | 30.0 | 30.0 | 85.4 | 100 0 0 100 0 0 0 0 0 100 | 100.0 |
| Store Laptop and Headphones | 10 / 10 | 6.0 | 0.0 | 58.0 | 0 0 0 20 0 0 20 0 20 0 | 16.0 |
| Stack Blocks | 10 / 10 | 66.0 | 60.0 | 39.6 | 100 15 100 15 100 15 15 100 100 100 | 100.0 |

### Precision

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores | SEEK score |
|---|---:|---:|---:|---:|---|---:|
| Fasten Screws | 10 / 10 | 60.0 | 30.0 | 88.4 | 50 50 50 100 50 0 50 50 100 100 | 95.0 |
| Plug in Charger | 10 / 10 | 20.0 | 20.0 | 38.9 | 0 0 100 0 0 0 0 0 100 0 | 90.0 |
| Insert Tubes | 10 / 10 | 12.0 | 0.0 | 35.5 | 20 20 40 20 0 20 0 0 0 0 | 80.0 |
| Pour Balls into Vase | 10 / 10 | 10.0 | 10.0 | 36.6 | 100 0 0 0 0 0 0 0 0 0 | 90.0 |
| Play Xylophone | 10 / 10 | 20.0 | 20.0 | 67.1 | 0 0 0 0 0 0 100 100 0 0 | 80.0 |
| Deposit Coin | 10 / 10 | 24.0 | 10.0 | 27.0 | 20 100 20 20 0 20 20 20 0 20 | 84.0 |
| Insert Key | 10 / 10 | 15.0 | 0.0 | 27.1 | 15 15 15 15 15 15 15 15 15 15 | 15.0 |
| Build Tower | 10 / 10 | 59.0 | 50.0 | 92.3 | 10 10 10 100 30 30 100 100 100 100 | 100.0 |

### Long-Horizon

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores | SEEK score |
|---|---:|---:|---:|---:|---|---:|
| Fill Pen Holder | 10 / 10 | 10.0 | 0.0 | 69.7 | 0 0 25 0 25 0 0 0 25 25 | 47.5 |
| Classify Objects | 10 / 10 | 94.0 | 90.0 | 63.1 | 100 100 100 100 100 100 100 100 40 100 | 100.0 |
| Put Bottles into Dustbin | 10 / 10 | 64.5 | 50.0 | 42.5 | 40 100 40 100 40 100 100 100 25 0 | 92.5 |
| Play Tic-Tac-Toe | 10 / 10 | 90.0 | 60.0 | 120.7 | 75 100 100 75 100 100 75 100 100 75 | 100.0 |
| Fill Egg Holder | 10 / 10 | 13.5 | 0.0 | 69.4 | 40 25 0 10 0 10 40 0 0 10 | 57.5 |
| Organize Table | 10 / 10 | 62.5 | 0.0 | 95.2 | 75 50 75 50 75 50 75 75 50 50 | 92.5 |
| Make Kong | 10 / 10 | 10.0 | 10.0 | 90.6 | 100 0 0 0 0 0 0 0 0 0 | 0.0 |
| Play Stacking Toy | 10 / 10 | 32.0 | 0.0 | 143.5 | 30 10 30 10 30 30 30 30 60 60 | 92.0 |

### Memory

| Task | Episodes | Score | Success (%) | Requests / episode | Episode scores | SEEK score |
|---|---:|---:|---:|---:|---|---:|
| Cover Blocks | 10 / 10 | 70.0 | 70.0 | 68.4 | 0 100 0 0 100 100 100 100 100 100 | 100.0 |
| Match and Pick from Conveyor | 10 / 10 | 100.0 | 100.0 | 36.2 | 100 100 100 100 100 100 100 100 100 100 | 90.0 |
| Swap Blocks | 10 / 10 | 90.0 | 90.0 | 44.9 | 100 0 100 100 100 100 100 100 100 100 | 100.0 |
| Swap T | 10 / 10 | 60.0 | 60.0 | 37.3 | 100 0 0 100 0 0 100 100 100 100 | 100.0 |
| Press by Number | 10 / 10 | 40.0 | 40.0 | 36.4 | 100 0 0 100 0 0 0 100 0 100 | 90.0 |
| Imitate Sorting Sequence | 10 / 10 | 90.0 | 90.0 | 66.4 | 100 100 100 100 0 100 100 100 100 100 | 81.5 |

Data: [results.json](results.json), [episodes.csv](episodes.csv).
