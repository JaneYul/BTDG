# README

The code of "*Block Term Decomposition with Distinct Time Granularities for Temporal Knowledge Graph Completion*".

### Environment

- python 3.6.13

- cuda10.0.130
- pytorch 1.8.0

### Datasets

- ICEWS14 and ICEWS05-15: [Learning Sequence Encoders for Temporal Knowledge Graph Completion](https://github.com/nle-ml/mmkb)

- Wikidata12k: [HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding](https://github.com/malllabiisc/HyTE)

### Test

For ICEWS14 and ICEWS05-15, the coarse time granularity $\Delta t$ is selected from \{year, quarter, month\}, the learning rate $lr$ from \{0.0005, 0.001, 0.005, 0.001\}, the dimension of entity embedding $d_e$ from \{30, 50, 100, 150\}, the dimension of relation embedding $d_r$ from \{20, 30, 50, 100\}, and the time smoothing weight $\alpha$ from \{1, 0.1, 0.01, 0.001\}.

For Wikidata12k, the coarse time granularity $\Delta t$ is selected from \{century, decade, 5 years\}, and the set of candidate values for $lr$, $d_e$, $d_r$, and $\alpha$ are the same as that in ICEWS.

The final parameters selected:

|             | coarse_grain | batch_size | label_smoothing | lr    | dr   | entity_dim | rel_dim | alpha | dropout1 | dropout2 |
| ----------- | ------------ | ---------- | --------------- | ----- | ---- | ---------- | ------- | ----- | -------- | -------- |
| ICEWS14     | month        | 128        | 0.1             | 0.001 | 1    | 200        | 50      | 0.01  | 0.5      | 0.5      |
| ICEWS05-15  | quarterly    | 128        | 0.1             | 0.001 | 1    | 100        | 50      | 0.01  | 0.5      | 0.5      |
| Wikidata12k | years5       | 16         | 0.1             | 0.001 | 1    | 30         | 20      | 0.1   | 0.5      | 0.5      |

Firstly, process the data using:

```bash
python process_data.py
```

Then obtain the test results using:

```bash
python main.py --dataset="icews14" --coarse_grain="month" --batch_size=128 --label_smoothing=0.1 --lr=0.001 --dr=1.0 --entity_dim=200 --rel_dim=50 --alpha=0.01 --dropout1=0.5 --dropout2=0.5
```

```bash
python main.py --dataset="icews05-15" --coarse_grain="quarterly" --batch_size=128 --label_smoothing=0.1 --lr=0.001 --dr=1.0 --entity_dim=100 --rel_dim=50 --alpha=0.1 --dropout1=0.5 --dropout2=0.5
```

```bash
python main.py --dataset="wikidata12k" --coarse_grain="years5" --batch_size=16 --label_smoothing=0.1 --lr=0.001 --dr=1.0 --entity_dim=30 --rel_dim=20 --alpha=0.1 --dropout1=0.5 --dropout2=0.5
```

