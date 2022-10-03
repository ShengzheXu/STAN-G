# STAN-G Synthetics
General time-series tarbular data synthesis tool box.

Idea initiates with our previous network traffic work [[STAN]](https://sanghani-projects.cs.vt.edu/stan/).

## Setting Up Your Environment

The simplest approach to make it a consistent environment is via Docker.

Step 1: Build a Docker image with pre-defined script.
```sh
cd ./make_nfattacker_docker
docker build -f ./nfattacker2.0 -t nfattacker:v2.0 ./
```

Step 2: Just Run the container.

```sh
docker run --rm -it --name nfattacker -v $(pwd):/workspace nfattacker:v2.0 bash
```

## Learning on Real Data and Generating Synthetic Data

Here is a minimal tutorial to call the API and generate data.

```python
from flowattacker.model import NetflowAttacker
from flowattacker.model.context_loader import ScenarioDataset

dataset = ScenarioDataset(working_dir='tmp')

b2i = NetflowAttacker()
b2i.fit(dataset, num_epochs=10)

synth_data = b2i.sample(n=100000, num_scenario = 10)
```

In addition, a higher-level dynamic context encoding component can be pluged-in.

```python
from flowattacker.model.context_loader import BehaviorContextLoader

bcl = BehaviorContextLoader(short_seq_len=5, long_time_delta=60,long_seq_len=5)

synth_data = b2i.sample(n=100000, num_scenario = 10, bcl=bcl)
```

Check out additional usage examples at [runner.py](./runner.py) .


## Making Your Own Data

Any types of tabular data including `continuous columns` and/or `discrete columns` can be applied, through a pre-defined data preprocessor. For standard tarbular data, just use the standard normalizers etc. For those attributes that you have special meanings, `@override` the `transform` and `rev_transform` functions under the `NetflowProcessor`

```python
from flowattacker.model.netflow_processor import NetflowProcessor

df = pd.read_csv('data/demo.csv')

ntt = NetflowProcessor()
transed = ntt.transform(df)

rev = ntt.rev_transform(transed)
```