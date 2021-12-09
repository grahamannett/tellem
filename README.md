
# Tell Em

# Purpose
Purpose of this is to implement explainability related stuff as clearly as possible.  Ideally will be able to use torch+tensorflow but initially will just focus on torch.  This library should be useable irl for explainability related tasks but as the focus is on clarity and not speed it may not be the most ideal in all cases (probably bigger datasets).

Also another big focus on making it clear and usable is to have tests that are less for verifying you implemented correctly and comparing against precomputed values, but making it easier to create tests if you are trying to implement new methods.  As such there are some specifics based around testing talked about below.  Its somewhat of an overlay over unittest/pytest and a compromise of usability and what I've seen really well written libraries do (for instance some ideas from AllenNLP)

# Implemented
- CAM
- GradCAM
- TCAV
- FGSM
- Perturbation Basics

# Intend to Implement

- Integrated Gradients
- webserver to display some of these!


# Usage

to install and use, download or clone the repository and then cd into the directory and run pip install:

```

git clone git@github.com:grahamannett/tellem.git
cd tellem/
pip install -e .
```

<!-- from there an example of how to use it would be -->

<!-- ```
python examples/example_tcav.py
``` -->

for examples of how to implement a new method check out [docs/guide](docs/guide.md)

To run the tests run

```
pytest tests/
```


# Notes on Testing Related
- I started using unittest but then looked at moving over to pytest as I know a lot of people recommend.  Im somewhat interested in moving stuff from [setup_methods](https://docs.pytest.org/en/latest/how-to/xunit_setup.html#method-and-function-level-setup-teardown) these methods to [pytest fixtures](https://docs.pytest.org/en/latest/how-to/fixtures.html#how-to-fixtures) but also want to make it as simplistic and obvious as possible to make a way to test the explainability method

# Other libraries

- Captum
  - https://arxiv.org/abs/2009.07896
- https://github.com/ourownstory/neural_prophet
- https://github.com/albermax/innvestigate


# Tests
- https://github.com/pytorch/captum/blob/master/tests/attr/test_lime.py


# To Implement
- Taylor Decomposition
- Layer-wise relevence decomposition
  - https://github.com/moboehle/Pytorch-LRP
  - https://git.tu-berlin.de/gmontavon/lrp-tutorial
  -
