<div align="center">
    <img src="images/Logo_ConfianceAI_Blanc.png" height="200" align="">

[![](https://img.shields.io/static/v1?label=&message=Online%20documentation&color=0077de)]([Web site])

[![Online documentation](https://img.shields.io/badge/MPL--2.0-blue)](https://opensource.org/licenses/Apache-2.0)

[![License](https://img.shields.io/badge/MPL--2.0-blue)](https://opensource.org/licenses/Apache-2.0)
<br>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-1c4a6c.svg)](https://flake8.pycqa.org/en/latest/)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

![Activity](https://img.shields.io/github/commit-activity/m/IRT-SystemX/DataQualityMetrics)
![Last commit](https://img.shields.io/github/last-commit/IRT-SystemX/DataQualityMetrics)

![python lib](https://github.com/irt-Systemx/DataQualityMetrics/actions/workflows/python_lib_publish.yml/badge.svg)
![Docker](https://github.com/IRT-SystemX/DataQualityMetrics/actions/workflows/docker_publish.yml/badge.svg)

</div>

##  DataQualityMetrics

Time-series Anomaly Detection Kit with Interactivity and Tools

A toolkit integrating and wrapping all anomaly detection components with a possibility to: interact between the various components (preprocessing, engeering, modeling) interact with a visualization/annotation tool (e.g. Debiai) via an API intended for expert-in-the-loop iterations

The component is referenced by the [European Trustworthy AI Foundation] in its [catalog]

## Documentation

The full documentation is available in the [web site]

## Code of Conduct

Everyone interacting in the project's codebases, issue trackers, chat rooms, and mailing lists is expected to follow the [Code of Conduct](CODE_OF_CONDUCT_v2.md).

## Quality check

Several tools are used for maintaining the quality of the python library, such as code formatting, linting, testing, documentation, and security analysis:
1. Code Formatting: [Black] is an opinionated Python code formatter that automatically formats your code to make it consistent. It helps reduce debates about code style by enforcing a standard format.
2. Linting (Code Quality Checks): [Flake8] is A wrapper around PyFlakes, pycodestyle, and mccabe that provides an easy-to-use interface for linting Python code.
[//]: # ([Pylint] is a widely used linter for Python that checks for errors in Python code, enforces coding standards, and looks for potential code smells.)
3. Testing: [pytest] is a framework that makes it easy to write simple as well as scalable test cases. It also supports fixtures, parameterization, and many plugins.

## Installation

This component is available with pip or as a Docker image. To install it, you can follow the [installation guide].


# Developer Area

## Get started

Please have a look at [the Sphinx documentation]

## Generate docs

````
pip install -r docs/docs_requirements.txt -r requirements.txt
sphinx-apidoc -o docs/source/generated DataQualityMetrics
sphinx-build -M html docs/source docs/build -W --keep-going
````

## Tests

````
# Move into whichever directory you want to serve
cd ./docs/build/html

# Start the http server 
python -m http.server 8080
````

## build and locally run the docker 

````
docker build -t dqm .
docker run --rm dqm python main.py
````

## build and locally run the python lib

````
pip install setuptools wheel

# installs your library in "editable" mode, 
# meaning changes you make to the code will be immediately reflected without needing to reinstall.
pip install -e .
# OR 
pip install .

# check if the library is successfully installed 
pip list | grep dqm
````


---
<p align="center justify-content:space-around">
  This component is maintained by: 
  <a href="https://www.irt-systemx.fr/" title="IRT SystemX">
   <img src="https://www.irt-systemx.fr/wp-content/uploads/2013/03/system-x-logo.jpeg"  height="50">
  </a>
</p>
---

[Component Name]: DataQualityMetrics
[IRT-SystemX]: https://github.com/IRT-SystemX
[PSF Code of Conduct]: https://policies.python.org/python.org/code-of-conduct/
[Support]: support@irt-systemx.fr
[web site]: https://IRT-SystemX.github.io/DataQualityMetrics/
[installation guide]: https://IRT-SystemX.github.io/DataQualityMetrics/
[European Trustworthy AI Foundation]:https://www.confiance.ai/foundation/
[catalog]: https://catalog.confiance.ai/records/sa1gd-1s022
[Black]: https://black.readthedocs.io/
[Pylint]: https://pylint.readthedocs.io/
[pytest]: https://docs.pytest.org/en/stable/
[Flake8]: https://flake8.pycqa.org/
[the Sphinx documentation]: https://www.sphinx-doc.org/fr/master/tutorial/getting-started.html#