# Deployment Strategies for Machine Learning Algorithms in Patient Healthcare Domains: An Imaging Perspective

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-jisoo.chae%40pennmedicine.upenn.edu-blue)](mailto:jisoo.chae@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)

Despite recent advancements in machine learning and artificial intelligence for applications to patient healthcare, real hospital settings have experienced few of the potential benefits and improvements to clinical medicine. To facilitate clinical adaptation of methods in machine learning, we propose **DPSP**: a standardized framework for step-by-step deployment that focuses on four key components: **D**ata acquisition; **P**roblem identification; **S**takeholder alignment; and **P**ipeline integration. We leverage recent literature and empirical evidence in radiologic imaging applications to justify our approach, and offer discussion to help other research groups and hospital practices leverage machine learning to improve patient care.

## Installation

To install and run our code, first clone the `PennMedBiobankAIDeployment` repository.

```
git clone https://github.com/michael-s-yao/PennMedBiobankAIDeployment
cd PennMedBiobankAIDeployment
```

Next, create a virtual environment and install the relevant dependencies.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Quantitative figures presented in our work can be easily reproduced by running the following commands:

```
python PMBB.py --use_sans_serif
python distribution_shift.py
```

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Allison Chae](mailto:jisoo.chae@pennmedicine.upenn.edu)

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

## Citation

When available, relevant citation information will be added in a future commit.

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
