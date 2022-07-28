
# Building better case definitions for COVID-19
This repo holds the code used to generate the results in our paper "Performance of Existing and Novel Symptom- and Antigen Testing-Based COVID-19 Case Definitions in a Community Setting". For more info about the study, read the preprint on medrXiv [here](https://doi.org/10.1101/2022.05.10.22274914). For more info about the Python scripts, read on.

## Code
Running the following scripts in order will produce all of the that appeared in our paper (though not necessarily with the same formatting, some of which was done by hand), including the case definitions, performance metrics, confidence intervals, and visualizations. 

1. `preprocessing.py` creates the analytic dataset from the raw data.
2. `combos.py` runs the combinatorial optimization and makes a new version of the analytic dataset that includes our novel case definitions.
3. `rf.py` runs the random forest part of our analysis.
4. `inference.py` constructs confidence intervals for our main metrics.
5. `viz.py` makes the figures in the paper.

The other two scripts, `tools.py` and `multi.py` contain the support functions needed to run the other scripts. For the linear-program-based combinatorial optimzation, we use the `case-def-optimization` package, which we included with this repository as a submodule.

## Software
All code was written in Python 3, and the specific dependcies are listed in the [requirements](requirements.txt.) file. The dependencies can be installed one-by-one or by running e.g. `pip install -r requirements.txt`. Where possible, we used the `multiprocessing` module to speed things up.

## Hardware
We ran all our analyses on a Dell scientific workstation with 24 logical cores and 64GB of RAM. The scripts will run fine on smaller machines, but `combos.py` and `rf.py` make take a while to run.

## Related documents

* [Open Practices](doc/open_practices.md)
* [Rules of Behavior](doc/rules_of_behavior.md)
* [Thanks and Acknowledgements](doc/thanks.md)
* [Disclaimer](doc/DISCLAIMER.md)
* [Contribution Notice](doc/CONTRIBUTING.md)
* [Code of Conduct](doc/code-of-conduct.md)

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](doc/DISCLAIMER.md)
and [Code of Conduct](doc/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/privacy.html](http://www.cdc.gov/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page are subject to the [Presidential Records Act](http://www.archives.gov/about/laws/presidential-records.html)
and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).


**General disclaimer** This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  Github is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software.

