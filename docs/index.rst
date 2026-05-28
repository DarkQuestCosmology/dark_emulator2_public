Dark Emulator2 documentation
============================

DarkEmulator2 is a public emulator for rapid evaluation of linear and nonlinear matter power spectra, developed within the `Dark Quest Project <https://darkquestcosmology.github.io/>`_.

The ``dark_emulator2`` Python package is calibrated on the Dark Quest II (DQ2) simulation suite and covers a nine-dimensional extended cosmology space including curvature, massive neutrinos, and dynamical dark energy.

Project links
-------------

* `Source repository <https://github.com/DarkQuestCosmology/dark_emulator2_public>`_
* `Example notebooks <https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook>`_
* `Dark Emulator2 paper, arXiv:2605.28596 <https://doi.org/10.48550/arXiv.2605.28596>`_
* `GINKAKU code paper, arXiv:2605.28581 <https://doi.org/10.48550/arXiv.2605.28581>`_
* `Supplementary emulator comparisons <https://darkquestcosmology.github.io/dark_emulator2_supplement/>`_

Installation
------------

Install from PyPI::

	pip install dark_emulator2

Install directly from GitHub::

	pip install git+https://github.com/DarkQuestCosmology/dark_emulator2_public.git

For source or development installation::

	git clone https://github.com/DarkQuestCosmology/dark_emulator2_public.git
	cd dark_emulator2_public
	pip install .

Related projects
----------------

For the previous public release based on Dark Quest I (DQ1), see the Dark Emulator `documentation <https://dark-emulator.readthedocs.io/>`_ and the `GitHub repository <https://github.com/DarkQuestCosmology/dark_emulator_public/>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
