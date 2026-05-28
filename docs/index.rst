Dark Emulator2 documentation
============================

DarkEmulator2 is a cosmology code for rapid evaluation of large-scale-structure summary statistics developed within the Dark Quest Project (https://darkquestcosmology.github.io/).

The ``dark_emulator2`` Python package provides fast and accurate predictions of the nonlinear and linear matter power spectrum, calibrated on the Dark Quest II (DQ2) simulation suite spanning a nine-dimensional cosmological parameter space of :math:`w_0 w_a \nu o \mathrm{CDM}`.

For the previous public release based on Dark Quest I (DQ1), see the Dark Emulator `documentation <https://dark-emulator.readthedocs.io/>`_ and the `GitHub repository <https://github.com/DarkQuestCosmology/dark_emulator_public/>`_.

Install from PyPI::

	pip install dark_emulator2

Install directly from GitHub::

	pip install git+https://github.com/DarkQuestCosmology/dark_emulator2_public.git

For source or development installation::

	git clone https://github.com/DarkQuestCosmology/dark_emulator2_public.git
	cd dark_emulator2_public
	pip install .

Example notebooks are available in `DarkQuestCosmology/dark_emulator2_public <https://github.com/DarkQuestCosmology/dark_emulator2_public/tree/main/notebook>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
