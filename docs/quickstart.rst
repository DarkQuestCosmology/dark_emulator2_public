Quickstart
==========

Basic usage
-----------

.. code-block:: python

   from dark_emulator2 import DarkEmulator2 as dq2

   de = dq2()
   param = de.param.get_fid_param()
   k, pk = de.get_pk(param, zred=0.0)
