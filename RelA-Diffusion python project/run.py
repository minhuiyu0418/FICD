# -*- coding: utf-8 -*-
"""Convenience runner.

From the project root:
  python run_project.py --mode all
"""
import runpy
from pathlib import Path

pkg_main = Path(__file__).parent / "mri_pet_synth" / "main.py"
runpy.run_path(str(pkg_main), run_name="__main__")
