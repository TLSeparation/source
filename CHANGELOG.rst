v1.2.1.2
========

- Fixed bug in classification.__init__.py failing to import *wlseparate_ref* as this function no longer exists;
- Updated documentation strings for Sphinx;		

v1.2.1.1
========
This versions has enough important modifications to get a new subversion number, starting the 1.2 phase.

Some of the changes included in this version are:

- Changed *geodescriptors* function name to *knn_features*;
- Updated version number in all files and setup.py;
- Changed *point_features.eigen* (now called knn_evals) name to accommodate for radius and knn options;
- Merged *array_majority* and *array_majority_rad* into the same function. Use kwargs to make it easier to parse arguments;
- Merged *class_filter* and *class_filter_rad* into the same function. Use kwargs to make it easier to parse arguments;
- Changed *point_compare* module name to *data_utils*;
- Revised version of *path_detection*;
- Changed new output configuration to *wlseparate_abs* and *wlseparate_ref_voting*;
- Removed *wlseparate_ref* as it's redundant. Same function can be run by using a single 'knn' parameter value in *wlseparate_ref_voting*;
- Changed *filtering* outputs. Now all functions (except for continuity_filter) output arrays of indices instead of points coordinates.;
- Revised documentation for the whole package. Now, all docstrings are compatible with Sphinx;

v1.1.4
======
Corrected list of required packages.

v1.1.3
======
Added new option for automated separation (auto_separation_2).
Renamed old separation.py to auto_separation_1.py.
Added classificaition probability output to gmm.py.
Added classification probability filter to separation. Now all points below some probability threshold will be left unclassified.
Added new wlseparate method to auto_separation_2, based on a voting scheme.