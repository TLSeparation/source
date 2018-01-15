v1.2.1.5
--------
- Added verbose option to some modules.
- Changed docstrings style to numpydoc.
- Added default class_ref DataFrame as a built-in object. User now has the option to use this new default or continue to load a
	.csv file.
- Added voxels.py module to create voxels from point clouds.
- Added voxelization step in automated_separation.large_tree_1 to improve performance in path_detection.


v1.2.1.4
--------
- Fixed imports. Now, to access any low level function, one has to go through the proper module hierarchy.

v1.2.1.3
--------
- Changed approach of relative import. Removed all sys.path.append statements and adopted double dots (..) for parent folder imports.

v1.2.1.2
--------

- Fixed bug in classification.__init__.py failing to import *wlseparate_ref* as this function no longer exists;
- Updated documentation strings for Sphinx;		

v1.2.1.1
--------
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
------
Corrected list of required packages.

v1.1.3
------
Added new option for automated separation (auto_separation_2).
Renamed old separation.py to auto_separation_1.py.
Added classificaition probability output to gmm.py.
Added classification probability filter to separation. Now all points below some probability threshold will be left unclassified.
Added new wlseparate method to auto_separation_2, based on a voting scheme.

