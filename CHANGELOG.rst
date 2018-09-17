v1.3.1
------
- Bug fix in 'generic_tree' script. Now 'path_detect_frequency' also uses the voxel size defined in the main script.

v1.3
----
- Major bump in version to point out operational status after series of minor improvements.

v1.2.2.7 
--------
- Minor changes mainly to update for a new stable version.

v1.2.2.6 
--------
- Removed 'future_code' from the package. These codes will be kept aside until they are ready to be added back into the package.
- Completely removed all references for *HDBSCAN* which caused import errors.
- Renamed *automated_separation.large_tree_5* to *automated_separation.generic_tree*.

v1.2.2.5 
--------
- Changed *remove_duplicates* function to allow indices output.
- Temporarily removed *continuous_clustering* module until further improvements.
- Replaced HDBSCAN for DBSCAN in the entire package. This aims to make installation simpler and avoid incompatibilities.
- Set full_matrices to False in *svd_evals* to improve processing efficiency (reduced processing time and memory usage).
- Added new autometed separation script *large_tree_5*.
- Removed old automated separation scripts: *large_tree_1* and *large_tree_2*.
- Added new filters: *plane_filter*, *cluster_filter* and *feature_filter*.
- Added new path detection script, *path_detect_frequency*.

v1.2.2.4
--------
- Corrected automated calculation of parameter cf_rad in *large_tree_3*.
- Added new gmm_nclasses parameter to *large_tree_3*.

v1.2.2.3
--------
- Changed *voxel_path_detect* parameters to speed up processing.
- Added maximum iterations to *detect_main_pathways* to avoid infinite loops or long processing times.

v1.2.2.2
--------
- Bug fixes in *automated_separation.large_tree_3*.

v1.2.2.1
--------
- Fixed base point index in *continuity_filter*.
- Added new voxelization wrapped around *detect_main_pathways* that aims to speed up the processing.
- Added new *automated_separation* script, *large_tree_3*.

v1.2.1.7
--------
- Changed clustering in filtering.cluster_filter from DBSCAN to HDBSCAN in order to improve memory efficiency.
- Minor adjustments in automated_separation.large_tree_1.
- Created new knn optimization function to detect knn values automatically.
- Added block processing to *subset_nbrs*.e
- Minor fixes for improvement on continuity_filter stability. 
- Added new automated separation script, automated_separation.large_tree_2.
- Corrected class_filter application on large_tree_1 and large_tree_2.
- Fixed class_filter input target values (finished changing valid values from 1 or 2 to 0 or 1).
- Added a new final filtering step to large_tree_2 using detect_main_pathways.

v1.2.1.6
--------
- Minor fixes.

v1.2.1.5
--------
- Added verbose option to some modules.
- Changed docstrings style to numpydoc.
- Added default class_ref DataFrame as a built-in object. User now has the option to use this new default or continue to load a .csv file.
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

