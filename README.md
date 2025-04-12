# variable_search
There are codes and files to extract data from photometrical plates and look for variable or new objects.

### source_extraction_and_astrometry.ipynb

The notebook processes FITS images to:
	•	Subtract background and detect astronomical sources
	•	Convert pixel coordinates to RA/Dec using WCS
	•	Save source catalogs and brightest object lists
	•	Visualize extracted sources and background
	•	Analyze astrometric differences (ΔRA, ΔDEC) across multiple frames

Useful for assessing image quality and preparing input for cross-matching or variability studies.

## catalogs_crossmatch.ipynb

The notebook performs coordinate-based cross-matching between detected sources and a reference catalog:
	•	Filters large catalogs using frame RA/Dec boundaries
	•	Matches sources using angular separation (SkyCoord)
	•	Flags uncertain matches based on configurable radius
	•	Saves full and unmatched source tables (CSV + FITS with metadata)

Useful for catalog enrichment, positional accuracy checks, and preparing data for photometric or classification pipelines.