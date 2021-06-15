# ice
This repo contains code I'm using for analysis on Ice-Binding Proteins (IBPs) for my bachelor project.

## Preliminary installations
Run
```
pip install -r requirements.txt
```
to install all packages I have in my ice environment, there're also some packages which are maybe not needed, though.

## main.py
Adapted from earlier versions (previously The_main_single.py) to calculate the ice volume fraction (Q), mean area (A), mean radius of curvature (r_k), mean radius of curvature cubed (r_k3), circumference (l), area (A) and mean crystal radius (2A/l).
It is possible to plot the radius and area distribution for a few frames.
Running the file will prompt the user to select a frame directory and selecting an ROI.

## extract_frames.py
Run to sample frames from movies. Adapted from earlier versions to extract frames for all movies in the selected directory.

## fit_data.py
Calculate best fits to the data and save them to the existing csv file.
(Also calculate critical radius.)

## Example folder structure
```
Ice
├───analysis (folders containing Q, A, r and r3 info, as well as crystal detection images.)
│   ├───0uM_X_10%_0
│   └───1uM_T18N_20%_0
├───csv (folders containing tracking data for every frame, output from *The_main_single.py*)
│   ├───0uM_X_10%_0
|   └───1uM_T18N_20%_0
├───frames (folders containing many frames sampled with *extract_frames.py*)
│   ├───0uM_X_10%_0
│   └───1uM_T18N_20%_0
└───movies (direct measurements in .avi format)
    ├───0uM_X_10%_0.avi
    └───1uM_T18N_20%_0.avi
```

## Other files
There are more files dedicated to plot different values nicely.