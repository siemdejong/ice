# ice
This repo contains code I'm using for analysis on Ice-Binding Proteins (IBPs) for my bachelor project.

## The_main_single.py
Adapted from earlier versions to calculate the ice volume fraction (Q), mean area (A), mean radius of curvature (r), mean radius of curvature cubed (r3). Running the file will prompt the user to select a frame directory and selecting an ROI.

## extract_frames.py
Run to sample frames from movies. Adapted from earlier versions to extract frames for all movies in the selected directory.

## combined_graphs.py
Plot graphs (Q, A, r or r3) where axes are categorized by sucrose level and IBP variant. *Doesn't work right now. Switched from json to csv.*

## fit_data.py
Used to calculate best fits to the data. Currently has troubles calculating the best linear fit for some sets.

## Example folder structure
'''bash
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
'''