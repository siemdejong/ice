# ice
All things ice.

## The_main_single.py
Adapted from earlier versions to calculate the ice volume fraction (Q), mean area (A), mean radius of curvature (r), mean radius of curvature cubed (r3). Running the file will prompt the user to select a frame directory and selecting an ROI.

## extract_frames.py
Run to sample frames from movies. Adapted from earlier versions to extract frames for all movies in the selected directory.

## combined_graphs.py
Plot graphs (Q, A, r or r3) where axes are categorized by sucrose level and IBP variant. *Doesn't work right now. Switched from json to csv.*

## fit_data.py
Used to calculate best fits to the data. Currently has troubles calculating the best linear fit for some sets.