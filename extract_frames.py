"""
Adapted by Siem de Jong
Adaptation makes it possible to extract frames for all files in a selected directory.
Append x to the filename to mark for exclusion.
"""

import cv2
import os
from tkinter import filedialog
from tkinter import *
from glob import glob
 
# set time interval for frames extraction
time_interval = 10 #seconds

def extractFrames(pathIn, pathOut):
    os.mkdir(pathOut)
 
    cap = cv2.VideoCapture(pathIn)
    frames_tot = cap.get(cv2.CAP_PROP_FRAME_COUNT) #get number of frames in the movie
    f_rate =  cap.get(cv2.CAP_PROP_FPS) #get frame rate SADLY THE MOVIES ARE STORED AT 21.52 FPS INSTEAD OF 13...
    f_start = 0 #starting frame
    f_end =  int(frames_tot) #final frame
    interval = int(f_rate * time_interval)

    print('The frame rate of the movie is: ',f_rate)
    while (cap.isOpened()):

        for i in range(f_start, f_end, interval):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            time = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Capture frame-by-frame
            ret, frame = cap.read()
 
            if ret == True:
                print('Read %d frame: ' % i, ret, time, end='\r')

                cv2.imwrite(os.path.join(pathOut, "{}s.png".format(time/1000)), frame)  # save frame as JPEG file           
                
            else:
                break
 
    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
 
def main(INPUT_FOLDER_NAME, REL_OUTPUT_FOLDER_NAME=os.path.join(os.pardir, 'frames')):
    FRAMES_FOLDER = os.path.join(INPUT_FOLDER_NAME, REL_OUTPUT_FOLDER_NAME)
    # try:
    #     os.mkdir(FRAMES_FOLDER)
    # except FileExistsError:
    #     print(f"{FRAMES_FOLDER} already exists. Aborting.")
    #     exit()

    files = [filename for filename in glob(os.path.join(INPUT_FOLDER_NAME, '*[!x].avi'))] # Find movies without the exclusion mark x at the end.

    for i, movie in enumerate(files):
        print(f"Extracting frames from movie {i+1} of {len(files)}")
        extractFrames(movie, os.path.join(FRAMES_FOLDER, os.path.splitext(os.path.basename(movie))[0]))
 
if __name__=="__main__":
    root = Tk() # File dialog
    INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    root.destroy()

    main(INPUT_FOLDER_NAME)
