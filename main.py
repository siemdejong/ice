"""
Adapted by Siem de Jong
Adapted from earlier versions (previously The_main_single.py) to calculate
the ice volume fraction (Q), mean area (A), mean radius of curvature (r_k),
mean radius of curvature cubed (r_k3), circumference (l), area (A) and mean crystal radius (2A/l).
It is possible to plot the radius and area distribution for a few frames.
Running the file will prompt the user to select a frame directory and selecting an ROI.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import sys
from tkinter import filedialog
from tkinter import *
import platform
import subprocess
from glob import glob

class FrameImg:
    ''' Add Description '''
    # ROI_crop = [xleft, ylow, dx, dy]
    ROI_crop = (254, 366, 625, 642)
    
    crop_boo = True

    if 'ROI_crop' not in locals():
        ROI_crop = None
    else:
        ylow = ROI_crop[1]
        yup = ROI_crop[1] + ROI_crop[3]
        xleft = ROI_crop[0]
        xright = ROI_crop[0] + ROI_crop[2]

    def __init__(self, file_name, file_path = os.getcwd(), frame_num = 1):
        self.file_name = file_name
        self.file_path = file_path
        self.frame_num = frame_num

        img, img_treshold = self.load_img()
        self.get_img_contours(img_treshold)
        self.process_contours()
        self.create_crystal_attr_list()
        self.check_contours()
        # self.drop_upper_outlier_area_crystal()
        # self.drop_edge_contours()

    def drop_upper_outlier_area_crystal(self):
        self.crystal_areas.sort()
        c_max = self.crystal_areas[len(self.crystal_areas) - 1] # Get largest crystal area
        c_max_s = self.crystal_areas[len(self.crystal_areas) - 2] # Get second largest crystal area
        area_ratio = c_max / c_max_s
        if area_ratio > 1: # If largest area is 10 times that of the second largest area
            max_crys = [c for c in self.crystalobjects if c.area == c_max][0] # Retreive crystal object
            print(f'Dropping max crystal, size is {round(area_ratio,2)} times the second biggest crystal')
            # Remove crystal attributes from list, and the crystal form the crystalobjects list.
            self.crystal_areas.remove(max_crys.area)
            self.crystal_lengths.remove(max_crys.length)
            self.contours_lenghts.remove(max_crys.contour_length)
            # self.crystal_centers.remove(max_crys.center_arr)
            # Currently not removing the center coord of max crys. Might cause issues late on
            self.crystalobjects.remove(max_crys)


    def load_img(self):
        ''' Loads in the image, and crops if it crop_boo is set to True. '''
        img_path = os.path.join(self.file_path, self.file_name)
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if FrameImg.crop_boo:
            # When no ROI is predefined, select the ROI.
            if FrameImg.ROI_crop is None:
                FrameImg.ROI_crop = cv2.selectROI('Select ROI', self.img, showCrosshair=False)
                cv2.destroyWindow('Select ROI')
                FrameImg.ylow = FrameImg.ROI_crop[1]
                FrameImg.yup = FrameImg.ROI_crop[1] + FrameImg.ROI_crop[3]
                FrameImg.xleft = FrameImg.ROI_crop[0]
                FrameImg.xright = FrameImg.ROI_crop[0] + FrameImg.ROI_crop[2]
            img = self.img[FrameImg.ylow:FrameImg.yup, FrameImg.xleft:FrameImg.xright]
        else:
            img = self.img

        self.img_attributes(img)
        img_denoised = self.denoise_img(img)
        # img_denoised = self.sharpen_img(img_denoised)
        img_treshold = self.tresholding_img(img_denoised)
        # self.plot_stages(img, img_denoised, img_treshold)
        # Add in possible plot stages method here before the images go out of scope.
        return img, img_treshold

    def plot_stages(self, img, img_denoised, img_treshold):
        images = [img, img_denoised, img_treshold]
        titles = ['original', 'denoised', 'threshold']
        for i in range(len(images)):
            plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
            plt.xticks([]), plt.yticks([])
            plt.title(titles[i])
        plt.suptitle(self.file_name, fontsize = 8)
        plt.show()

    def img_attributes(self, img):
        ''' Retreive image height and width and store them as an attribute. Used for checking if the
        contours are flipped '''
        self.img_height , self.img_width = img.shape[:2]

    def denoise_img(self, img):
        ''' Desnoise image. 
            Docs : https://docs.opencv.org/2.4/modules/photo/doc/denoising.html '''
        return cv2.fastNlMeansDenoising(src=img, dst=None, h=10,
        templateWindowSize=11, searchWindowSize=27)
    
    def sharpen_img(self, img):
        """Sharpen image using unsharp masking."""
        gaussian_3 = cv2.GaussianBlur(img, (0, 0), 1)
        unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
        return unsharp_image

    def tresholding_img(self,img_denoised):
        ''' Treshold image. 
            Docs: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 '''
        return cv2.adaptiveThreshold(src=img_denoised, maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv2.THRESH_BINARY, blockSize=threshold_blocksize, C=threshold_constant) # big crystals -> high blockSize; small crystals -> small blockSize.

    def get_img_contours(self, img_treshold):
        ''' Retreive image contours. ## NEED TO WORK WITH RETR_CCOMP instead of RETR_EXTERNAL TO GET HIERACHY FOR DEALIG WITH INCEPTIONS
            Docs: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a'''
        self.contours, self.hierarchy = cv2.findContours(img_treshold,
            cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    def process_contours(self):
        ''' First creates two empty lists to store the crystals and the remaining/other objects. Next,
            loops through all the retreives contours: If the len(contour), which is the amount of coordinate
            points is large enough, it creates a CrystalObject class instance. Depending on whether this is a contour
            with a child or not, child contour info will be passed to the crystalobject as well.
            If a contour is a child, no seperate object is created for those contour. Then, if the 'contour lenght'
            is greater than 2, it adds the CrystalObject to the list. If the conditions are not met, the object
            is stored in the other objects list.   '''
        self.crystalobjects = []
        self.otherobjects = []
        num_contours = len(self.contours)
        print(f'Number of contours found: {num_contours}')
        children = []
        parents = []
        for i, contour in enumerate(self.contours):
            if len(contour) > 30: # Checks the amount of coordinate points in the contour
                print(f'Processing img contours {i}/{num_contours}', end = '\r')

                if self.hierarchy[0][i][3] == -1:  # Only create a crystal when it's not a hole
                    # Check whether contour is a parent, if so add the childrens contour as input as well
                    if self.hierarchy[0][i][2] != -1:
                        child_contour = self.contours[i+1]
                        obj = CrystalObject(contour, self.hierarchy[0][i], child_contour, True, contour_num=i, frame_num=self.frame_num)
                    else:
                        obj = CrystalObject(contour, self.hierarchy[0][i], None, False, contour_num=i, frame_num=self.frame_num)

                    if obj.x_center == 0 and obj.y_center == 0:
                        self.otherobjects.append(obj)
                    else:
                        self.crystalobjects.append(obj)

        print('Contour processing done  .......')
        print(f'Number of contours stored: {len(self.crystalobjects)}')
        print(f'Number of other objects stored: {len(self.otherobjects)}')

    def create_crystal_attr_list(self):
        self.crystal_areas = [i.area for i in self.crystalobjects]
        self.crystal_lengths = [i.length for i in self.crystalobjects]
        self.contours_lenghts = [i.contour_length for i in self.crystalobjects]
        self.crystal_centers = [i.center_arr for i in self.crystalobjects]

    def check_contours(self):
        ''' Function thats checks if the x and y axes have been swapped, and corrects this if so.
            Done by checking if the highest x found in the contour coordinates if higher than the image
            width, or the higest y contour coordinate is higher than the image height. 
            Correction is done for both the smoothed contours array, and the smoothed contours dataframe.
            Whether the action is performed or not is stored in the flipped contour attribute.  '''
        self.y_maximum = max([i.s_contours_df['y'].max() for i in self.crystalobjects])
        self.x_maximum = max([i.s_contours_df['x'].max() for i in self.crystalobjects])
        if self.x_maximum > self.img_width or self.y_maximum > self.img_height:
             self.flipped_contours = True
             print(f'{self.x_maximum} vs {self.img_width}; {self.y_maximum} vs {self.img_height} ')
             for c in self.crystalobjects:
                # c.s_contours[:, 0], c.s_contours[:, 1] = c.s_contours[:, 1], c.s_contours[:, 0].copy()
                c.s_contours_df.rename(columns={'x': 'y', 'y': 'x', 'sx' : 'sy', 'sy' : 'sx'}, inplace=True)
        else:
            self.flipped_contours = False
        print(f'Contours flipped: {self.flipped_contours}')


    def drop_edge_contours(self):
        ''' Funtion to cutoff the crystals that are within a 'cutoff_pct' * the width and height of the img.
            First, creates a list of cutoff coordinate values for both axis, then loops through each crystal countour 
            coorinates to check if the cutoff coordinate values occur in the smoothed contours coordinates.
            If so, it removes the crystal from the crystalobjects list, and adds it to the edge_objects list. ''' 
        self.edge_objects = []
        cutoff_pct = 0.00
        
        cutoff_y_val = cutoff_pct * self.img_height
        y_drop = [self.img_height - i for i in range(0,int(cutoff_y_val))]        
        y_drop.extend(list(range(0,int(cutoff_y_val))))
        
        cutoff_x_val = cutoff_pct * self.img_width
        x_drop = [self.img_width - i for i in range(0,int(cutoff_x_val))]
        x_drop.extend(list(range(0,int(cutoff_x_val))))
        pre_drop_count = len(self.crystalobjects)
        for crystal in reversed(self.crystalobjects):
            for valx, valy in zip(x_drop, y_drop):
                if valx in crystal.s_contours_df.sx.values or \
                        valy in crystal.s_contours_df.sy.values:
                    self.crystalobjects.remove(crystal)
                    self.edge_objects.append(crystal)
                    break  # to stop the loop   
        post_drop_count = len(self.crystalobjects)
        print(f'Edge dropping dropped {pre_drop_count - post_drop_count} crystals')       

    def plot_contours(self, mark_center = False, mark_number = False,
            save_image = False, file_name = f'contourplot1.png'):
        ''' Plot the contours on the orignal image. '''
        self.contoured_img = self.img
        num_crystal = len(self.crystalobjects)
        print(f'Total number of contours to plot: {num_crystal}')
        for i, crystal in enumerate(self.crystalobjects):
            print(f'Drawing contour of crystal {i} / {num_crystal}', end = '\r')
            # Draw original contours in black
            cv2.drawContours(self.contoured_img, crystal.contour_raw,
                                -1, (0, 0, 0), 1)
            # Draw smoothed countours in white.
            cv2.drawContours(self.contoured_img, [crystal.s_contours.astype(int)],
                                -1, (255, 255, 255), 1)
            if mark_center:
                cv2.circle(self.contoured_img, (crystal.x_center, crystal.y_center), 1,
                    (255, 255, 255), -1)
            if mark_number:
                cv2.putText(self.contoured_img, f'{crystal.contour_num}', (crystal.x_center - 2,
                    crystal.y_center - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if save_image:
                cv2.imwrite(file_name, self.contoured_img)
        print('Crystal contour drawing done ..................')
        cv2.imshow(f'{self.file_name}', self.contoured_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_s_contours(self, show_plot = False,
            save_image = False, frame_img_name = '',
            file_name = f'contourplot3.png'):
        ''' Plots the smoothed contours for each crystalobject in the frame on the original image.  '''
        fig = plt.figure()
        if FrameImg.crop_boo:
            org_img = mpimg.imread(os.path.join(self.file_path,self.file_name))[FrameImg.ylow:FrameImg.yup, FrameImg.xleft:FrameImg.xright]
        else:
            org_img = mpimg.imread(os.path.join(self.file_path,self.file_name))
        plt.imshow(org_img)
        for crystal in self.crystalobjects:
            # plt.plot(crystal.s_contours[...,0], crystal.s_contours[...,1],'k', linewidth=1)
            plt.scatter(crystal.s_contours[...,0], crystal.s_contours[...,1], s=0.05, color='red')
            # plt.scatter(crystal.center_arr[0],crystal.center_arr[1], s=2)
        fig.suptitle(frame_img_name, fontsize = 8)
        if save_image:
            fig.savefig(f'{file_name}')
        plt.close()


class CrystalObject:
    '''The crystal object keeps track of all the attributes of a crystal belonging to a contour in single frame
        NB: there are two very similar dataframe functions for the case of regular crystal or crystal with hole.'''
    def __init__(self, contour_raw, hierarchy, child_contour_raw, parent_bool, contour_num, frame_num):
        self.contour_raw = contour_raw
        self.hierarchy = hierarchy
        self.frame_num = frame_num
        self.contour_num = contour_num
        self.contour_length = len(self.contour_raw)
        self.parent_bool = parent_bool
        self.child_contour_raw = child_contour_raw

        self.child_moments = cv2.moments(child_contour_raw)
        self.moments = cv2.moments(contour_raw)
        self.get_center_point()

        if self.contour_length > 2:
            self.get_area()
            self.get_length()
            # This is where the separate smoothing paths are chosen based on regular or crystal with hole
            if self.parent_bool == True:
                self.set_contours_dataframe()
                self.set_contours_child_dataframe()
                self.s_contours_df = pd.concat([self.s_contours_df, self.s_contours_df_child])
            else:
                self.set_contours_dataframe()
            self.s_contours = self.s_contours_df.drop(['curvature','mean(curvature)', 'sx', 'sy', 'contour_num', 'frame', 'area', 'length'], axis=1).to_numpy()

    def get_center_point(self):
        ''' Using the contour moments, retreives the center x and y coordinates,
            and an array of said coordinates. Initially just calculates the COM of the outer contour. However, if there
            is a hole in the contour the COM will be recomputed with the hole taken into account'''

        # calculate crystal moments and center coords
        if self.moments['m00'] != 0:
            self.x_center = int(self.moments['m10'] / self.moments['m00'])
            self.y_center = int(self.moments['m01'] / self.moments['m00'])
        else:
            self.x_center = 0
            self.y_center = 0

        # in parent case, change the COM according to the hole
        if self.parent_bool == True:
            # calculate child moments and COM
            if self.child_moments['m00'] != 0:
                self.x_center_child = int(self.child_moments['m10'] / self.child_moments['m00'])
                self.y_center_child = int(self.child_moments['m01'] / self.child_moments['m00'])
            else:
                self.x_center_child = 0
                self.y_center_child = 0
            # calculate common COM, assuming uniform thickness and density
            a = cv2.contourArea(self.contour_raw) # outer area
            a1 = cv2.contourArea(self.child_contour_raw) # hole area
            a2 = a - a1 # crystal area
            # print(f'moments (parent, child): {self.moments}, {self.child_moments}')
            # print(f"before any computation the center of parent is ({self.x_center, self.y_center})")
            # print(a1, a2, cv2.contourArea(self.contour_raw), cv2.contourArea(self.child_contour_raw))
            if a2 > 0:
                self.x_center = (a * self.x_center - a1 * self.x_center_child) / (a2)
                self.y_center = (a * self.y_center - a1 * self.y_center_child) / (a2)
            # print(f"After computation the center of parent now is is ({self.x_center, self.y_center})")

        self.center_arr = np.array([self.x_center, self.y_center])

    def get_area(self):
        ''' Sets the area of the contour. Two separate paths for regular crystals of crystals with holes'''
        if self.parent_bool == True:
            self.area = cv2.contourArea(self.contour_raw) - cv2.contourArea(self.child_contour_raw)
        else:
            self.area = cv2.contourArea(self.contour_raw)

    def get_length(self):
        ''' Sets the 'length' of the contour. Two separate paths for regular crystals of crystals with holes'''
        if self.parent_bool == True:
            self.length = cv2.arcLength(self.contour_raw, True) + cv2.arcLength(self.child_contour_raw, True)
        else:
            self.length = cv2.arcLength(self.contour_raw, True)

    def calculate_curvature(self, df, smoothing):
        ''' Calculates the curvature, ands its mean, of the of curvature coordinates, and creates the sx and sy
            columns, which are the rounded values of the coordinates which used in the edge crystal removal function.
            This function starts by creating df columns of the first and second derivate, together with the helper
            columns. Next, creates the curvature and mean curvature columns, and THRESH_BINARY drops the created
            helper columns. Finally, it removes the padding rows (previously done by the listacrop function). '''
        for z in ['x', 'y']:
            df[f'd{z}'] = np.gradient(df[f'{z}'])
            df[f'd{z}'] = df[f'd{z}'].rolling(smoothing, center=True).mean()
            df[f'd2{z}'] = np.gradient(df[f'd{z}'])
            df[f'd2{z}'] = df[f'd2{z}'].rolling(smoothing, center=True).mean()
            df[f's{z}'] = df[f'{z}']#.round(2)
        df['curvature'] = df.eval('(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
        df['curvature'] = df.curvature.rolling(smoothing, center=True).mean()#.round(2)
        self.mean_curvature = df['mean(curvature)'] = np.mean(abs(df.curvature))
        df = df.drop(['dx','d2x','dy','d2y'], axis=1)
        df = df.dropna()    
        return df        

    def set_contours_dataframe(self, smoothing=10):
        '''
        Function to prepare the crystal dataframe in case of a regular crystal. Loads in the contours array, padds them (reason?), and then
        smooths the coordinates using rolling mean. Next, calls the calculate_cruvature function in order to
        retrieve the curvature and its mean. Adds the contour number, frame, area, and lenght of the contour,
        and saves the resulting df. Finally, it creates an additional df with the rounded coordinates.
        '''

        contours_raw = np.reshape(self.contour_raw, (self.contour_raw.shape[0],2))
        contours = np.pad(contours_raw, ((20,20), (0,0)), 'wrap')
        df = pd.DataFrame(contours, columns={'x', 'y'})
        df = df.reset_index(drop=True).rolling(smoothing).mean().dropna()
        df = self.calculate_curvature(df, smoothing)
        df['contour_num'] = self.contour_num
        df['frame'] = self.frame_num
        df['area'] = self.area
        df['length'] = self.length
        self.s_contours_df = df
        
    def set_contours_child_dataframe(self, smoothing=10):
        '''
        Function to prepare the crystal dataframe. Loads in the contours array, padds them (reason?), and then
        smooths the coordinates using rolling mean. Next, calls the calculate_cruvature function in order to
        retrieve the curvature and its mean. Adds the contour number, frame, area, and lenght of the contour,
        and saves the resulting df. Finally, it creates an additional df with the rounded coordinates.
        '''

        child_contours_raw = np.reshape(self.child_contour_raw, (self.child_contour_raw.shape[0], 2))
        child_contours = np.pad(child_contours_raw, ((20, 20), (0, 0)), 'wrap')
        df = pd.DataFrame(child_contours, columns={'x', 'y'})
        df = df.reset_index(drop=True).rolling(smoothing).mean().dropna()
        df = self.calculate_curvature(df, smoothing)
        df['contour_num'] = self.contour_num
        df['frame'] = self.frame_num
        df['area'] = self.area
        df['length'] = self.length
        self.s_contours_df_child = df


def get_img_files_ordered(dir_i):
    """ Function returns a list of all input frames, ordered by their frame number, 
    together with a count of the total number of frames.  """
    img_files = []
    for filename in [os.path.basename(filename) for filename in glob(os.path.join(dir_i, '*[!x].png'))]: # Allow for exclusion (x) of images (masking is a better solution).
    # for filename in os.listdir(dir_i)
        try:
            file_i = {
            'filename' : filename,
            # 'file_num' : int(file.split('frame')[1].split('.')[0])
            # The data acquired by Siem after using extract_frames.py by Agata uses another file naming scheme.
            # (<time>s.png instead of frame<frame>.png)
            'file_num': int(filename.split('.')[0]) 
            }
            img_files.append(file_i)
        except IndexError:
            pass
    ordered_img_files = sorted(img_files, key=lambda k: k['file_num'])
    file_count = len(ordered_img_files)
    return ordered_img_files,file_count

def set_and_check_folder(FOLDER_NAME, create_boo=False):
    """ Function to set up a directory and check if it exists."""
    fol_path = os.path.join(os.getcwd(), FOLDER_NAME)
    if os.path.isdir(fol_path):
        return fol_path
    else:
        if create_boo:
            os.mkdir(fol_path)
            print(f'Dir {FOLDER_NAME} has been created.')
            return fol_path
        else:  
            print(f'Dir {FOLDER_NAME} does not seem to exist. Terminating program.')
            sys.exit()

def create_frame_list(img_files, file_count, imgs_dir,
        output_img_dir, IMAGE_FORMAT, plot_boolean = False):
    """ Function that loops through the input img files folder, and creates
    and instance of FrameImg for each image found with the correct image format.
    Optionally, plots all found contours on the inputted image. """
    frame_list = []
    for f_numerator, file in enumerate(img_files):
        file_name = file['filename']
        if file_name.endswith(IMAGE_FORMAT):
            print('------------------------------------------------------')
            print(f'Processing: {file_name}; #{f_numerator + 1}/{file_count}')
            frame_i = FrameImg(file_name, imgs_dir, f_numerator)
            frame_list.append(frame_i)
            if plot_boolean:
                frame_i.plot_s_contours(show_plot = False, save_image = True, frame_img_name = file_name,
                    file_name = os.path.join(output_img_dir, f'contour_plot_frame_{f_numerator + 1}{IMAGE_FORMAT}'))
        else:
            print(f'{file_name} has a different file format than the expected {IMAGE_FORMAT}.')
    return frame_list

def common_COM(p1, a1, p2, a2):
    """Function that returns that common COM of two objects based on the separate COM's
        and the areas of the objects"""
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    A = a1 + a2

    x_common = (x1 * a1 + x2 * a2) / A
    y_common = (y1 * a1 + y2 * a2) / A

    return x_common, y_common

def open_file(path):
    """Because I don't want to look for the right directory. Just gimme da folder pls."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

# Below, functions come in groups of two/three:
#   Making a list - (Calculating) - Plotting
def plot_ivf(ice_volume_fraction_list, times):
    """Plot the volume fraction in time."""
    fig_volume_fraction = plt.figure()
    fig_volume_fraction.tight_layout()
    gs_vol_frac = fig_volume_fraction.add_gridspec(nrows=1, ncols=1)
    fig_volume_fraction_ax = fig_volume_fraction.add_subplot(gs_vol_frac[0, 0])
    fig_volume_fraction_ax.scatter(times, ice_volume_fraction_list)
    fig_volume_fraction_ax.set_ylabel('Ice volume fraction')
    fig_volume_fraction_ax.set_xlabel('Time [s]')
    fig_volume_fraction.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_volume_fraction.savefig(os.path.join(output_img_dir, 'volume_fraction.png'), bbox_inches='tight')
    plt.close()

def calculate_volume_fraction(frame_list):
    """Calculate the volume fraction Q = v_ice / (v_ice + v_liq) of the ROI."""
    print("Calculating volume fractions.")
    ice_volume_fraction_list = []
    times = []
    
    for frame in frame_list:
        tot_area = np.sum([crystal.area for crystal in frame.crystalobjects])
        ROI_area = frame.img_width * frame.img_height
        ice_volume_fraction = tot_area / ROI_area # Assuming tot_area and ROI_area have the same units.
        ice_volume_fraction_list.append(ice_volume_fraction)
        time = float(frame.file_name.split('s')[0])
        times.append(time)
    
    plot_ivf(ice_volume_fraction_list, times)

    return times, ice_volume_fraction_list # Also return times for use on other analysis.

# --------------------------

def plot_crystal_numbers_per_ROIarea(crystal_numbers_per_px2, times, area=100):
    """Plot the amount of crystals per area in time.
    area: the area in square micrometers in which you will find the amount of crystals
    """
    fig_crystal_numbers = plt.figure()
    fig_crystal_numbers.tight_layout()
    gs_vol_frac = fig_crystal_numbers.add_gridspec(nrows=1, ncols=1)
    fig_crystal_numbers_ax = fig_crystal_numbers.add_subplot(gs_vol_frac[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_crystal_numbers_ax.scatter(times, crystal_numbers_per_px2 / space_scale**2 * 1e-12 * area)
    fig_crystal_numbers_ax.set_ylabel(f'Number of crystals per {area} $\mu$m$^2$')
    fig_crystal_numbers_ax.set_xlabel('Time [s]')
    fig_crystal_numbers.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_crystal_numbers.savefig(os.path.join(output_img_dir, 'number_of_crystals.png'), bbox_inches='tight')
    plt.close()


def amount_of_crystals_per_ROIarea(frame_list):
    """Calculate the amount of crystals per ROI area in the ROI in time."""
    print("Plotting number of crystals")
    crystal_numbers = []
    times = []
    for frame in frame_list:
        number_of_crystals_in_frame = len(frame.crystalobjects)
        crystal_numbers.append(number_of_crystals_in_frame)
        time = float(frame.file_name.split('s')[0])
        times.append(time)
    
    ROI_area = frame_list[0].img_width * frame_list[0].img_height
    crystal_numbers_per_px2 = np.array(crystal_numbers) / ROI_area
    
    plot_crystal_numbers_per_ROIarea(crystal_numbers_per_px2, times)

    return crystal_numbers

# --------------------------

def plot_avg_crystal_area(avg_crystal_areas, times):
    """Plot the amount of crystals in time."""
    fig_avg_crystal_areas = plt.figure()
    fig_avg_crystal_areas.tight_layout()
    gs_vol_frac = fig_avg_crystal_areas.add_gridspec(nrows=1, ncols=1)
    fig_avg_crystal_areas_ax = fig_avg_crystal_areas.add_subplot(gs_vol_frac[0, 0])
    fig_avg_crystal_areas_ax.scatter(times, np.asarray(avg_crystal_areas)*space_scale**2*1e12)
    fig_avg_crystal_areas_ax.set_ylabel(r'Mean crystal area [$\mu$m$^2$]')
    fig_avg_crystal_areas_ax.set_xlabel('Time [s]')
    fig_avg_crystal_areas.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_avg_crystal_areas.savefig(os.path.join(output_img_dir, 'average_crystal_area.png'), bbox_inches='tight')
    plt.close()

def calculate_avg_crystal_area(frame):
    """Calculate the average crystal area inside a frame."""
    avg = np.mean([crystal.area for crystal in frame.crystalobjects])

    return avg

def avg_crystal_area(frame_list):
    """Plot the average crystal area in time."""
    print("Calculating average areas per time.")

    avg_area_list = list(map(calculate_avg_crystal_area, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_avg_crystal_area(avg_area_list, times)

    return avg_area_list

# --------------------------

def plot_mean_radius_of_curvature(mean_radius_list, times):
    """Plot the amount of crystals in time."""
    fig_mean_radius = plt.figure()
    fig_mean_radius.tight_layout()
    gs_mean_radius = fig_mean_radius.add_gridspec(nrows=1, ncols=1)
    fig_mean_radius_ax = fig_mean_radius.add_subplot(gs_mean_radius[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_mean_radius_ax.scatter(times, np.asarray(mean_radius_list)*space_scale*1e6)
    fig_mean_radius_ax.set_ylabel(r'Mean crystal radius [$\mu$m]')
    fig_mean_radius_ax.set_xlabel('Time [s]')
    fig_mean_radius.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_mean_radius.savefig(os.path.join(output_img_dir, 'mean_radius.png'), bbox_inches='tight')
    plt.close()


def calculate_mean_radius_of_curvature(frame):
    """Calculate the mean radius from the mean curvature by taking its reciprocal."""
    mean_curv = np.mean([crystal.mean_curvature for crystal in frame.crystalobjects])
    mean_radius = 1 / mean_curv
    return mean_radius


def mean_radius_of_curvature(frame_list):
    """Plot the mean radius of curvature of the crystals in the ROI."""
    print("Calculating mean radius of curvature per time")

    mean_radius_list = list(map(calculate_mean_radius_of_curvature, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_mean_radius_of_curvature(mean_radius_list, times)

    return mean_radius_list

# --------------------------

def plot_median_mean_radius(mean_radius_list, times):
    """Plot the amount of crystals in time."""
    fig_mean_radius = plt.figure()
    fig_mean_radius.tight_layout()
    gs_mean_radius = fig_mean_radius.add_gridspec(nrows=1, ncols=1)
    fig_mean_radius_ax = fig_mean_radius.add_subplot(gs_mean_radius[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_mean_radius_ax.scatter(times, np.asarray(mean_radius_list)*space_scale*1e6)
    fig_mean_radius_ax.set_ylabel(r'median of mean crystal radius [$\mu$m]')
    fig_mean_radius_ax.set_xlabel('Time [s]')
    fig_mean_radius.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_mean_radius.savefig(os.path.join(output_img_dir, 'median_mean_radius.png'), bbox_inches='tight')
    plt.close()


def calculate_median_mean_radius(frame):
    """Calculate the mean radius from the mean curvature by taking its reciprocal."""
    mean_curv = np.median([crystal.mean_curvature for crystal in frame.crystalobjects])
    mean_radius = 1 / mean_curv
    return mean_radius


def median_mean_radius(frame_list):
    """Plot the mean radius of the crystals in the ROI."""
    print("Calculating mean radius of curvature per time")

    median_mean_radius_list = list(map(calculate_median_mean_radius, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_median_mean_radius(median_mean_radius_list, times)

    return median_mean_radius_list

# --------------------------

def plot_mean_radius_of_curvature3(mean_radius3_list, times):
    """Plot the amount of crystals in time."""
    fig_mean_radius3 = plt.figure()
    fig_mean_radius3.tight_layout()
    gs_mean_radius3 = fig_mean_radius3.add_gridspec(nrows=1, ncols=1)
    fig_mean_radius3_ax = fig_mean_radius3.add_subplot(gs_mean_radius3[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_mean_radius3_ax.scatter(times, np.asarray(mean_radius3_list)*space_scale**3*1e18)
    fig_mean_radius3_ax.set_ylabel(r'Mean crystal radius cubed [$\mu$m$^3$]')
    fig_mean_radius3_ax.set_xlabel('Time [s]')
    fig_mean_radius3.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_mean_radius3.savefig(os.path.join(output_img_dir, 'mean_radius3.png'), bbox_inches='tight')
    plt.close()

def calculate_mean_radius_of_curvature3(frame):
    """Calculate the mean radius cubed from the mean curvature by taking its reciprocal and cubing the outcome."""
    mean_curv = np.mean([crystal.mean_curvature for crystal in frame.crystalobjects])
    mean_radius = 1 / mean_curv
    return mean_radius**3

def mean_radius_of_curvature3(frame_list):
    """Plot the mean radius cubed of the crystals in the ROI."""
    print("Calculating mean radius cubed per time")

    mean_radius3_list = list(map(calculate_mean_radius_of_curvature3, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_mean_radius_of_curvature3(mean_radius3_list, times)

    return mean_radius3_list

# --------------------------

def circumference(frame_list):
    """Plot the mean circumference of the crystals in the ROI."""
    print("Calculating mean circumference per time.")

    def calculate_mean_circumference(frame):
        """Calculate the mean circumference per frame."""
        mean_circ = np.mean([crystal.length for crystal in frame.crystalobjects])
        return mean_circ
    
    def plot_mean_circ(mean_circ_list, times):
        """Plot the mean circumference per time."""
        space_scale = 86.7*10**(-9) #m

        fig_mean_circ = plt.figure()
        fig_mean_circ.tight_layout()
        gs_mean_circ = fig_mean_circ.add_gridspec(nrows=1, ncols=1)
        fig_mean_circ_ax = fig_mean_circ.add_subplot(gs_mean_circ[0, 0])
        # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
        fig_mean_circ_ax.scatter(times, np.asarray(mean_circ_list)*space_scale*1e6)
        fig_mean_circ_ax.set_ylabel(r'Mean circumference [$\mu$m]')
        fig_mean_circ_ax.set_xlabel('Time [s]')
        fig_mean_circ.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                    wspace=0.3, hspace=0.5)
        fig_mean_circ.savefig(os.path.join(output_img_dir, 'mean_circumference.png'), bbox_inches='tight')
        plt.close()

    mean_circumference_list = list(map(calculate_mean_circumference, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]

    plot_mean_circ(mean_circumference_list, times)

    return mean_circumference_list

# --------------------------

def plot_mean_radius3_A_div_l(mean_radius3_list, times):
    """Plot the amount of crystals in time."""
    fig_mean_radius3_Al = plt.figure()
    fig_mean_radius3_Al.tight_layout()
    gs_mean_radius3_Al = fig_mean_radius3_Al.add_gridspec(nrows=1, ncols=1)
    fig_mean_radius3_Al_ax = fig_mean_radius3_Al.add_subplot(gs_mean_radius3_Al[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_mean_radius3_Al_ax.scatter(times, np.asarray(mean_radius3_list)*space_scale**3*1e18)
    fig_mean_radius3_Al_ax.set_ylabel(r'$\langle R \rangle^3$ [$\mu$m$^3$]')
    fig_mean_radius3_Al_ax.set_xlabel('Time [s]')
    fig_mean_radius3_Al.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_mean_radius3_Al.savefig(os.path.join(output_img_dir, 'mean_radius3_a_div_l.png'), bbox_inches='tight')
    plt.close()

def calculate_mean_radius3_A_div_l(frame):
    mean_radius = np.mean([2 * crystal.area / crystal.length for crystal in frame.crystalobjects])
    return mean_radius**3

def r3_by_A_div_l(frame_list, plot=True):
    """Compute <r>^3 by doing <2*A/l>^3"""
    print("Calculating mean r cubed by doing A / l per time.")
    mean_radius3_list = list(map(calculate_mean_radius3_A_div_l, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]

    if plot:
        plot_mean_radius3_A_div_l(mean_radius3_list, times)

    return mean_radius3_list

# --------------------------
# Below, two functions are posed to show the distribution of radii and areas.

def r_distribution(frame_list):
    """Calculate the distribution of radii. Only use this function when you select A FEW frames for analysis."""
    # Only plot area distributions if 10 frames are selected.
    if len(frame_list) < 10:
        fig, axs = plt.subplots(1, 2)

        for frame in frame_list:
            rk = 1 / np.array([crystal.mean_curvature for crystal in frame.crystalobjects]) * space_scale * 1e6
            r = np.array([2 * crystal.area / crystal.length for crystal in frame.crystalobjects]) * space_scale * 1e6
            mean_area = np.mean([crystal.area for crystal in frame.crystalobjects])
            mean_circ = np.mean([crystal.length for crystal in frame.crystalobjects])
            axs[0].hist(rk, bins=20)
            axs[1].hist(r, bins=20)
            axs[0].vlines(np.nanmean(rk), 0, 20, color='red', label='mean') # Use nanmean, because some rk are apparantly still nans...
            axs[0].vlines(np.nanmedian(rk), 0, 20, color='blue', label='median')
            axs[1].vlines(np.nanmean(r), 0, 20, color='red', label='R')
            axs[1].vlines(2 * mean_area / mean_circ * space_scale * 1e6, 0, 20, color='orange', label='Rcr')
            axs[0].set_xlabel('Radius of curvature [um]')
            axs[1].set_xlabel('Radius [um]')
            for ax in axs:
                ax.set_ylabel('frequency')
                ax.set_title(f'frame {round(float(frame.file_name.split("s")[0]))}s')
                ax.set_xlim([0, np.max([rk, r])])
                ax.set_ylim([0, 5])
                ax.legend()
        plt.savefig(os.path.join(output_img_dir, 'R distributions.png'))
        plt.show()

def A_distribution(frame_list):
    """Calculate the distribution of areas. Only use this function when you select A FEW frames for analysis,
    because it will plot distribution of max 10 frames.
    """
    # Only plot area distributions if 10 frames are selected.
    if len(frame_list) < 10:
        fig, axs = plt.subplots(1, len(frame_list))
        fig.suptitle('Area distribution')

        for frame, ax in zip(frame_list, axs):
            areas = np.array([crystal.area for crystal in frame.crystalobjects])
            ax.hist(areas * space_scale**2 * 1e12, bins=20)
            ax.set_xlabel('Area [um^2]')
            ax.set_ylabel('frequency')
            ax.set_title(f'frame {round(float(frame.file_name.split("s")[0]))}s')

        plt.savefig(os.path.join(output_img_dir, 'A distributions.png'))

# --------------------------

def export_quantities(times, Q, N, A, r_k, r_k3, mean_r3_A_div_l, l, ROI_area, mr_k):
    """Export the calculated quantities to csv."""
    data = {
        'times': times,
        'Q': Q,
        'N': N,
        'A': A,
        'r_k': r_k,
        'r_k3': r_k3,
        'mean_r3_A_div_l': mean_r3_A_div_l,
        'l': l,
        'ROI_area': ROI_area,
        'mr_k': mr_k
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.path.basename(IMAGE_OUTPUT_FOLDER_NAME) + '.csv'), index_label='index')


if __name__ == "__main__":
    start_time = time.time()

    # Use this variable to convert pixels to meters.
    space_scale = 86.7*10**(-9) #m

    # Frame directory selection.
    IMAGE_FORMAT = '.png'
    root = Tk() # File dialog
    INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    root.destroy()

    # Directory creation.
    try:
        os.mkdir(os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'analysis')))
        os.mkdir(os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'csv')))
    except FileExistsError:
        pass
    IMAGE_OUTPUT_FOLDER_NAME = os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'analysis', os.path.basename(INPUT_FOLDER_NAME)))
    CSV_EXPORT_FOLDER = os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'csv', os.path.basename(INPUT_FOLDER_NAME)))

    imgs_dir = set_and_check_folder(INPUT_FOLDER_NAME)
    output_img_dir = set_and_check_folder(IMAGE_OUTPUT_FOLDER_NAME, True)
    csv_export_dir = set_and_check_folder(CSV_EXPORT_FOLDER, True)
    img_files, file_count = get_img_files_ordered(imgs_dir)
    open_file(IMAGE_OUTPUT_FOLDER_NAME)

    # Thresholding
    # 599 and 0 are 'okay' for 0uM 10% for example.
    # See thesis S. de Jong for possible good values for different sucrose and IBP concentrations.
    threshold_blocksize = int(input('Threshold blocksize: '))
    threshold_constant = int(input('Threshold subtraction constant: '))

    # If the contours should be plotted on the frames, select True.
    PLOT_FRAME_CONTOURS = True

    # Write thresholding settings to file.
    with open(os.path.join(IMAGE_OUTPUT_FOLDER_NAME, 'settings.txt'), 'w') as settings_file:
        settings_file.write('Adaptive thresholding\n')
        settings_file.write(f'\tblockSize = {threshold_blocksize}\n')
        settings_file.write(f'\tconstant = {threshold_constant}\n\n')

    # Create frame list with contours.
    frame_list = create_frame_list(img_files, file_count, imgs_dir,
        output_img_dir, IMAGE_FORMAT, PLOT_FRAME_CONTOURS)
    
    # Write ROI settings to file settings file.
    with open(os.path.join(IMAGE_OUTPUT_FOLDER_NAME, 'settings.txt'), 'a') as settings_file:
        settings_file.write(f'ROI crop = {FrameImg.ROI_crop}')

    # Log time it took to process images.
    img_processing_time = time.time() - start_time 
    
    # Extract data from the frame list.
    times, Q = calculate_volume_fraction(frame_list) # Timestamps and ice volume fractions per frame.
    N = amount_of_crystals_per_ROIarea(frame_list) # Number of crystals per frame.
    A = avg_crystal_area(frame_list) # Total area per frame.
    mr_k = median_mean_radius(frame_list) # Median of the mean radius of curvature.
    r_k = mean_radius_of_curvature(frame_list) # Mean radius of curvature.
    r_k3 = mean_radius_of_curvature3(frame_list) # Mean radius cubed of curvature.
    l = circumference(frame_list) # The circumferences.
    mean_r3_A_div_l = r3_by_A_div_l(frame_list) # Mean radius.

    # Calculate and plot the distribution of area and radius.
    # A_distribution(frame_list)
    # r_distribution(frame_list)

    # Calculate area of ROI.
    ROI_area = frame_list[0].img_height * frame_list[0].img_width

    # Export all extracted quantities to a csv file.
    export_quantities(times, Q, N, A, r_k, r_k3, mean_r3_A_div_l, l, ROI_area, mr_k)

    # Fit parameters to the extracted data and plot the extracted data with their fits.
    try:
        import fit_data
        df_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.path.basename(IMAGE_OUTPUT_FOLDER_NAME) + '.csv')
        df = pd.read_csv(df_path, index_col='index').dropna() # Drop rows which have at least one NaN.

        # Correct for faster playback speed. Only do this if you know you need this!
        df.times = df.times * 13 / 21.52 
        df['time_corrected'] = True # Mark current csv file as 'corrected for time'

        # Perform fitting.
        print("Fitting parameters.")
        df = fit_data.fitting(df, df_path)
        fit_data.plot(df, df_path)

        try:
            import plot_Q
            print("Plotting Qs.")
            Q_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
            df_Q = plot_Q.extract_Q(Q_path)
            plot_Q.plot_Q(df_Q, Q_path)
        except FileNotFoundError:
            print("Cannot plot Q's, because plot_Q.py is missing.")
        
        try:
            import plot_A
            print("Plotting As.")
            A_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
            df_A = plot_A.extract_A(A_path)
            plot_A.plot_A(df_A, A_path)
        except FileNotFoundError:
            print("Cannot plot A's, because plot_A.py is missing.")

        try:
            # import plot_critical.py
            # print("Plotting r^3.")
            # r3_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
            # df_r3 = plot_critical.extract_r3(r3_path)
            # plot_critical.plot_r3(df_r3, r3_path)

            import plot_r3
            r3_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
            df_r3 = plot_r3.extract_r3(r3_path)
            plot_r3.plot_r3(df_r3, r3_path)
        except FileNotFoundError:
            # print("Cannot plot r^3, because plot_critical.py is missing.")
            print("Cannot plot r^3, because plot_r3.py is missing.")

        try:
            import plot_k
            print("Plotting kd.")
            k_path = os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.pardir)
            df_k = plot_k.extract_k(k_path)
            plot_k.plot_k(df_k, k_path)
        except FileNotFoundError:
            print("Cannot plot k, because plot_k.py is missing.")
    except FileNotFoundError:
        print("Cannot fit, because fit_data.py is missing.")

    print('######################################################')
    print(f'{os.path.basename(INPUT_FOLDER_NAME)} done.')
    print(f'img processing time: {img_processing_time} ')
    print("Total runtime --- %s seconds ---" % (time.time() - start_time))
    print('######################################################')


