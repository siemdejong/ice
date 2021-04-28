import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
from scipy import stats
import trackpy as tp
# import pims
import skimage.io as io
from math import sqrt
from pprint import pprint
import itertools
import sys
import pickle
import math
import json
from tkinter import filedialog
from tkinter import *
import csv

class FrameImg:
    ''' Add Description '''
    # ROI_crop = [xleft, yleft, dx, dy]

    crop_boo = True

    if 'ROI_crop' not in locals():
        ROI_crop = None
    else:
        ylow = ROI_crop[1]
        yup = ROI_crop[1] + FrameImg.ROI_crop[3]
        xleft = ROI_crop[0]
        xright = ROI_crop[0] + FrameImg.ROI_crop[2]

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
        if area_ratio > 5: # If largest area is 10 times that of the second largest area
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
        img_treshold = self.tresholding_img(img_denoised)
        # self.plot_stages(img, img_denoised, img_treshold)
        # Add in possible plot stages method here before the images go out of scope.
        return img, img_treshold

    def plot_stages(self, img, img_denoised, img_treshold):
        images = [img, img_denoised, img_treshold]
        for i in range(len(images)):
            plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
            plt.xticks([]), plt.yticks([])
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

    def tresholding_img(self,img_denoised):
        ''' Treshold image. 
            Docs: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 '''
        return cv2.adaptiveThreshold(src=img_denoised, maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY, blockSize=199, C=1)

    def get_img_contours(self, img_treshold):
        ''' Retreive image contours. ## NEED TO WORK WITH RETR_CCOMP instead of RETR_EXTERNAL TO GET HIERACHY FOR DEALIG WITH INCEPTIONS
            Docs: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a'''
        self.contours, self.hierarchy = cv2.findContours(img_treshold,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def process_contours(self):
        ''' First creates two empty lists to store the crystals and the reminaining/other objects. Next, 
            loops through all the retreives contours: If the len(contour), which is the amount of coordinate
            points is large enough, it creates a CrystalObject class instance. Then, if the 'contour lenght'
            is greater than 2, it adds the CrystalObject to the list. If the conditions are not met, the object
            is stored in the other objects list.   '''
        self.crystalobjects = []
        self.otherobjects = []
        num_contours = len(self.contours)
        print(f'Number of contours found: {num_contours}')
        for i, contour in enumerate(self.contours):
            if len(contour) > 30: # Checks the amount of coordinate points in the contour <- NEEDS TO BE REWORKED 
                print(f'Processing img contours {i}/{num_contours}', end = '\r')
                obj = CrystalObject(contour, contour_num= i, frame_num = self.frame_num)
                if obj.x_center == 0 and obj.y_center == 0:
                    self.otherobjects.append(obj)
                else:
                    self.crystalobjects.append(obj) # <- SEEMS TO NEVER BE THE CASE.........

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
            plt.plot(crystal.s_contours[...,0], crystal.s_contours[...,1],'k', linewidth=1)
            # plt.scatter(crystal.center_arr[0],crystal.center_arr[1], s=2)
        fig.suptitle(frame_img_name, fontsize = 8)
        if save_image:
            fig.savefig(f'{file_name}')
        plt.close()


class CrystalObject:
    ''' Add Description '''
    def __init__(self, contour_raw, contour_num, frame_num):
        self.contour_raw = contour_raw
        self.frame_num = frame_num
        self.contour_num = contour_num
        self.contour_length = len(self.contour_raw)

        self.moments = cv2.moments(contour_raw)
        self.get_center_point()
        if self.contour_length > 2:
            self.get_area()
            self.get_length()
            self.set_contours_dataframe()
            # print(self.s_contours_df.x.max())

    def get_center_point(self):
        ''' Using the contour moments, retreives the center x and y coordinates,
            and an array of said coordinates. '''
        if self.moments['m00'] != 0:
            self.x_center = int(self.moments['m10'] / self.moments['m00'])
            self.y_center = int(self.moments['m01'] / self.moments['m00'])
        else:
            self.x_center = 0
            self.y_center = 0
        self.center_arr = np.array([self.x_center, self.y_center])

    def get_area(self):
        ''' Sets the area of the contour '''
        self.area = cv2.contourArea(self.contour_raw)

    def get_length(self):
        ''' Sets the 'length' of the contour ??? '''
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
        ''' Function to prepare the crystal dataframe. Loads in the contours array, padds them (reason?), and then
            smooths the coordinates using rolling mean. Next, calls the calculate_cruvature function in order to 
            retrieve the curvature and its mean. Adds the contour number, frame, area, and lenght of the contour, 
            and saves the resulting df. Finally, it creates an additional df with the rounded coordinates. ''' 
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
        self.s_contours = self.s_contours_df.drop(['curvature','mean(curvature)', 'sx', 'sy', 'contour_num', 'frame', 'area', 'length'], axis=1).to_numpy()
        


class CrystalRecog:
    def __init__(self, c_obj):
        """ Start off by creating the empty lists (which will be appended with the attributes
        for the specific crystal for each FrameImg. Then, add the attributes for the First 
        Frame to start off with. """
        self.s_contours_dfs = []
        self.raw_contours = []
        self.s_contours = []
        self.lengths = []
        self.areas = []
        self.center_arrays = []
        self.mean_curvatures = []
        self.c_count = 0
        self.frames_used = []
        self.count_num = c_obj.contour_num
        self.add_crystalobject(c_obj)
        
    def add_crystalobject(self, c_obj):
        """ Add cryal attributes to the respective class intances. """
        self.s_contours_dfs.append(c_obj.s_contours_df)
        self.raw_contours.append(c_obj.contour_raw)
        self.s_contours.append(c_obj.s_contours)
        self.lengths.append(c_obj.length)
        self.areas.append(c_obj.area)
        self.center_arrays.append(c_obj.center_arr)
        self.mean_curvatures.append(c_obj.s_contours_df['mean(curvature)'].min())
        self.frames_used.append(str(c_obj.frame_num))
        self.c_count += 1

    def retreive_outer_bounds(self, padding_margin):
        max_y = self.s_contours_dfs[len(self.s_contours_dfs)-1].y.max()
        min_y = self.s_contours_dfs[len(self.s_contours_dfs)-1].y.min()
        y_padding = (max_y - min_y)*padding_margin
        max_y += y_padding
        min_y -= min_y - y_padding
        max_x = self.s_contours_dfs[len(self.s_contours_dfs)-1].x.max()
        min_x = self.s_contours_dfs[len(self.s_contours_dfs)-1].x.min()
        x_padding = (max_x - min_x)*padding_margin
        max_x += x_padding
        min_x -= x_padding
        return min_y, max_y, min_x, max_x

    # def add_dubble_crystalobject(self, c_obj1, c_obj2):
    #     s_contours_df = pd.concat([c_obj1.s_contours_df,c_obj2.s_contours_df])
    #     self.s_contours_dfs.append(s_contours_df)
    #     raw_contours = np.concatenate((c_obj1.contour_raw, c_obj2.contour_raw))
    #     self.raw_contours.append(raw_contours)

    #     area = (c_obj1.area + c_obj2.area) * 0.5
    #     self.areas.append(area)
    #     x_center = (c_obj1.center_arr[0] + c_obj2.center_arr[0])*0.5
    #     y_center = (c_obj1.center_arr[1] + c_obj2.center_arr[1])*0.5
    #     center_arr = np.array([x_center, y_center])
    #     print(f'{c_obj1.center_arr} + {c_obj2.center_arr} = {center_arr}')
    #     self.center_arrays.append(center_arr)

    def plot_contours_across_frames(self, file_count, output_img_dir):
        """ Add description """
        fig = plt.figure()
        fig.tight_layout()
        gs1 = fig.add_gridspec(nrows=3, ncols=2)
        fig.suptitle(t=f'#{self.count_num}; FU{self.c_count}/{file_count}', fontsize=12, va='top')
        fig_ax1 = fig.add_subplot(gs1[:-1, :])
        fig_ax1.title.set_text('Contours')
        fig_ax1.title.set_fontsize(12)
        for contour in self.s_contours:
            fig_ax1.plot(contour[...,0], contour[...,1])
        fig_ax1.invert_yaxis()

        fig_ax2 = fig.add_subplot(gs1[-1, :-1])
        fig_ax2.title.set_text('Area')
        fig_ax2.plot(self.areas)
        fig_ax2.title.set_fontsize(10)

        fig_ax3 = fig.add_subplot(gs1[-1, -1])
        fig_ax3.title.set_text('Mean Curvature')
        fig_ax3.plot(self.mean_curvatures)
        fig_ax3.title.set_fontsize(10)

        frames_used = ','.join(self.frames_used)
        fig.text(0.02, 0.02, 'FU: ' + frames_used, color='grey',fontsize=4)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
            wspace=0.3, hspace=0.5)
        fig.savefig(os.path.join(output_img_dir, f'newtest_img{self.count_num}.png'))
        plt.close()



# MAKE METHOD OF CrystalTracking?
def euqli_dist(p, q, squared=False):
    # Calculates the euclidean distance, the "ordinary" distance between two
    # points. The standard Euclidean distance can be squared in order to place
    # progressively greater weight on objects that are farther apart. This
    # frequently used in optimization problems in which distances only have
    # to be compared.
    if squared:
        return ((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2)
    else:
        return sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2))
# MAKE METHOD OF CrystalTracking?
def closest(cur_pos, positions):
    # FIND THE SOURCE FOR THIS CODE ON STACKOVERFLOW!!
    low_dist = float('inf')
    closest_pos = None
    index = None
    for index, pos in enumerate(positions):
        dist = euqli_dist(cur_pos,pos)
        if dist < low_dist:
            low_dist = dist
            closest_pos = pos
            ret_index = index
    else: # This breaks the tracking panels, but allows for calculation of volume fraction over time.
        ret_index = None
    return closest_pos, ret_index, low_dist

def get_img_files_ordered(dir_i):
    """ Function returns a list of all input frames, ordered by their frame number, 
    together with a count of the total number of frames.  """
    img_files = []
    for file in os.listdir(dir_i):
        try:
            file_i = {
            'filename' : file,
            # 'file_num' : int(file.split('frame')[1].split('.')[0])
            'file_num': int(file.split('.')[0]) # The data acquired by Siem after using extract_frames.py by Agata uses another file naming scheme. (<time>s.png instead of frame<frame>.png)
            }
            img_files.append(file_i )
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

def setup_detection_box(target_crys, padding_margin):
    """ Retreive the upper and lower x and y coordinates, in which a crystal's contour is contained, with 
    and additional padding margin to be able to detect possible fusions/splits over time. """
    pass

def plot_ivf(ice_volume_fraction_list, times):
    # Plot the volume fraction in time.
    fig_volume_fraction = plt.figure()
    fig_volume_fraction.tight_layout()
    gs_vol_frac = fig_volume_fraction.add_gridspec(nrows=1, ncols=1)
    fig_volume_fraction_ax = fig_volume_fraction.add_subplot(gs_vol_frac[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_volume_fraction_ax.scatter(times, ice_volume_fraction_list)
    fig_volume_fraction_ax.set_ylabel('Ice volume fraction')
    fig_volume_fraction_ax.set_xlabel('Time [s]')
    fig_volume_fraction_ax.set_xticks(np.linspace(round(times[0]), round(times[-1]),num=2)) # Set this for appropriate axis ticks.
    fig_volume_fraction.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_volume_fraction.savefig(os.path.join(output_img_dir, 'volume_fraction.png'))
    plt.close()

def calculate_volume_fraction(frame_list):
    """
    Calculate the volume fraction Q = v_ice / (v_ice + v_liq) of the ROI.
    Dumps volume fractions per time to a json file, for later reuse so not everything has to be calculated again.
    """
    print("Calculating volume fractions.")
    ice_volume_fraction_list = []
    times = []
    for frame in frame_list:
        tot_area = 0
        for crystal in frame.crystalobjects:
            tot_area += crystal.area
        ROI_area = frame.img_width * frame.img_height
        ice_volume_fraction = tot_area / ROI_area # Assuming tot_area and ROI_area have the same units.
        ice_volume_fraction_list.append(ice_volume_fraction)
        time = float(frame.file_name.split('s')[0])
        times.append(time)
    
    plot_ivf(ice_volume_fraction_list, times)

    return times, ice_volume_fraction_list


def plot_crystal_numbers(crystal_numbers, times):
    "Plot the amount of crystals in time."
    fig_crystal_numbers = plt.figure()
    fig_crystal_numbers.tight_layout()
    gs_vol_frac = fig_crystal_numbers.add_gridspec(nrows=1, ncols=1)
    fig_crystal_numbers_ax = fig_crystal_numbers.add_subplot(gs_vol_frac[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_crystal_numbers_ax.scatter(times, crystal_numbers)
    fig_crystal_numbers_ax.set_ylabel('Number of crystals')
    fig_crystal_numbers_ax.set_xlabel('Time [s]')
    fig_crystal_numbers.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_crystal_numbers.savefig(os.path.join(output_img_dir, 'number_of_crystals.png'))
    plt.close()


def amount_of_crystals(frame_list):
    "Calculate the amount of crystals in the ROI in time."
    print("Plotting number of crystals")
    crystal_numbers = []
    times = []
    for frame in frame_list:
        number_of_crystals_in_frame = len(frame.crystalobjects)
        crystal_numbers.append(number_of_crystals_in_frame)
        time = float(frame.file_name.split('s')[0])
        times.append(time)
    
    plot_crystal_numbers(crystal_numbers, times)

    return crystal_numbers

def plot_avg_crystal_area(avg_crystal_areas, times):
    "Plot the amount of crystals in time."
    space_scale = 86.7*10**(-9) #m

    fig_avg_crystal_areas = plt.figure()
    fig_avg_crystal_areas.tight_layout()
    gs_vol_frac = fig_avg_crystal_areas.add_gridspec(nrows=1, ncols=1)
    fig_avg_crystal_areas_ax = fig_avg_crystal_areas.add_subplot(gs_vol_frac[0, 0])
    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    fig_avg_crystal_areas_ax.scatter(times, np.asarray(avg_crystal_areas)*space_scale**2*1e12)
    fig_avg_crystal_areas_ax.set_ylabel(r'Mean crystal area [$\mu$m$^2$]')
    fig_avg_crystal_areas_ax.set_xlabel('Time [s]')
    fig_avg_crystal_areas.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
    fig_avg_crystal_areas.savefig(os.path.join(output_img_dir, 'average_crystal_area.png'))
    plt.close()

def calculate_avg_crystal_area(frame):
    "Calculate the average crystal area inside a frame."
    avg = np.mean([crystal.area for crystal in frame.crystalobjects])

    return avg

def avg_crystal_area(frame_list):
    "Plot the average crystal area in time."
    print("Calculating average areas per time.")

    avg_area_list = list(map(calculate_avg_crystal_area, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_avg_crystal_area(avg_area_list, times)

    return avg_area_list

def plot_mean_radius(mean_radius_list, times):
    "Plot the amount of crystals in time."
    space_scale = 86.7*10**(-9) #m

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
    fig_mean_radius.savefig(os.path.join(output_img_dir, 'mean_radius.png'))
    plt.close()

def calculate_mean_radius(frame):
    "Calculate the mean radius from the mean curvature by taking its reciprocal."
    mean_curv = np.mean([crystal.mean_curvature for crystal in frame.crystalobjects])
    mean_radius = 1 / mean_curv
    return mean_radius

def mean_radius(frame_list):
    "Plot the mean radius of the crystals in the ROI."
    print("Calculating mean radius of curvature per time")

    mean_radius_list = list(map(calculate_mean_radius, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_mean_radius(mean_radius_list, times)

    return mean_radius_list

def plot_mean_radius3(mean_radius3_list, times):
    "Plot the amount of crystals in time."
    space_scale = 86.7*10**(-9) #m

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
    fig_mean_radius3.savefig(os.path.join(output_img_dir, 'mean_radius3.png'))
    plt.close()

def calculate_mean_radius3(frame):
    "Calculate the mean radius cubed from the mean curvature by taking its reciprocal and cubing the outcome."
    mean_curv = np.mean([crystal.mean_curvature for crystal in frame.crystalobjects])
    mean_radius = 1 / mean_curv
    return mean_radius**3

def mean_radius3(frame_list):
    "Plot the mean radius cubed of the crystals in the ROI."
    print("Calculating mean radius cubed per time")

    mean_radius3_list = list(map(calculate_mean_radius3, frame_list))
    times = [float(frame.file_name.split('s')[0]) for frame in frame_list]
    
    plot_mean_radius3(mean_radius3_list, times)

    return mean_radius3_list


def export_quantities(times, Q, N, A, r, r3):
    """Export the calculated quantities to csv."""
    data = {
        'times': times,
        'Q': Q,
        'N': N,
        'A': A,
        'r': r,
        'r3': r3
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(IMAGE_OUTPUT_FOLDER_NAME, os.path.basename(IMAGE_OUTPUT_FOLDER_NAME) + '.csv'), )


if __name__ == "__main__":
    start_time = time.time()
    # Constants:
    IMAGE_FORMAT = '.png'
    root = Tk() # File dialog
    INPUT_FOLDER_NAME =  filedialog.askdirectory(title = "Select directory")
    root.destroy()

    try:
        os.mkdir(os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'analysis')))
        os.mkdir(os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'csv')))
    except FileExistsError:
        pass
    IMAGE_OUTPUT_FOLDER_NAME = os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'analysis', os.path.basename(INPUT_FOLDER_NAME)))
    CSV_EXPORT_FOLDER = os.path.join(INPUT_FOLDER_NAME, os.path.join(os.pardir, os.pardir, 'csv', os.path.basename(INPUT_FOLDER_NAME)))

    # Agata says these don't matter for me, but they do, right? Because I rely on the curvatures calculated during tracking.
    MAX_CENTER_DISTANCE = 0
    AREA_PCT = 0
    CENTER_PCT = 0
    MIN_PLOT_FRAMES = math.inf
    PLOT_FRAME_CONTOURS = True

    imgs_dir = set_and_check_folder(INPUT_FOLDER_NAME)
    output_img_dir = set_and_check_folder(IMAGE_OUTPUT_FOLDER_NAME, True)
    csv_export_dir = set_and_check_folder(CSV_EXPORT_FOLDER, True)
    img_files, file_count = get_img_files_ordered(imgs_dir)
    frame_list = create_frame_list(img_files, file_count, imgs_dir,
        output_img_dir, IMAGE_FORMAT, PLOT_FRAME_CONTOURS)

    img_processing_time = time.time() - start_time # Log time it took to process images.
    
    # Create initial crystals
    crystal_tracking_list = []
    for obj in frame_list[0].crystalobjects:
        crystal_tracking_list.append(CrystalRecog(obj))
    print('Frame # 1:')
    print(f'Used count: {len(crystal_tracking_list)}')

    # Start from 1 here, because frame 0 / the first frame already done above
    for i in range(1,len(frame_list)):
        print(f'Frame # {i+1}:')
        c_central_list = frame_list[i].crystal_centers
        c_crystal_areas_list = frame_list[i].crystal_areas
        pre_frame_center_coord_count = len(c_central_list)
        for target_crys in crystal_tracking_list:
            # find the coordinates, index of said coordinates, and distance to last center point
            clostest_coord, index_closest, distance = closest(target_crys.center_arrays[len(target_crys.center_arrays) -1], c_central_list)
            if distance < MAX_CENTER_DISTANCE:
                # Find Crystal object corresponding to the central coord
                for crys in frame_list[i].crystalobjects:
                    if (crys.center_arr == clostest_coord).all():
                        if crys.area*(1-AREA_PCT) <= target_crys.areas[len(target_crys.areas) -1] <= crys.area*(1+AREA_PCT):
                            target_crys.add_crystalobject(crys)
                            c_central_list.pop(index_closest)
                            c_crystal_areas_list.remove(crys.area)

            # else:
            #     print(f'Attempting to match {target_crys.count_num}')
            #     target_area = target_crys.areas[len(target_crys.areas) -1]
            #     for areas in itertools.combinations(c_crystal_areas_list, 2):
            #         if  target_area*(1-AREA_PCT) <= sum(areas) <= target_area*(1+AREA_PCT):
            #             print(f'found {sum(areas)} being equal to {target_area} ???')
            #             index_list = [c_crystal_areas_list.index(area) for area in areas]
            #             crys_list = []
            #             for crys in frame_list[i].crystalobjects:
            #                 for ind in index_list:
            #                     if crys.area == c_crystal_areas_list[ind]:
            #                         crys_list.append(crys)
            #             obj_center_contained = False
            #             print('**********')
            #             for crys in crys_list:
            #                 print(f'---------{len(crys_list)}')
            #                 if target_crys.min_y*(1-CENTER_PCT) <= crys.center_arr[1] <= target_crys.max_y*(1+CENTER_PCT) and target_crys.min_x*(1-CENTER_PCT) <= crys.center_arr[0] <= target_crys.max_x*(1+CENTER_PCT):
            #                     print(f'{target_crys.min_y*(1-CENTER_PCT)} <= {crys.center_arr[1]} <= {target_crys.max_y*(1+CENTER_PCT)}')
            #                     print(f'{target_crys.min_x*(1-CENTER_PCT)} <= {crys.center_arr[0]} <= {target_crys.max_x*(1+CENTER_PCT)}')
            #                     obj_center_contained = True
            #                 else:
            #                     obj_center_contained = False
            #                     break
            #             if obj_center_contained == True:
            #                 print('YAAASS QUEEN')
            #                 if len(crys_list) == 2:
            #                     print('doubleeee')
            #                     target_crys.add_dubble_crystalobject(crys_list[0], crys_list[1])
            #                 if len(crys_list) == 3:
            #                     print('T-T-T-T-tripple comboooooooo! Figure this shit out')
                        # From the crystals in crys_list, find max_x, max_y, etc.
                        # Check if these values are within min/max y and x of CrystalRecog

        # For each Crystal:
            # Set up detection box (Upper and lower x and y coords + some margin)
            # Find all center points in detection box (Just center points will work?)
                # Find/identify corresponding crystals
                # Check if all contour points of identified crystals are in the detection box
            # Compare area of crystalRecog to previous one.
                # Evaluate if adding other identified crystals would be closer to previous area, together with checking if the
                    #combined center point would be closer to the previous center point of the Crystal.
                        # If yes, loop backwards through frames, and add the additional crystal attributes to crystal Recog
                        # Then continue in loop as normal


        post_frame_center_coord_count = len(c_central_list)
        print('------------------------------------------------------')
        print(f'Frame # {i + 1}:')
        print(f'C coords went from {pre_frame_center_coord_count} to {post_frame_center_coord_count}  ')
        print(f'Used count: {pre_frame_center_coord_count - post_frame_center_coord_count }')
    crystal_linking_time = (time.time() - start_time) - img_processing_time # Log time it took to link crystals


    crystal_tracking_count = len(crystal_tracking_list)
    # for i,crystallcoll in enumerate(crystal_tracking_list):
    #     if crystallcoll.count_num == 79:
    #         print(crystallcoll.areas)
    #     print(f'Plotting Crystal {i}/{crystal_tracking_count}', end = '\r')
    #     crystallcoll.plot_contours_across_frames(file_count, output_img_dir)
    for i,b in enumerate(crystal_tracking_list):
        print(f'Plotting Crystal {i}/{crystal_tracking_count}', end = '\r')
        if b.c_count > MIN_PLOT_FRAMES:

            space_scale = 86.7*10**(-9) #m
            gamma_0 = 29.8 #mJ/m^2
            d_tolman = 0.24*10**(-9) #m
            solution_thickness = 2*10**(-6)

            fig = plt.figure()
            fig.tight_layout()
            gs1 = fig.add_gridspec(nrows=2, ncols=2)
            fig_ax1 = fig.add_subplot(gs1[0, 0])
            fig_ax1.title.set_text('Contours')
        
            for contour in b.s_contours: 
                fig_ax1.plot(contour[...,0], contour[...,1])
            fig_ax1.invert_yaxis()
            fig_ax1.title.set_fontsize(12)
            fig.suptitle(t=f'#{b.count_num}; FU{b.c_count}/{file_count}', fontsize=12, va='top')

            fig_ax2 = fig.add_subplot(gs1[0, 1])
            fig_ax2.title.set_text('Area')
            fig_ax2.plot(np.asarray(b.areas)*space_scale*space_scale*10**12)
            fig_ax2.set_ylabel('area [um^2]')
            fig_ax2.set_xlabel('time')
            fig_ax2.title.set_fontsize(10)

            fig_ax3 = fig.add_subplot(gs1[1, 0])
            fig_ax3.title.set_text('Mean curvatures')
            fig_ax3.plot(np.asarray(b.mean_curvatures)/(space_scale*10**6))
            fig_ax3.set_ylabel('mean curvature [1/um]')
            fig_ax3.set_xlabel('time')
            fig_ax3.title.set_fontsize(10)

           

            fig_ax4 = fig.add_subplot(gs1[1, 1])
            fig_ax4.title.set_text('Gibbs surface energy')
            G = 2* gamma_0*np.asarray(b.areas)*space_scale*space_scale + (gamma_0 * np.asarray(b.lengths) * space_scale* solution_thickness *(1-((np.asarray(b.mean_curvatures)/space_scale) *2*d_tolman)))
            fig_ax4.plot(G)
            fig_ax4.set_ylabel('G_total [mJ]')
            fig_ax4.set_xlabel('time')
            fig_ax4.title.set_fontsize(10)

            frames_used = ','.join(b.frames_used)
            fig.text(0.02, 0.02, 'FU: ' + frames_used, color='grey',fontsize=4)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
            fig.savefig(os.path.join(output_img_dir, f'newtest_img{b.count_num}.png'))
            plt.close()

    times, Q = calculate_volume_fraction(frame_list)

    N = amount_of_crystals(frame_list)

    A = avg_crystal_area(frame_list)

    r = mean_radius(frame_list)

    r3 = mean_radius3(frame_list)

    export_quantities(times, Q, N, A, r, r3)

    for i, crystallcoll in enumerate(crystal_tracking_list):
        df_i = pd.concat(crystallcoll.s_contours_dfs)
        csv_file_name = f'{crystallcoll.count_num}.csv'
        csv_export_dir_i = os.path.join(csv_export_dir, csv_file_name)
        df_i.to_csv(csv_export_dir_i)
        



    print('######################################################')
    print(f'{os.path.basename(INPUT_FOLDER_NAME)} done.')
    print(f'img processing time: {img_processing_time} ')
    print(f'Crystal linking time : {crystal_linking_time}')
    print("Total runtime --- %s seconds ---" % (time.time() - start_time)) # To see how long program
    print('######################################################')


