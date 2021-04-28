import json
import matplotlib.pyplot as plt
import numpy as np

def graph_multiple_volume_fractions(paths):
    # Plot the volume fraction in time.
    fig_volume_fraction = plt.figure()
    fig_volume_fraction.tight_layout()
    gs_vol_frac = fig_volume_fraction.add_gridspec(nrows=1, ncols=1)
    fig_volume_fraction_ax = fig_volume_fraction.add_subplot(gs_vol_frac[0, 0])

    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    for path, measurement in zip(paths, [0, 1, 4, 10]):
        with open(path, 'r') as openfile:
            json_object = json.load(openfile)

        ice_volume_fraction_list_old = list(json_object.keys())
        ice_volume_fraction_list = []
        for item in ice_volume_fraction_list_old:
            ice_volume_fraction_list.append(float(item))
        times = list(json_object.values())

        fig_volume_fraction_ax.scatter(times, ice_volume_fraction_list, s=0.5, label=measurement)
        fig_volume_fraction_ax.set_ylabel('Ice volume fraction')
        fig_volume_fraction_ax.set_xlabel('Time [s]')
        # fig_volume_fraction_ax.set_xticks(np.linspace(round(times[0]), round(times[-1]),num=2)) # Set this for appropriate axis ticks.
        fig_volume_fraction_ax.set_yticks(np.linspace(0, 1, num=11))
        fig_volume_fraction.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                    wspace=0.3, hspace=0.5)

    fig_volume_fraction_ax.legend()
    fig_volume_fraction.savefig('graph_ivfs.png')

def graph_multiple_areas(paths):
    space_scale = 86.7*10**(-9) #m

    # Plot the volume fraction in time.
    fig_area = plt.figure()
    fig_area.tight_layout()
    gs_vol_frac = fig_area.add_gridspec(nrows=1, ncols=1)
    fig_area_ax = fig_area.add_subplot(gs_vol_frac[0, 0])

    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    for path, measurement in zip(paths, [0, 1, 4, 10]):
        with open(path, 'r') as openfile:
            json_object = json.load(openfile)

        area_list_old = list(json_object.keys())
        area_list = []
        for item in area_list_old:
            area_list.append(float(item))
        times = list(json_object.values())

        fig_area_ax.scatter(times, np.asarray(area_list)*space_scale**2*1e12, s=0.5, label=measurement)
        fig_area_ax.set_ylabel(r'Mean area [$\mu$m$^2$]')
        fig_area_ax.set_xlabel('Time [s]')
        # fig_area_ax.set_xticks(np.linspace(round(times[0]), round(times[-1]),num=2)) # Set this for appropriate axis ticks.
        # fig_area_ax.set_yticks()
        fig_area.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                    wspace=0.3, hspace=0.5)

    fig_area_ax.legend()
    fig_area.savefig('graph_area.png')

def graph_multiple_crystal_amounts(paths):

    # Plot the volume fraction in time.
    fig_amount = plt.figure()
    fig_amount.tight_layout()
    gs_amount = fig_amount.add_gridspec(nrows=1, ncols=1)
    fig_amount_ax = fig_amount.add_subplot(gs_amount[0, 0])

    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    for path, measurement in zip(paths, [0, 1, 4, 10]):
        with open(path, 'r') as openfile:
            json_object = json.load(openfile)

        amount_list_old = list(json_object.keys())
        amount_list = []
        for item in amount_list_old:
            amount_list.append(float(item))
        times = list(json_object.values())

        fig_amount_ax.scatter(times, amount_list, s=0.5, label=measurement)
        fig_amount_ax.set_ylabel(r'Number of crystals')
        fig_amount_ax.set_xlabel('Time [s]')
        # fig_amount_ax.set_xticks(np.linspace(round(times[0]), round(times[-1]),num=2)) # Set this for appropriate axis ticks.
        # fig_area_ax.set_yticks()
        fig_amount.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                    wspace=0.3, hspace=0.5)

    fig_amount_ax.legend()
    fig_amount.savefig('graph_crystal_amount.png')

def graph_multiple_mean_radii(paths):
    space_scale = 86.7*10**(-9) #m

    # Plot the volume fraction in time.
    fig_mean_radii = plt.figure()
    fig_mean_radii.tight_layout()
    gs_mean_radii = fig_mean_radii.add_gridspec(nrows=1, ncols=1)
    fig_mean_radii_ax = fig_mean_radii.add_subplot(gs_mean_radii[0, 0])

    # fig_volume_fraction_ax.title.set_text(frame_list[0].imgs_dir)
    for path, measurement in zip(paths, [0, 1, 4, 10]):
        with open(path, 'r') as openfile:
            json_object = json.load(openfile)

        mean_radii_list_old = list(json_object.keys())
        mean_radii_list = []
        for item in mean_radii_list_old:
            mean_radii_list.append(float(item))
        times = list(json_object.values())

        fig_mean_radii_ax.scatter(times, np.asarray(mean_radii_list)*space_scale*1e6, s=0.5, label=measurement)
        fig_mean_radii_ax.set_ylabel(r'Mean radius of curvature [$\mu$m]')
        fig_mean_radii_ax.set_xlabel('Time [s]')
        # fig_amount_ax.set_xticks(np.linspace(round(times[0]), round(times[-1]),num=2)) # Set this for appropriate axis ticks.
        # fig_area_ax.set_yticks()
        fig_mean_radii.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                    wspace=0.3, hspace=0.5)

    fig_mean_radii_ax.legend()
    fig_mean_radii.savefig('graph_mean_radii.png')

paths_ivf = [
    # WT 20%
    # "D:/Siem/07042021/0uM_20_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/08042021/1uM_WT_20%_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/12042021/4uM_WT_20%_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/09042021/10uM_WT_20%_0_OUTPUT/volume_fractions.json",  

    # WT 30%
    # "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/volume_fractions.json",
    # "D:/Siem/09042021/1uM_WT_30%-2_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/13042021/4uM_WT_30%_1_OUTPUT/volume_fractions.json",
    # "D:/Siem/14042021/10uM_WT_30%_0_OUTPUT/volume_fractions.json", 

    # T18N 10%
    # "D:/Siem/07042021/0uM_10_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/14042021/1uM_T18N_10%_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/15042021/4uM_T18N_10%_1_OUTPUT/volume_fractions.json",
    # "D:/Siem/19042021/10uM_T18N_10%_0_OUTPUT/volume_fractions.json",

    # T18N 20%
    # "D:/Siem/07042021/0uM_20_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/13042021/1uM_T18N_20%_0_OUTPUT/volume_fractions.json",
    # "D:/Siem/15042021/4uM_T18N_20%_3_OUTPUT/volume_fractions.json",
    # "D:/Siem/19042021/10uM_T18N_20%_0_OUTPUT/volume_fractions.json",

    # T18N 30%
    # "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/volume_fractions.json",
    # "D:/Siem/14042021/1uM_T18N_30%_3_OUTPUT/volume_fractions.json",
    # "D:/Siem/15042021/4uM_T18N_30%_1_OUTPUT/volume_fractions.json",
    # "D:/Siem/19042021/10uM_T18N_30%_1_2_OUTPUT/volume_fractions.json",
]

paths_areas = [
    # WT 10%
    # "D:/Siem/07042021/0uM_10_0_OUTPUT/avg_areas.json",
    # # "D:/Siem/08042021/1uM_WT_20%_0_OUTPUT/avg_areas.json",
    # "D:/Siem/12042021/4uM_WT_10%_0_OUTPUT/avg_areas.json",
    # "D:/Siem/13042021/10uM_WT_10%_0_OUTPUT/avg_areas.json",  

    # WT 20%
    "D:/Siem/07042021/0uM_20_0_OUTPUT/avg_areas.json",
    "D:/Siem/08042021/1uM_WT_20%_0_OUTPUT/avg_areas.json",
    "D:/Siem/12042021/4uM_WT_20%_0_OUTPUT/avg_areas.json",
    "D:/Siem/09042021/10uM_WT_20%_0_OUTPUT/avg_areas.json",  

    # WT 30%
    # "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/avg_areas.json",
    # "D:/Siem/09042021/1uM_WT_30%-2_0_OUTPUT/avg_areas.json",
    # "D:/Siem/13042021/4uM_WT_30%_1_OUTPUT/avg_areas.json",
    # "D:/Siem/14042021/10uM_WT_30%_0_OUTPUT/avg_areas.json", 

    # T18N 10%
    # "D:/Siem/07042021/0uM_10_0_OUTPUT/avg_areas.json",
    # "D:/Siem/14042021/1uM_T18N_10%_0_OUTPUT/avg_areas.json",
    # "D:/Siem/15042021/4uM_T18N_10%_1_OUTPUT/avg_areas.json",
    # "D:/Siem/19042021/10uM_T18N_10%_0_OUTPUT/avg_areas.json",

    # T18N 20%
    # "D:/Siem/07042021/0uM_20_0_OUTPUT/avg_areas.json",
    # "D:/Siem/13042021/1uM_T18N_20%_0_OUTPUT/avg_areas.json",
    # "D:/Siem/15042021/4uM_T18N_20%_3_OUTPUT/avg_areas.json",
    # "D:/Siem/19042021/10uM_T18N_20%_0_OUTPUT/avg_areas.json",

    # T18N 30%
    # "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/avg_areas.json",
    # "D:/Siem/14042021/1uM_T18N_30%_3_OUTPUT/avg_areas.json",
    # "D:/Siem/15042021/4uM_T18N_30%_1_OUTPUT/avg_areas.json",
    # "D:/Siem/19042021/10uM_T18N_30%_1_2_OUTPUT/avg_areas.json",
]

paths_amount = [
    # WT 10%
    # "D:/Siem/07042021/0uM_10_0_OUTPUT/number_of_crystals.json",
    # # "D:/Siem/08042021/1uM_WT_20%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/12042021/4uM_WT_10%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/13042021/10uM_WT_10%_0_OUTPUT/number_of_crystals.json",  

    # WT 20%
    # "D:/Siem/07042021/0uM_20_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/08042021/1uM_WT_20%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/12042021/4uM_WT_20%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/09042021/10uM_WT_20%_0_OUTPUT/number_of_crystals.json",  

    # WT 30%
    # "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/number_of_crystals.json",
    # "D:/Siem/09042021/1uM_WT_30%-2_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/13042021/4uM_WT_30%_1_OUTPUT/number_of_crystals.json",
    # "D:/Siem/14042021/10uM_WT_30%_0_OUTPUT/number_of_crystals.json", 

    # T18N 10%
    # "D:/Siem/07042021/0uM_10_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/14042021/1uM_T18N_10%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/15042021/4uM_T18N_10%_1_OUTPUT/number_of_crystals.json",
    # "D:/Siem/19042021/10uM_T18N_10%_0_OUTPUT/number_of_crystals.json",

    # T18N 20%
    # "D:/Siem/07042021/0uM_20_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/13042021/1uM_T18N_20%_0_OUTPUT/number_of_crystals.json",
    # "D:/Siem/15042021/4uM_T18N_20%_3_OUTPUT/number_of_crystals.json",
    # "D:/Siem/19042021/10uM_T18N_20%_0_OUTPUT/number_of_crystals.json",

    # T18N 30%
    "D:/Siem/07042021/0uM_30_2nd_1_OUTPUT/number_of_crystals.json",
    "D:/Siem/14042021/1uM_T18N_30%_3_OUTPUT/number_of_crystals.json",
    "D:/Siem/15042021/4uM_T18N_30%_1_OUTPUT/number_of_crystals.json",
    "D:/Siem/19042021/10uM_T18N_30%_1_2_OUTPUT/number_of_crystals.json",
]

paths_mean_radii = [
    # WT 10%
    "E:/Ice/07042021/0uM_10_0_OUTPUT/mean_radius.json",
    "E:/Ice/08042021/1uM_WT_20%_0_OUTPUT/mean_radius.json",
    "E:/Ice/12042021/4uM_WT_10%_0_OUTPUT/mean_radius.json",
    "E:/Ice/13042021/10uM_WT_10%_0_OUTPUT/mean_radius.json",  

    # WT 20%
    # "E:/Ice/07042021/0uM_20_0_OUTPUT/mean_radius.json",
    # "E:/Ice/08042021/1uM_WT_20%_0_OUTPUT/mean_radius.json",
    # "E:/Ice/12042021/4uM_WT_20%_0_OUTPUT/mean_radius.json",
    # "E:/Ice/09042021/10uM_WT_20%_0_OUTPUT/mean_radius.json",  

    # WT 30%
    # "E:/Ice/07042021/0uM_30_2nd_1_OUTPUT/mean_radius.json",
    # "E:/Ice/09042021/1uM_WT_30%-2_0_OUTPUT/mean_radius.json",
    # "E:/Ice/13042021/4uM_WT_30%_1_OUTPUT/mean_radius.json",
    # "E:/Ice/14042021/10uM_WT_30%_0_OUTPUT/mean_radius.json", 

    # T18N 10%
    # "E:/Ice/07042021/0uM_10_0_OUTPUT/mean_radius.json",
    # "E:/Ice/14042021/1uM_T18N_10%_0_OUTPUT/mean_radius.json",
    # "E:/Ice/15042021/4uM_T18N_10%_1_OUTPUT/mean_radius.json",
    # "E:/Ice/19042021/10uM_T18N_10%_0_OUTPUT/mean_radius.json",

    # T18N 20%
    # "E:/Ice/07042021/0uM_20_0_OUTPUT/mean_radius.json",
    # "E:/Ice/13042021/1uM_T18N_20%_0_OUTPUT/mean_radius.json",
    # "E:/Ice/15042021/4uM_T18N_20%_3_OUTPUT/mean_radius.json",
    # "E:/Ice/19042021/10uM_T18N_20%_0_OUTPUT/mean_radius.json",

    # T18N 30%
    # "E:/Ice/07042021/0uM_30_2nd_1_OUTPUT/mean_radius.json",
    # "E:/Ice/14042021/1uM_T18N_30%_3_OUTPUT/mean_radius.json",
    # "E:/Ice/21042021_redos/4uM_T18N_30%_14_OUTPUT/mean_radius.json",
    # "E:/Ice/21042021_redos/10uM_T18N_30%_0_OUTPUT/mean_radius.json",
]

# graph_multiple_volume_fractions(paths_ivf)
# graph_multiple_areas(paths_areas)
# graph_multiple_crystal_amounts(paths_amount)
graph_multiple_mean_radii(paths_mean_radii)