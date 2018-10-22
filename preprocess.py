#!usr/bin/python2

import datetime
import pandas as pd
import numpy as np
import scipy.ndimage
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from glob import glob

tstart = None
tend = None


def start_time():
    """To record the start time """

    global tstart

    tstart = datetime.datetime.now()


def get_delta():
    """To calculate spent time """

    global tstart
    tend = datetime.datetime.now()

    return tend - tstart



def input_mhd(in_mhd):
    """input mhd image"""

    #the simplest method to show 2D,one slice
    img_original = sitk.ReadImage(in_mhd)
    img_array = sitk.GetArrayFromImage(img_original)
    origin = np.array(img_original.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(img_original.GetSpacing())  # spacing of voxels in world coor. (mm)
    #center = np.array([node_x, node_y, node_z])  # nodule center
    #v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
    shape= img_array.shape #depth,height,width


    #another 2D method have many parameter
    #idxslice = 0  #Slice index to visualize with 'sitk_show'
    #labelgraymatter = 1  #int label to assign to the segmented gray matter
    #sitk_show(sitk.Tile(img_original[:, :, idxslice],(2, 1, 0)))

    #CT images of this competition don't need this step,data has already been what we want
    #hu_pixel = hu(img_original, img_array)
    #plt.hist(hu_pixel.flatten(), bins=80, color='c')

    #check the data
    # plt.hist(img_array.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()
    # # Show some slice in the middle
    # plt.imshow(img_array[80], cmap=plt.cm.gray)
    # plt.show()

    #resample
    #img_resampled, new_spacing = resample(img_array,spacing, [1, 1, 1])
    #new_shape = img_resampled.shape
    #print("Shape before resampling\t", img_array.shape)
    #print("Shape after resampling\t", img_resampled.shape)

    #associate csv with image
    center = [79.4668791001, -115.730855121, 112.700994197]
    diameter = 38.5535692493
    #segement nodule
    seg_nodule(img_array, center, diameter, origin, spacing, shape)
    return


def hu(img_origin,img):
    """convert to Hounsfield Unit"""
    img = img.astype(np.int16)
    #img[img == -2000] = 0
    # for slice in range(img.shape[0]):
    #     intercept = img_origin[slice].RescaleIntercept
    #     slope = img_origin[slice].RescaleSlope
    #     if slope != 1:
    #         img[slice] = slope * img[slice].astype(np.float64)
    #         img[slice] = img[slice].astype(np.int16)

    slope = 1
    intercept = 1024
    # for slice in range(img.shape[0]):
    #     img[slice] = slope * img[slice].astype(np.float64)
    #     img[slice] = img[slice].astype(np.int16)
    #     img[slice] += np.int16(intercept)

    return np.array(img, dtype=np.int16)


def resample(image, spacing, new_spacing=[1, 1, 1]):
    """resample the picture and let every voxel represent 1mm*1mm*1mm"""
    # Determine current pixel spacing for dicom
    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def seg_nodule(img, center, diameter, origin, spcaing, shape):
    """segment the nodule according to  world coordinates in csv"""
    add_range = 0
    mask = np.zeros(shape)
    voxel_center = (center - origin) / spcaing
    intv_center = np.rint(voxel_center)
    diameter3D = [diameter, diameter, diameter]
    voxel_center_low = np.rint(voxel_center - diameter3D/spcaing - add_range)
    voxel_center_high = np.rint(voxel_center + diameter3D/spcaing + add_range)
    print voxel_center_low
    for x in range(int(voxel_center_low[0]), int(voxel_center_high[0]+1)):
        for y in range(int(voxel_center_low[1]), int(voxel_center_high[1]+1)):
            for z in range(int(voxel_center_low[2]), int(voxel_center_high[2] + 1)):
                mask[x, y, z] = 1.0
    #seg_img = img*mask
    #np.savetxt("media.txt", mask[int(intv_center[2])])

    #plt.imshow(seg_img[int(intv_center[2])], cmap=plt.cm.gray)
    #plt.show()
    #sitk.WriteImage(seg_img, "seg_img.mhd")
    return


def sitk_show(img, title=None, margin=0.05, dpi=40):
    """show mhd images using SimpleITK and matplotlib"""

    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()


def plot_3d(image, threshold=0):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()



#help(SimpleITK)
start_time()

read_csv = os.path.join("./data/csv/train/annotations.csv")
mhd_path = "./data/mhd_train/LKDS-00081.mhd"
mhd_train = glob(mhd_path + "*.mhd")
mask_path = "./data/mask"
info = in_data = pd.read_csv(read_csv)

input_mhd(mhd_path)

#plot_3d(img_array, 0) #too slow

print get_delta()

