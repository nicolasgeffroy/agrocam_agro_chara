## General function

import os
import PIL.Image
import base64
import requests
from io import BytesIO
from typing import Optional, Union

def load_image(image: Union[str, "PIL.Image.Image"], timeout: Optional[float] = None, mode = ["RGB"]) -> "PIL.Image.Image":
    # Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/image_utils.py
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.
        mode ('list'): (added)
            Image representation space ("RGB", "HSV", "LAB"...)

    Returns:
        `np.array`: A PIL Image converted in a np.array.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            image = PIL.Image.open(BytesIO(requests.get(image, timeout=timeout).content))
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            if image.startswith("data:image/"):
                image = image.split(",")[1]

            # Try to load as base64
            try:
                b64 = base64.decodebytes(image.encode())
                image = PIL.Image.open(BytesIO(b64))
            except Exception as e:
                raise ValueError(
                    f"Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image file, or a base64 encoded string. Got {image}. Failed with {e}"
                )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise TypeError(
            "Incorrect format used for image. Should be an url linking to an image, a base64 string, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    if len(mode) == 1:
        image_fin = image.convert(mode[0])
        image_fin = np.array(image_fin)
    else :
        for m in range(len(mode)) :
            if m==0:
                image_fin = image.convert(mode[m])
                image_fin = np.array(image_fin)
            else:
                image_temp = image.convert(mode[m])
                image_temp = np.array(image_temp)
                image_fin = np.append(image_fin, image_temp, axis=2)
    return image_fin

## Different function used to extract different agronomic characteristics from the vineyard

import numpy as np

def height_para(img):
    """
    Calculate the normalized height of the canopy vineyard in the image.

    This function determines the vertical span of non-zero pixels in the canopy mask,
    representing the height of the canopy.
    The result is then normalized by dividing by the total image height (1080 pixels).

    Parameters
    ----------
    img : numpy.ndarray
        Input canopy mask or image as either a 3D or a 2D numpy array.

    Returns
    -------
    height : float or numpy.nan
        Normalized height of the canopy (height / 1080).
        Returns `numpy.nan` if the image contains only zero pixels.
    """

    ## Summing pixel values across width (and channels)
    if len(img.shape) == 3 :
        # Sum pixel values along the width (axis=1) and then across channels (axis=1) if the input is a 3D array.
        sum_RGB = np.sum(img, axis=1)
        sum_RGB = sum_RGB.sum(axis=1)
    elif len(img.shape) == 2:
        # If the input is a 2D array, sum pixel values along the width (axis=1)
        sum_RGB = np.sum(img, axis=1)
    else :
        print("Incorrect dimension")

    ## Handling edge case: all-zero image
    # If the sum of all pixel values is zero, return NaN (no canopy detected).
    if sum_RGB.sum() == 0:
        return np.nan

    ## Finding the top of the canopy
    # Iterate from the bottom to the top of the image to find the highest non-zero row.
    for i in range(len(sum_RGB)):
        if sum_RGB[i] != 0:
            high = i  # Row index of the top of the canopy
            break
    
    ## Finding the bottom of the canopy
    # Iterate from the top to the bottom of the image to find the lowest non-zero row.
    for i in reversed(range(len(sum_RGB))):
        if sum_RGB[i] != 0:
            low = i  # Row index of the bottom of the canopy
            break

    ## Calculating and returning normalized height
    # Calculate the height as the difference between the top and bottom width 
    # normalized by the total image width (1080 pixels).
    return round(-(high - low) / 1080, 3)

# import numpy as np

def porosity_para(img_zone, img_enti, type_entity="sheath", corr=50):
    """
    Calculate the porosity of a plant canopy zone relative to the entire plant area.

    This function computes the porosity by substracting the area (number of pixel) of the canopy 
    by the area of the upper part of the image (determining using the trunc or the sheath). 
    It is then normalized by the area of the upper part of the image.

    Parameters
    ----------
    img_zone : numpy.ndarray
        3D or 2D array representing the trunc and sheath.
    img_enti : numpy.ndarray
        3D or 2D array representing the canopy.
    type_entity : strings
        Informing the type of mask the `img_zone` represent.
        Either the sheath mask ("sheath") or the trunk mask ("trunk").
    corr : int, optional
        Correction factor to adjust the lower boundary of the zone. Default is 50.

    Returns
    -------
    porosity : float or numpy.nan
        Porosity value (ratio of empty space to total zone area).
        Returns `numpy.nan` if either input image is all zeros.
    """
    
    ## Summing pixel values across width (and channels)
    if len(img_zone.shape) == 3 :
        # Sum pixel values along the width (axis=1) and then across channels (axis=1) if the input is a 3D array.
        sum_RGB = np.sum(img_zone, axis=1)
        sum_RGB = sum_RGB.sum(axis=1)
    elif len(img_zone.shape) == 2:
        # If the input is a 2D array, sum pixel values along the width (axis=1)
        sum_RGB = np.sum(img_zone, axis=1)
    else :
        print("Incorrect dimension")

    ## Handling edge case: all-zero image
    # If either image is all zeros, return NaN (no data to process).
    if img_zone.sum() == 0 or img_enti.sum() == 0:
        return np.nan

    ## Determining the mask of the upper part of the image using either the trunk or sheath mask.
    ## Find the bottom of the upper part of the image using the sheath mask or the trunk mask.
    if type_entity == "sheath" :
        # When using the sheath mask, we use the first non-zero row (for the bottom of the sheath).
        for i in reversed(range(len(sum_RGB))):
            if sum_RGB[i] != 0:
                low_zone = i
                break
        # Adjust the lower boundary of the zone using the correction factor.
        low_zone = low_zone + corr
    elif type_entity == "trunk":
        # When using the trunk mask, we use the first non-zero row (for the top of the trunk).
        for i in range(len(sum_RGB)):
            if sum_RGB[i] != 0:
                low_zone = i
                break
    else :
        print("Wrong entity name")
    ## Creating a mask for the upper zone by taking its bottom (determined earlier) and all the pixel to the top of the image.
    # Initialize a zero array for the zone mask.
    img_zone_plus = np.zeros((1080, 1920, 3))
    # Fill the zone mask up to the adjusted lower boundary.
    for i in range(1080):
        for j in range(1920):
            # Set pixels in the upper zone to 1.
            if i < low_zone:
                img_zone_plus[i, j, :] = 1
            # Binarize the entire plant image (set non-zero pixels to 1).
            if img_enti[i, j, :].sum() != 0:
                img_enti[i, j, :] = 1

    ## Calculating porosity
    # Subtract the binarized canopy mask from the upper zone mask to find empty spaces.
    img_z_e = img_zone_plus - img_enti
    # Calculate porosity as the ratio of empty space to total upper zone area.
    return round(img_z_e.sum() / img_zone_plus.sum(), 3)

# import numpy as np

def hue_para(ori_img,img):
    """
    Calculate the average hue channel intensity in order to caracterize the leaf color in the canopy 
    (if the canopy mask is used) and the state of the interrow (if the interrow mask is used).

    This function filters out black pixels (where R=G=B=0) and computes the mean intensity
    of the hue channel for the remaining pixels.

    Parameters
    ----------
    img : numpy.ndarray
        Input HSV image as a 3D numpy array (height × width × channels).

    Returns
    -------
    hue_mean : float or numpy.nan
        Mean intensity of the hue channel for non-black pixels, rounded to 2 decimal places.
        Returns `numpy.nan` if the image contains only black pixels.
    """

    ## Filtering out black pixels
    img = ori_img * img
    # Create a boolean mask where True indicates non-black pixels (R, G, and B all non-zero).
    filter = np.sum(img != [0, 0, 0], axis=2) == 3
    # Apply the filter.
    img = img[filter]
    # If no non-black pixels are found, return NaN.
    if filter.sum() == 0:
        return np.nan

    ## Calculating mean hue intensity
    # Extract the hue channel (first channel in HSV format in PIL formating).
    img_hue = [img[i][0] for i in range(img.shape[0])]
    # Calculate the mean hue intensity across all non-black pixels.
    img_hue = np.mean(img_hue)
    # Round the result to 2 decimals.
    return round(img_hue, 2)

import copy
from pandas import DataFrame, to_datetime
from tqdm import tqdm

def agro_para_extr(data, type_entity="sheath", save=False):
    """
    Extract agronomic parameters from Agrocam images and masks.

    This function loads image and mask data, then calculates four key agronomic parameters:
    vine height, vine porosity, inter-row greenness, and vine greenness.
    Results are stored in a DataFrame and optionally saved to CSV.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame in the format returned by the function `data_loading` with its default parameters.
    type_entity : strings
        Informing the type of mask used for determining the upper zone of the image.
    save : bool, optional
        If True, saves the results to a CSV file every 50 iterations and at the end.
        Default is False.

    Returns
    -------
    para_agro : pd.DataFrame
        DataFrame containing extracted agronomic parameters for each image:
        - time: Date of image capture
        - treatment: Experimental treatment
        - dist_sheath_trunc: Distance between top of trunk and bottom of canopy
        - H_vigne: Vine height (normalized)
        - P_vigne: Vine porosity
        - Hue_vigne: Vine hue (HSV hue)
        - Hue_rang: Inter-row hue (HSV hue)
    """

    ## Loading and preparing data
    # Initialize a DataFrame to store extracted parameters.
    data['day_time'] = to_datetime(data['day_time'], format="%Y-%m-%d %H:%M:%S")
    para_agro = DataFrame({
        "image": data["image"],
        "time": data["day_time"].dt.date,
        "treatment": data["treatment"]
    })
    # Add the distance between top of trunk and bottom of canopy.
    para_agro["dist_sheath_trunc"] = data["dist_sheath_trunc"]
    # Initialize columns for agronomic parameters.
    para_agro["H_vigne"], para_agro["P_vigne"], para_agro["Hue_vigne"], para_agro["Hue_rang"] = 0, 0, 0, 0

    ## Extracting parameters for each image
    for i in tqdm(range(data.shape[0])):
        
        # Skip a specific problematic image.
        all_target = load_image(data.loc[i, "all"], mode=["L"])
        label = ["bck", "feuille", "inter", "sheath", "trunk"]
        all_classes = {i : 0 for i in label if i != "bck"}
        for o in range(1,len(label)):
            # Extract the predicted and target binary mask of the selected class
            mask = (all_target == o) * 1.0
            mask = np.expand_dims(mask, 2)
            all_classes[label[o]] = mask
        
        # Load original image in HSV color space used for hue calculation.
        ori_img = load_image(data.loc[i, "image"], mode=["HSV"])
        feuille_2 = copy.deepcopy(all_classes["feuille"])  # Create a copy for hue analysis
        
        ## Calculating agronomic parameters
        # Calculate vine height (normalized).
        para_agro.loc[i, "H_vigne"] = height_para(all_classes["feuille"])
        # Calculate vine porosity using the sheath mask or the trunk mask.
        para_agro.loc[i, "P_vigne"] = porosity_para(all_classes[type_entity], all_classes["feuille"], corr=data.loc[i, "dist_sheath_trunc"])
        # Calculate inter-row mean hue.
        para_agro.loc[i, "Hue_rang"] = hue_para(ori_img, all_classes["inter"])
        # Calculate vine mean hue.
        para_agro.loc[i, "Hue_vigne"] = hue_para(ori_img, feuille_2)

        ## Periodically saving results
        # Save results to CSV every 50 iterations and at the end of processing.
        if ((i % 10 == 0) or (i == data.shape[0] - 1)) and save:
            para_agro.to_csv("Core/Results/Agro_chara_vine.csv", index=False)

    return para_agro

if __name__ == "__main__":
    import argparse
    from pandas import read_csv
    ## Ask the relevant arguments
    parser = argparse.ArgumentParser(description='Train or Use a segmentation model on a set of images.')
    # Arguments for the generation of the training database
    parser.add_argument('--name_of_database_used', type=str, required=False, 
                        help='URL of the csv file containing images, mask and information about it.', 
                        default="Core/Results/Image_chara_all.csv")
    # Name of the entity used for generating the upper zone of the image
    parser.add_argument('--name_of_mask_used', type=str, required=False, 
                        help='Name of the entity used to determine the upper part of the image', 
                        default="sheath")
    # Store the parsed arguments
    args = parser.parse_args()
    
    ## Read the database with all the Agrocam Image
    data = read_csv(args.name_of_database_used)
    _ = agro_para_extr(data, type_entity=args.name_of_mask_used, save=True)