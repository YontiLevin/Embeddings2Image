import logging
import numpy as np
import cv2
import requests
from math import ceil

USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'


def standard_resize(image, max_side):
    if image is None:
        return None, None, None
    original_h, original_w, _ = image.shape
    if all(side < max_side for side in [original_h, original_w]):
        return image, original_h, original_w
    aspect_ratio = float(np.amax((original_w, original_h)) / float(np.amin((original_h, original_w))))

    if original_w >= original_h:
        new_w = max_side
        new_h = max_side / aspect_ratio
    else:
        new_h = max_side
        new_w = max_side / aspect_ratio

    new_h = int(new_h)
    new_w = int(new_w)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image, new_w, new_h


def url_to_img_array(url):
    if not isinstance(url, basestring):
        logging.warning("input is neither an ndarray nor a string, so I don't know what to do")
        return None

    # replace_https_with_http:
    if 'http' in url and 'https' not in url:
        url = url.replace("https", "http")
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers)
        img_array = cv2.imdecode(np.asarray(bytearray(response.content)), 1)
    except requests.ConnectionError:
        logging.warning("connection error - check url or connection")
        return None
    except:
        logging.warning(" error other than connection error - check something other than connection")
        return None

    return img_array


def get_image(img_path, image_new_size):
    if img_path.startswith('http'):
        np_img = url_to_img_array(img_path)
    else:
        img = cv2.imread(img_path)
        np_img = np.array(img)

    if np_img is None:
        return None, None, None, None, None

    small_image, x1, y1 = standard_resize(np_img, image_new_size)
    if small_image is None:
        return None, None, None, None, None

    dx = int(ceil((image_new_size - x1) / 2))
    dy = int(ceil((image_new_size - y1) / 2))
    return small_image, x1, y1, dx, dy

