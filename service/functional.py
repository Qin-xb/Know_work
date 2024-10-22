import numpy as np
from scipy.ndimage import zoom

def resize(img: np.ndarray, size, interpolation='bilinear', max_size=None, antialias=True) -> np.ndarray:

    if not isinstance(size, (list, tuple)):
        size = [size]
    
    if len(size) == 1:  # If a single int is provided, we resize while maintaining aspect ratio
        h, w = img.shape[:2]
        if h > w:
            scale = size[0] / w
        else:
            scale = size[0] / h
        output_size = (int(h * scale), int(w * scale))
    else:
        output_size = size

    # Handle max_size
    if max_size:
        max_dim = max(output_size)
        if max_dim > max_size:
            scale = max_size / max_dim
            output_size = (int(output_size[0] * scale), int(output_size[1] * scale))

    # Compute the zoom factors
    zoom_factors = [output_size[0] / img.shape[0], output_size[1] / img.shape[1], 1]

    if interpolation == 'nearest':
        resized_img = zoom(img, zoom_factors, order=0)
    elif interpolation == 'bilinear':
        resized_img = zoom(img, zoom_factors, order=1)
    else:
        raise ValueError("Unsupported interpolation mode. Choose between 'nearest' and 'bilinear'.")

    return resized_img

