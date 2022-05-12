import numpy as npy


def compute_orientation_from_polyline(polyline: npy.ndarray) -> npy.ndarray:
    assert isinstance(polyline, npy.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(polyline[0,:]) == 2, '<Math>: not a valid polyline. polyline = {}'.format(polyline)

    if (len(polyline)<2):
        raise NameError('Cannot create orientation from polyline of length < 2')

    orientation = []
    for i in range(0, len(polyline)-1):
        pt1 = polyline[i]
        pt2 = polyline[i+1]
        tmp = pt2 - pt1
        orientation.append(npy.arctan2(tmp[1], tmp[0]))

    for i in range(len(polyline)-1, len(polyline)):
        pt1 = polyline[i-1]
        pt2 = polyline[i]
        tmp = pt2 - pt1
        orientation.append(npy.arctan2(tmp[1], tmp[0]))
    orientation = npy.array(orientation)

    return orientation