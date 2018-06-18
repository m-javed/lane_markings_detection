import os
import cv2
import matplotlib.image as mpimg
import numpy as np
from scipy import stats
from moviepy.editor import VideoFileClip
from collections import deque


class LaneDetectorWithMemory:
    def __init__(self):
        self.left_lane_coefficients = create_lane_line_coefficients_list()
        self.right_lane_coefficients = create_lane_line_coefficients_list()

        self.previous_left_lane_coefficients = None
        self.previous_right_lane_coefficients = None

    def mean_coefficients(self, coefficients_queue, axis=0):
        return [0, 0] if len(coefficients_queue) == 0 else np.mean(
            coefficients_queue, axis=axis)

    def determine_line_coefficients(self, stored_coefficients,
                                    current_coefficients):
        if len(stored_coefficients) == 0:
            stored_coefficients.append(current_coefficients)
            return current_coefficients

        mean = self.mean_coefficients(stored_coefficients)
        abs_slope_diff = abs(current_coefficients[0] - mean[0])
        abs_intercept_diff = abs(current_coefficients[1] - mean[1])

        if abs_slope_diff > MAXIMUM_SLOPE_DIFF or \
                abs_intercept_diff > MAXIMUM_INTERCEPT_DIFF:
            # In this case use the mean
            return mean
        else:
            # Save our coefficients and return a smoothenedd one
            stored_coefficients.append(current_coefficients)
            return self.mean_coefficients(stored_coefficients)

    def lane_detection_pipeline(self, img):
        marked_image = process_image(img)
        return marked_image


def roi(img, vert):
    """
    Finds the region of image defined by the polygon formed by vertices
    keeps the image inside roi and the rest is set to zero/black
    :param img: image to be processed
    :param vert: vertices for the roi
    :return: masked image
    """

    imshape = img.shape
    mask = np.zeros_like(img)

    if len(imshape) > 2:
        # number of image channels maybe 3 or 4
        channel_count = imshape[2]
        print(channel_count)
        ignore_mask_color = (255,)* channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon with the fill color
    cv2.fillPoly(mask, np.array(vert), ignore_mask_color)

    # get masked image
    img_masked = cv2.bitwise_and(img, mask)

    return img_masked


def show_image(image, window_name = "image: "):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black.
    :param img: Image to extract ROI
    :param vertices: [numpy array] vertices of integer points.
    :return:
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    imshape = img.shape

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(imshape) > 2:
        channel_count = imshape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color="red", thickness=5):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    :param img: original image
    :param lines: lines to be drawn
    :param color: color of lines default = red [255, 0, 0]
    :param thickness: thickness of the line marker
    :return:
    """
    if color == "red":
        color = [255, 0, 0]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Draws Hough lines on a Canny transformed image
    :param img: canny transformed image
    :param rho: distance in polar coordinate system
    :param theta: angle in polar coordinate system
    :param threshold:
    :param min_line_len: minimum line length to be considered as a line
    :param max_line_gap: maximum line gap to be considered as a line
    :return: image(img) with Hough lines drawn on it
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros_like(img)
    draw_lines(line_img, lines)
    return line_img, lines


def separate_lines(lines, imshape):
    """
    Separates left and line lane marking based on their position and slope
    :param lines: detected lines
    :param imshape: shape of the original image
    :return:
    """

    middle_x = imshape[1] / 2

    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            if dx == 0:
                # Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1

            # Similarly, if the y value remains constant as x increases, discard line
            if dy == 0:
                continue

            slope = dy / dx

            # This is pure guess than anything...
            # but get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.1
            if abs(slope) <= epsilon:
                continue

            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])

    return left_lane_lines, right_lane_lines


def color_lanes(img, left_lane_lines, right_lane_lines,
                left_lane_color="red", right_lane_color="blue"):
    if left_lane_color == "red":
        left_lane_color = [255, 0, 0]
    if right_lane_color == "blue":
        right_lane_color = [0, 0, 255]

    colored_img = draw_lines(img, left_lane_lines, color=left_lane_color)
    colored_img = draw_lines(colored_img, right_lane_lines, color=right_lane_color)

    return colored_img


def find_lane_lines_formula(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

    # A straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)


def trace_lane_line(img, lines, top_y, imshape):
    A, b = find_lane_lines_formula(lines)

    bottom_y = imshape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A
    top_x_to_y = (top_y - b) / A

    new_lines = [
        [[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines)


def trace_both_lane_lines(img, left_lane_lines, right_lane_lines, vert, imshape):
    region_top_left = vert[0][1]

    left_lane_img = trace_lane_line(img, left_lane_lines, region_top_left[1], imshape)
    full_lanes_img = trace_lane_line(left_lane_img, right_lane_lines,region_top_left[1], imshape)

    return full_lanes_img


def process_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Create HSV(Hue Saturation Value) color space
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    # Convert to hsv image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # RGB2HLS

    # Create White and Yellow Masks
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(img_gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    img_mask_yw = cv2.bitwise_and(img_gray, mask_yw)

    # Filter noise using Gaussian blur
    gauss_gray = gaussian_blur(img_mask_yw, 5)

    # Canny Edge Detection (recommended ratio of 1:2 or 1:3)
    lower_canny_threshold = 50
    upper_canny_threshold = 150
    canny_edges = canny(gauss_gray, lower_canny_threshold, upper_canny_threshold)


    # get image vertices
    imshape = img.shape
    # ROI mask parameters
    mask_x_pad = 80
    mask_y_pad = 85
    vertices = np.array([[(0, imshape[0]),
                          (imshape[1] / 2 - mask_x_pad,
                           imshape[0] / 2 + mask_y_pad),
                          (imshape[1] / 2 + mask_x_pad,
                           imshape[0] / 2 + mask_y_pad),
                          (imshape[1], imshape[0])]],
                        dtype=np.int32)

    # Get image only inside the Region of Interest
    img_roi = region_of_interest(canny_edges, vertices)

    # Detect and draw Hough Lines
    hough_threshold = 10  # 10
    hough_min_lin_len = 20  # 20
    hough_max_lin_gap = 15  # 15
    img_line, lines = hough_lines(img_roi, 1, np.pi / 180, hough_threshold,
                                  hough_min_lin_len, hough_max_lin_gap)

    # separate left and right lanes
    img_sep_lns = separate_lines(lines, imshape)
#    img_color_lns = color_lanes(img, img_sep_lns[0], img_sep_lns[1])

    marked_image = trace_both_lane_lines(img, img_sep_lns[0],
                                           img_sep_lns[1], vertices, imshape)

    return marked_image


def create_lane_line_coefficients_list(length = 10):
    return deque(maxlen=length)


MAXIMUM_SLOPE_DIFF = 0.1
MAXIMUM_INTERCEPT_DIFF = 50.0


def lane_detection_images(infolder, outfolder):
    """Detects road lane markings based on computer vision

    :param infolder: Folder containing the test images
    :param outfolder: Folder to save marked test images
    :return: None
    """

    for file in os.listdir(infolder):
        img = mpimg.imread(infolder + "/" + file)

        # Convert to grayscale image
        img_full_lanes = process_image(img)
        show_image(img_full_lanes)
        mpimg.imsave(outfolder + "/" + file, img_full_lanes)


def lane_detection_videos(infolder, outfolder):
    """Detects road lane markings based on computer vision

        :param infolder: Folder containing the test videos
        :param outfolder: Folder to save marked videos
        :return: None
        """

    for file in os.listdir(infolder):
        detector = LaneDetectorWithMemory()
        clip = VideoFileClip(infolder + "/" + file)
        clip = clip.fl_image(detector.lane_detection_pipeline)
        clip.write_videofile(outfolder + "/" + file, audio=False)


def main():
    infolder_images = os.getcwd() + "/test_images"
    outfolder_images = os.getcwd() + "/images_marked"
#    lane_detection_images(infolder_images, outfolder_images)

    infolder_videos = os.getcwd() + "/test_videos"
    outfolder_videos = os.getcwd() + "/videos_marked"
    lane_detection_videos(infolder_videos, outfolder_videos)



if __name__ == '__main__':
    main()