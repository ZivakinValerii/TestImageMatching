# This is a sample Python script.
import cv2
from matplotlib import pyplot
import numpy as np
import copy
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Image Matching Argument Parser')
    parser.add_argument('--first_image', default='DJI_0268.JPG', type=str, help='Path to the first image')
    parser.add_argument('--second_image', default='DJI_0267.JPG', type=str, help='Path to the second image')
    parser.add_argument('--detector_type', default='Akaze', choices=['Sift', 'Akaze'],
                        help='type of feature detector (default=Akaze)')
    parser.add_argument('--match_threshold', type=float, default=0.7, help='Threshold for matching (default = 0.7)')
    parser.add_argument('--good_matches_threshold', type=int, default=20,
                        help='Threshold for good matches (default = 20)')
    parser.add_argument('--acceptable_rotation_angle', type=int, default=30,
                        help='Acceptable rotation angle (default = 30)')
    parser.add_argument('--show_result', action='store_true',
                        help='Whether to show the resulting image (default = False)')
    return parser


def get_rotation_angle_homography(H):
    # Извлекаем угол поворота из гомографической матрицы
    cos_theta = H[0, 0]
    sin_theta = H[1, 0]
    rotation_angle = np.arctan2(sin_theta, cos_theta)
    return np.degrees(rotation_angle)


def combine_two_images(image1_path, image2_path, detector_type, match_threshold, good_matches_threshold,
                       acceptable_rotation_angle, show_result: bool):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    if detector_type == 'Akaze':
        detector = cv2.AKAZE_create()
    elif detector_type == 'Sift':
        detector = cv2.SIFT_create()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # sometimes gray scale could be better
    kp1, descriptors1 = detector.detectAndCompute(image1, None)  # kp = keypoints

    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kp2, descriptors2 = detector.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors2, descriptors1, k=2)
    good = []
    for m, n in matches:
        if m.distance < match_threshold * n.distance:
            good.append(m)

    if len(good) < good_matches_threshold:
        return 0.0

    matches = copy.copy(good)
    src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homog_result = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    H = homog_result[0]

    ang = get_rotation_angle_homography(H)
    if abs(ang) > acceptable_rotation_angle:
        return 0.0
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    corners1 = np.float32(([0, 0], [0, height1], [width1, height1], [width1, 0]))
    corners2 = np.float32(([0, 0], [0, height2], [width2, height2], [width2, 0]))
    warped_corners2 = np.zeros((4, 2))

    for i in range(0, 4):
        cornerX = corners2[i, 0]
        cornerY = corners2[i, 1]

        warped_corners2[i, 0] = (H[0, 0] * cornerX + H[0, 1] * cornerY + H[0, 2]) / (
                H[2, 0] * cornerX + H[2, 1] * cornerY + H[2, 2])
        warped_corners2[i, 1] = (H[1, 0] * cornerX + H[1, 1] * cornerY + H[1, 2]) / (
                H[2, 0] * cornerX + H[2, 1] * cornerY + H[2, 2])

    allCorners = np.concatenate((corners1, warped_corners2), axis=0)

    [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

    translation = np.float32(([1, 0, -1 * xMin], [0, 1, -1 * yMin], [0, 0, 1]))
    warped_res_img = cv2.warpPerspective(image1, translation, (xMax - xMin, yMax - yMin))
    full_transformation = np.dot(translation, H)  # again, images must be translated to be 100% visible in new canvas
    warped_image2 = cv2.warpPerspective(image2, full_transformation, (xMax - xMin, yMax - yMin))
    result_image = np.where(warped_image2 != 0, warped_image2, warped_res_img)

    max_h = max(height1, height2)
    max_w = max(width1, width2)
    res_h, res_w = result_image.shape[:2]
    part_h = 1.0 - (res_h - max_h) / max_h
    part_w = 1.0 - (res_w - max_w) / max_w

    if show_result:
        pyplot.imshow(result_image), pyplot.show()

    return part_h * part_w


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('Path to the first image:', args.first_image)
    print('Path to the second image:', args.second_image)
    print('Type of detector:', args.detector_type)
    print('Match threshold:', args.match_threshold)
    print('Good matches threshold:', args.good_matches_threshold)
    print('Acceptable rotation angle:', args.acceptable_rotation_angle)
    print('Show result:', args.show_result)
    print(combine_two_images(args.first_image, args.second_image, args.detector_type,
                             args.match_threshold, args.good_matches_threshold, args.acceptable_rotation_angle,
                             args.show_result))
