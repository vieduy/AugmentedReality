# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)

import argparse
import pickle
import cv2
import numpy as np
import math
import os
from .objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 100


def main(model_name, number_model):
    """
    This functions loads the target surface image,
    """
    homography = None
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[678, 0, 320], [0, 671, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.xfeatures2d.SIFT_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model_paths = ['ref_final', 'ref_7']
    models = []

    for i in range(int(number_model)):
        model = cv2.imread(os.path.join(dir_name, 'reference/' + model_paths[i] + '.jpg'), 0)

        # Compute model keypoints and its descriptors
        kp_model, des_model = orb.detectAndCompute(model, None)
        models.append([kp_model, des_model, model])

    obj_paths = ['Chick', 'Whale', 'Red Fox', 'bird']
    objs = []

    # Load 3D model from OBJ file
    for i in range(int(number_model)):
        if int(number_model) == 1:
            obj = OBJ(os.path.join(dir_name, './models/' + model_name + '.obj'), swapyz=True)
            obj_paths[0] = model_name
        else:
            obj = OBJ(os.path.join(dir_name, './models/' + obj_paths[i] + '.obj'), swapyz=True)
        objs.append(obj)

    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return

        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        projections = []

        # match frame descriptors with model descriptors
        for i, model in enumerate(models):
            matches = bf.match(model[1], des_frame)

            # sort them in the order of their distance
            # the lower the distance, the better the match
            matches = sorted(matches, key=lambda x: x.distance)

            # compute Homography if enough matches are found
            if len(matches) > MIN_MATCHES:
                # differenciate between source points and destination points
                src_pts = np.float32([model[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # compute Homography
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Draw a rectangle that marks the found model in the frame
                h, w = model[2].shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)

                # connect them with lines
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                # if a valid homography matrix was found render cube on model plane
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(camera_parameters, homography)
                        projections.append(projection)

                        # project cube or model
                        frame = render(frame, objs[i], projection, models[i][2], obj_paths[i], False)
                        # frame = render(frame, model, projection)
                    except:
                        pass
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return projection, camera_parameters


def render(img, obj, projection, model, objectName, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3)* 30
    if objectName == "tree":
        scale_matrix = np.eye(3) * 0.015
    elif objectName == "house":
        scale_matrix = np.eye(3) * 0.65

    elif objectName == "Whale":
        scale_matrix = np.eye(3) * 25
    h, w = model.shape
    numberOfPoints = len(obj.faces)
    for counter in range(len(obj.faces)):
        if objectName == 'Whale' or objectName == 'bird' or objectName == 'Red Fox' or objectName == 'Chick':
            face_vertices = obj.faces[len(obj.faces) - counter - 1][0]
        else:
            face_vertices = obj.faces[counter][0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False and counter < len(obj.faces):
            if objectName == 'Red Fox':
                if 0 <= counter <= 65:
                    cv2.fillConvexPoly(img, imgpts, (250, 250, 250))
                else:
                    cv2.fillConvexPoly(img, imgpts, (0, 128, 255))
            if objectName == 'Chick':
                if 0 <= counter <= 39:
                    cv2.fillConvexPoly(img, imgpts, (0, 128, 255))
                else:
                    cv2.fillConvexPoly(img, imgpts, (102, 255, 255))

            if objectName == 'bird':
                if 0 <= counter <= 7:
                    cv2.fillConvexPoly(img, imgpts, (250, 250, 250))
                else:
                    cv2.fillConvexPoly(img, imgpts, (255, 128, 4))

            if objectName == 'Whale':
                if 0 <= counter <= 228:
                    cv2.fillConvexPoly(img, imgpts, (250, 250, 250))
                else:
                    cv2.fillConvexPoly(img, imgpts, (65, 44, 4))
            if objectName == "ship":
                if 12 <= counter <= 41:
                    cv2.fillConvexPoly(img, imgpts, (65, 44, 4))
                else:
                    cv2.fillConvexPoly(img, imgpts, (250, 250, 250))
            if objectName == "tree":
                if counter < numberOfPoints / 1.5:
                    cv2.fillConvexPoly(img, imgpts, (27, 211, 50))
                else:
                    cv2.fillConvexPoly(img, imgpts, (33, 67, 101))
            elif objectName == "house":
                if counter < 1 * numberOfPoints / 32:
                    cv2.fillConvexPoly(img, imgpts, (226, 219, 50))
                elif counter < 2 * numberOfPoints / 8:
                    cv2.fillConvexPoly(img, imgpts, (250, 250, 250))
                elif counter < 13 * numberOfPoints / 16:
                    cv2.fillConvexPoly(img, imgpts, (28, 186, 249))
                else:
                    cv2.fillConvexPoly(img, imgpts, (14, 32, 130))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r', '--rectangle', help='draw rectangle delimiting target surface on frame', action='store_true')
parser.add_argument('-mk', '--model_keypoints', help='draw model keypoints', action='store_true')
parser.add_argument('-fk', '--frame_keypoints', help='draw frame keypoints', action='store_true')
parser.add_argument('-ma', '--matches', help='draw matches between keypoints', action='store_true')
# TODO jgallostraa -> add support for model specification
# parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

# with open('ar_camera.pkl', 'wb') as f:
#     pickle.dump(K, f)
#     pickle.dump(np.dot(np.linalg.inv(K), cam),f)

# if __name__ == '__main__':
#     main()
