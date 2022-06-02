import imutils
from imutils import paths
import argparse
import cv2

# Python SIFT Tutorial
# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
# https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# construct the argument parser and parse the arguments
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="path to input directory of images to stitch")
args = vars(ap.parse_args())

images = []

def load_images():
    global images
    # grab arguments
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(args["images"])))

    # load images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

def draw_keypoints():
    i = 0

    keypoint_imgs = []
    keypoints = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()

        kp = sift.detect(gray, None)

        image = cv2.drawKeypoints(gray, kp, image)
        img = cv2.drawKeypoints(gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        keypoint_imgs.append(img)
        keypoints.append(kp)

        cv2.imwrite(f'./out/sift_points-{i}.jpg', img)
        i += 1


def match_keypoints():

    # In the first step multiple images can be used - this only uses the first and the second image
    # I assumed this would be sufficient

    print("[INFO] match images...")
    img1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img)
    plt.show()

    cv2.imwrite(f'./out/matched-image.jpg', img)


def stitch_images():
    global images

    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    # if the status is '0', then OpenCV successfully performed image
    if status == 0:
        cv2.imwrite('./out/result.png', stitched)
        cv2.waitKey(0)

        draw_keypoints()
        match_keypoints()

    else:
        print("[INFO] image stitching failed ({})".format(status))

if __name__ == '__main__':
    load_images()
    stitch_images()
