# import the opencv library
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        print('capture success')
        cv2.imwrite('../reference/img.jpg', frame)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
#
# # Implementation of matplotlib function
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# import matplotlib.image as mpimg
#
# img = mpimg.imread('img/img_gray.jpg')
# # img = np.uint8((0.2126 * img) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
#
# fig = plt.figure()
# # fig.add_subplot(221)
# plt.title('image 1')
# plt.set_cmap('gray')
# plt.imshow(img)
#
# fig.suptitle('matplotlib.figure.Figure.ginput() \
# function Example', fontweight="bold")
#
# print("After 3 clicks :")
# x = fig.ginput(2)
# print(x)
#
# plt.show()
