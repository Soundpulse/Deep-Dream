import cv2
import os

# Playing video from file:
cap = cv2.VideoCapture('test_vid.mp4')

try:
    if not os.path.exists('image_data'):
        os.makedirs('image_data')
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0
ret = True
while ret:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    strn = str(currentFrame).zfill(5)
    name = './image_data/frame' + strn + '.png'
    print('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
