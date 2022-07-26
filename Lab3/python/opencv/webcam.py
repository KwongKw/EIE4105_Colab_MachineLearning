# https://devtalk.nvidia.com/default/topic/1027250/jetson-tx2/how-to-use-usb-webcam-in-jetson-tx2-with-python-and-opencv-/
# A simple example that read the webcam connected to Jetson TX2
# To run the program, type
#   python3 webcam.py
# Type 'q' to quit, 'o' to display original frames, and 'g' to display gray-level frames.

import cv2

cap = cv2.VideoCapture("/dev/video1")       # video0 is the built-in cam and video1 is the webcam
frm_type = 'original'
while(True):
    # Capture frame-by-frame
    ret, ori = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

    if frm_type == 'gray':       
        # Display the resulting frame
        cv2.imshow('frame', gray)
    else:
        # Display the origial frame
        cv2.imshow('frame', ori)
    
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
    if k == ord('g'):
        frm_type = 'gray'
    if k == ord('o'):       
        frm_type = 'original'
            

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()