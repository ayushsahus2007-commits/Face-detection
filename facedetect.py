import cv2


# Open the default camera (0 is the device index)
cam = cv2.VideoCapture(0)


#Capturing face 
cap=cv2.CascadeClassifier("/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/cv2/data/haarcascade_frontalface_default.xml")


while True:
    ret, frame = cam.read()


    #Flipping the frame
    f_frame=cv2.flip(frame,1)


    #Changing colour to black and white
    col=cv2.cvtColor(f_frame,cv2.COLOR_BGR2GRAY)


    #Reading Structures
    face=cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    #Making Square Bracket
    for (x,y,w,h) in face:
        cv2.rectangle(f_frame,(x,y),(x+w,y+h),(0,255,0),2)


    # Display the resulting frame
    cv2.imshow('Camera',f_frame)


    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture and close windows
cam.release()
cv2.destroyAllWindows()