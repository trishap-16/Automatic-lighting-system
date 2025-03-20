import cv2


def detect_():
   # Reading the Image
   image = cv2.imread("static\\uploaded_files\\image.jpg")
   height, width, channels = image.shape
   aspect_ratio = width / height
   new_width=700
   new_height = int(new_width / aspect_ratio)
   image = cv2.resize(image, (new_width, new_height))

   # initialize the HOG descriptor
   hog = cv2.HOGDescriptor()
   hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

   # detect humans in input image
   (humans, _) = hog.detectMultiScale(image, winStride=(10, 10),
   padding=(32, 32), scale=1.1)

   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

   # Convert the image to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Detect faces in the image
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

   # loop over all detected humans
   for (x, y, w, h) in humans:
      pad_w, pad_h = int(0.15 * w), int(0.01 * h)
      cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

   # Draw rectangles around the detected faces
   for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

   # display the output image
   cv2.imwrite("static\\uploaded_files\\out_image.jpg", image)


   MESSAGE = ""
   im_pth=""
   # getting no. of human detected
   if len(humans)>0 or len(faces)>0:
      print("Room is occupied")
      MESSAGE += "Room is occupied<br>Turn on the Lights"
      im_pth="BULBON.png"
   else:
      print("Room is empty")
      MESSAGE += "Room is empty<br>Turn off Lights to save power "
      im_pth="BULBOFF.png"
   print('Human Detected : ', len(humans)+len(faces))
   MESSAGE += ('<br>People Detected : '+ str(len(humans)+len(faces)))


   return MESSAGE,im_pth