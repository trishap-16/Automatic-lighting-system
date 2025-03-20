import cv2

# Load the image
img = cv2.imread('samp_inp/p4.jpg')
height, width, channels = img.shape
aspect_ratio = width / height
new_width=700
new_height = int(new_width / aspect_ratio)
img = cv2.resize(img, (new_width, new_height))


# Create a cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Image with detected faces', img)
print("Number of faces:",len(faces))
cv2.waitKey(0)
cv2.destroyAllWindows()
