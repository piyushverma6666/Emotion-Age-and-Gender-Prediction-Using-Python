import cv2
from deepface import DeepFace

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier("C:/Users/piyus/Downloads/python project/haarcascade_frontalface_default.xml")

# Start capturing video from the webcam
video = cv2.VideoCapture(0)  # Use 0 to access the default webcam

while video.isOpened():
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        try:
            # Analyze the face to detect age, gender, and emotions
            analysis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['age', 'gender', 'emotion'])
            
            # Extract results from the first dictionary in the list
            result = analysis[0]
            age = result['age']
            gender = result['dominant_gender']
            emotion = result['dominant_emotion']
            
            # Display the results
            cv2.putText(frame, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"Age: {age}, Gender: {gender}, Emotion: {emotion}")
        except Exception as e:
            print("No face or error in detection:", str(e))
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if the 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything is done, release the capture
video.release()
cv2.destroyAllWindows()
