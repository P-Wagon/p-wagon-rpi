import cv2
import easyocr

def detect_license_plate_live():
    # Load the pre-trained Haar Cascade classifier for license plates
    plate_cascade = cv2.CascadeClassifier('haarcascade_plate_number.xml')

    # Create an EasyOCR reader instance
    reader = easyocr.Reader(['en'])

    # Open a connection to the webcam
    cap = cv2.VideoCapture(1)

    plate_text = None
    max_confidence = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates in the frame
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected license plate
        for (x, y, w, h) in plates:
            # Extract the license plate region from the frame
            plate_roi = frame[y:y+h, x:x+w]

            # Use EasyOCR to read text from the license plate region
            results = reader.readtext(plate_roi)

            # Find the text with the highest confidence
            for result in results:
                if result[2] > max_confidence:
                    max_confidence = result[2]
                    plate_text = result[1]

            # Draw a rectangle around the license plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('License Plate Detection', frame)

        # Print the most accurate plate text
        if plate_text is not None:
            print(plate_text)
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Run the function
detect_license_plate_live()