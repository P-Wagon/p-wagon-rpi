import cv2
import os
import requests
import threading
from base64 import b64encode
import json
from datetime import datetime

# Global variables
unique_texts = []
lock = threading.Lock()

suspected_plate = input("Enter the suspected plate: ")

def detect_and_crop_license_plate_from_webcam():
    # Load the pre-trained Haar Cascade classifier for license plates
    plate_cascade = cv2.CascadeClassifier('haarcascade_plate_number.xml')

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates in the frame
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected license plates
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop the detected license plate region
            plate_roi = frame[y:y+h, x:x+w]

            # Generate a unique filename using current timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'cropped_images/cropped_plate_{timestamp}.jpg'

            # Save the cropped license plate image
            cv2.imwrite(filename, plate_roi)

            # Start a new thread for OCR processing
            threading.Thread(target=process_image, args=(filename,)).start()

        # Display the frame with license plate detection
        cv2.imshow('License Plate Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def makeImageData(imgpath):
    img_req = None
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def requestOCR(url, api_key, imgpath):
    imgdata = makeImageData(imgpath)
    response = requests.post(url, 
                             data=imgdata, 
                             params={'key': api_key}, 
                             headers={'Content-Type': 'application/json'})
    return response

def process_image(image_location):
    with open('vision_api.json') as f:
        data = json.load(f)
        
    ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
    api_key = data["key"]
    
    result = requestOCR(ENDPOINT_URL, api_key, image_location)

    try:
        if result.status_code != 200 or result.json().get('error'):
            return None
        else:
            result = result.json()['responses'][0]['textAnnotations']

        final_description = ''
        for index in range(len(result)):
            description = result[index]["description"]
            description_without_spaces = description.replace(" ", "")
            
            if len(description_without_spaces) == 10 and \
               description_without_spaces[:2].isalpha() and \
               description_without_spaces[-4:].isdigit() and \
               description_without_spaces[2:4].isdigit() and \
               description_without_spaces[4:6].isalpha():
                final_description = description_without_spaces

        with lock:
            if final_description and final_description not in unique_texts:
                unique_texts.append(final_description)

        return final_description
    except KeyError:
        return None

def find_potential_matches(suspected_plate):
    # Initialize a list to store potential matches
    potential_matches = []

    # Iterate through the unique texts list
    for text in unique_texts:
        # Check for exact match
        if text == suspected_plate:
            # If exact match, add it to the list and break the loop
            potential_matches.append(text)
            break
        
    if len(potential_matches) == 0:
        # Check for approximate match (7 out of 10 characters similar)
        for match in unique_texts:
            count = 0
            for i in range(10):
                if suspected_plate[i] == match[i]:
                    count += 1
            if count >= 7:
                potential_matches.append(match)

    return potential_matches

if __name__ == "__main__":
    # Call the function to start license plate detection and OCR processing from webcam
    detect_and_crop_license_plate_from_webcam()

    # Print the list of unique final_text values
    for text in unique_texts:
        print(text)

    # Find potential matches for the suspected plate
    matches = find_potential_matches(suspected_plate)

    # Print the potential matches
    print("Potential Matches:", matches)