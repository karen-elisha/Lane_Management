import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import pytesseract

# Initialize Mediapipe hands detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup hands detection with default parameters
hands = mp_hands.Hands(
    max_num_hands=2,  # Track up to two hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Set the path to Tesseract executable (adjust this path based on your Tesseract installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as necessary

# Function to detect flash
def detect_flash(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    height, width = thresh.shape
    left_half = thresh[:, :width // 2]
    right_half = thresh[:, width // 2:]
    left_flash = np.any(left_half)
    right_flash = np.any(right_half)
    return left_flash, right_flash

# Function to enhance the image for OCR
def enhance_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpen filter
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)  # Apply sharpening
    _, thresh = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY)  # Binarize image
    return thresh

# Function to perform OCR on the captured image
def recognize_text(image):
    enhanced_image = enhance_image_for_ocr(image)  # Enhance the image for OCR
    # Add configuration for recognizing only digits
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(enhanced_image, config=custom_config)  # Use pytesseract to extract text
    return text.strip()  # Return the recognized text

# Function to take a photo and perform OCR
def take_photo(frame):
    # Create directory for photos if it doesn't exist
    directory = "photos"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Flip the frame horizontally to save it inverted
    inverted_frame = cv2.flip(frame, 1)
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(directory, f"photo_{timestamp}.jpg")
    # Save the inverted frame as a photo
    cv2.imwrite(photo_path, inverted_frame)
    print(f"Photo taken and saved as: {photo_path}")
    
    # Perform OCR on the frame and return the recognized text
    recognized_text = recognize_text(inverted_frame)
    if recognized_text:  # Check if text was recognized
        print("Recognized Numbers:", recognized_text)  # Print the recognized numbers
    else:
        print("No numbers recognized.")
    return recognized_text

# Function to load and process an image from a dataset
def load_and_process_image(image_name, dataset_path):
    image_path = os.path.join(dataset_path, image_name)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_name}")
        return None
    
    print("Image loaded successfully.")
    
    # Perform hand and flash detection, and OCR
    left_flash, right_flash = detect_flash(image)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = hand_type.classification[0].label
            cv2.putText(image, hand_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    recognized_text = recognize_text(image)
    if recognized_text:
        print("Recognized Numbers from the loaded image:", recognized_text)
    else:
        print("No numbers recognized in the loaded image.")
    
    return image

# Function to preprocess images
def preprocess_images(dataset_path, output_path):
    # Check if output directory exists, create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Preprocess all images in the dataset
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, (224, 224))  # Resize to 224x224 pixels
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
            # Normalize the image
            normalized_image = gray_image / 255.0  # Normalize to [0, 1]
            
            # Apply Gaussian blur
            blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

            # Save the preprocessed image
            output_image_name = f"processed_{image_name}"  # Prefix to distinguish processed images
            cv2.imwrite(os.path.join(output_path, output_image_name), (blurred_image * 255).astype(np.uint8))

    print("Preprocessing completed for all images.")

def main():
    # Specify the paths
    dataset_path = r"C:\Users\H P\Downloads\dataset"  # Change to your dataset folder path
    output_path = r"C:\Users\H P\OneDrive\Desktop\output"  # Change this to your desired output folder path
    image_name = r"C:\Users\H P\Downloads\dataset\1234.jpg"  # Change to your image file name

    # Preprocess images before proceeding with the main functionalities
    preprocess_images(dataset_path, output_path)

    # Load and process the specified image
    load_and_process_image(image_name, dataset_path)

    # Open the webcam for real-time detection
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Dictionaries to store detection statuses
    flash_status = {
        "left_flash": False,
        "right_flash": False
    }
    hand_status = {
        "left_hand_detected": False,
        "right_hand_detected": False
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Flip the frame to correct the mirror effect
        frame = cv2.flip(frame, 1)
        
        # Step 1: Detect flash
        left_flash, right_flash = detect_flash(frame)
        flash_status["left_flash"] = left_flash
        flash_status["right_flash"] = right_flash
        
        # Step 2: Perform hand detection after flash detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Reset the hand status before updating it
        hand_status["left_hand_detected"] = False
        hand_status["right_hand_detected"] = False
        
        # Step 3: Update hand detection status
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the hand type (left or right)
                hand_label = hand_type.classification[0].label
                if hand_label == "Left":
                    hand_status["left_hand_detected"] = True
                elif hand_label == "Right":
                    hand_status["right_hand_detected"] = True
                
                # Display the hand type on the frame
                cv2.putText(frame, hand_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Step 4: Display flash detection status on the frame
        if flash_status["left_flash"]:
            print("Flash detected on the left side!")
            cv2.putText(frame, "Left Flash ON", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if flash_status["right_flash"]:
            print("Flash detected on the right side!")
            cv2.putText(frame, "Right Flash ON", (frame.shape[1] // 2 + 10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Check conditions for taking a photo
        if (hand_status["left_hand_detected"] and hand_status["right_hand_detected"]):
            print("Both hands detected; continuing the loop.")
        elif (hand_status["left_hand_detected"] and flash_status["right_flash"]) or \
             (hand_status["right_hand_detected"] and flash_status["left_flash"]):
            print("Taking a photo!")
            text = take_photo(frame)
            print(f"Recognized numbers from photo: {text}")

        # Show the frame with all overlays
        cv2.imshow('Flash and Hand Detection', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Print the final detection statuses
    print("Final Flash Status:", flash_status)
    print("Final Hand Status:", hand_status)

if __name__ == '__main__':
    main()
