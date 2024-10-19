# Lane_Management
This project uses Mediapipe for real-time hand gesture detection and OpenCV for video capture and image processing. Flash triggers photo capture when combined with gestures, and Tesseract OCR extracts digits from images. It also preprocesses images, making it ideal for gesture-based text recognition systems.

#Lane management Using AI integrated cameras
 
 Bangalore is currently facing a significant
 challenge related to lane discipline and
 management. According to recent data,
 approximately 5,000 accidents in the city have
 been attributed to issues with lane
 management. To address this critical problem,
 our team has developed a solution: AI-Integrated
 Lane Management, designed to enhance road
 safety and efficiency.


#Key Features:
*Real-Time Hand Detection:

Uses Mediapipe to detect hands and hand landmarks in real time.
Tracks up to two hands and identifies their left or right orientation.
Draws hand landmarks and shows real-time detection status.
*Flash Detection:

Detects bright light flashes in specific areas of the frame (left or right).
This feature allows the system to trigger actions when a flash is detected from either side.
Optical Character Recognition (OCR):

*Uses Tesseract OCR to recognize text from captured images.
Focuses on recognizing digits, utilizing Tesseract’s specific configuration (--psm 6 and outputbase digits).
Photo Capturing and Processing:

*Captures frames from the webcam and saves them as photos.
Enhances images (sharpening, binarizing) for better OCR results.
Performs OCR on the captured photos to extract and display recognized digits.
Dataset Image Preprocessing:

Reads images from a dataset, resizes them, converts them to grayscale, normalizes, and applies Gaussian blur.
Saves the preprocessed images for further analysis.




#Technologies Used:
*OpenCV:

Used for computer vision tasks such as reading frames from the webcam, image processing, and displaying real-time results on the screen.
Performs image manipulations like resizing, converting to grayscale, blurring, and saving photos.
*Mediapipe:

Provides the hand detection module, which is used to track hand positions and landmarks in real time.
Mediapipe’s Hands solution detects hand gestures and identifies whether the hand is left or right.
Tesseract OCR:

Performs Optical Character Recognition to extract text (digits) from images.
Configured to enhance accuracy for recognizing digits from enhanced images.
*NumPy:

Used for numerical operations such as creating filters (sharpening), applying thresholds, and image matrix manipulation.
Efficiently processes the image arrays for operations like flash detection and image normalization.
*Datetime:

Used for generating timestamps when saving photos, ensuring unique filenames based on the current date and time.
*File I/O:

The os module is used to create directories, manage paths, and handle image file operations, such as saving and loading images.



