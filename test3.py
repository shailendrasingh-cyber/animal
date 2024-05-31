import os
import time
import logging
import cv2 as cv
from ultralytics import YOLO
import pygame

MODEL_DIR = 'best.pt'
ALERT_SOUND = 'alert.wav'  # Path to the WAV file to be played when an animal is detected

logging.basicConfig(
    filename="log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def play_alert_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound(ALERT_SOUND)
    alert_sound.play()

def main():
    # Load the YOLO model
    model = YOLO(MODEL_DIR)
    logging.info("Model loaded successfully")

    # Set up the webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error opening webcam.")
        return

    logging.info("Webcam opened successfully")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture image")
            break

        # Predict the frame
        predict = model.predict(frame, conf=0.75)
        
        # Plot boxes
        plotted = predict[0].plot()

        # Check if any boxes are detected
        if len(predict[0].boxes) > 0:
            play_alert_sound()
            logging.info("Animal detected and alert sound played")

        # Display the frame
        cv.imshow('Webcam Frame', plotted)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    logging.info("Webcam released and windows closed")

if __name__ == '__main__':
    main()
 