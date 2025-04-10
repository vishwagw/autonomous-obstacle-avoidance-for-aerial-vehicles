# live detector mode for drone cam or webcam:
import cv2
import numpy as np
import time

def obstacle_detection():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        return
    
    height, width = frame.shape[:2]
    
    # Define detection zone parameters (adjustable)
    zone_width_percent = 0.3  # Width of detection zone as percentage of frame width
    zone_width = int(width * zone_width_percent)
    left_boundary = int((width - zone_width) / 2)
    right_boundary = int((width + zone_width) / 2)
    
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    
    # Process frames
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw detection zone lines
        cv2.line(vis_frame, (left_boundary, 0), (left_boundary, height), (0, 255, 0), 2)
        cv2.line(vis_frame, (right_boundary, 0), (right_boundary, height), (0, 255, 0), 2)
        
        # Extract region of interest (ROI)
        roi = frame[:, left_boundary:right_boundary]
        
        # Apply background subtraction to detect moving objects
        fg_mask = bg_subtractor.apply(roi)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process detected obstacles
        obstacle_detected = False
        obstacle_distance = "Unknown"
        obstacle_size = "Unknown"
        
        for contour in contours:
            # Filter out small contours (noise)
            if cv2.contourArea(contour) < 500:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box on visualization frame (adjust x-coordinate to match original frame)
            cv2.rectangle(vis_frame, (x + left_boundary, y), (x + w + left_boundary, y + h), (0, 0, 255), 2)
            
            # Mark as obstacle detected
            obstacle_detected = True
            
            # Estimate distance (basic approximation based on y-coordinate)
            # Lower y means object is farther away (top of frame)
            # Higher y means object is closer (bottom of frame)
            relative_distance = 1 - (y / height)
            obstacle_distance = f"~{int(relative_distance * 100)}% away"
            
            # Estimate size
            relative_size = (w * h) / (width * height)
            if relative_size < 0.01:
                obstacle_size = "Small"
            elif relative_size < 0.05:
                obstacle_size = "Medium"
            else:
                obstacle_size = "Large"
            
            # Display text on frame
            cv2.putText(vis_frame, f"OBSTACLE: {obstacle_size}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Distance: {obstacle_distance}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display no obstacle message if none detected
        if not obstacle_detected:
            cv2.putText(vis_frame, "No obstacles detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display zone info
        cv2.putText(vis_frame, "Detection Zone", (left_boundary + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frames
        cv2.imshow('Obstacle Detection', vis_frame)
        cv2.imshow('Detection Mask', fg_mask)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    obstacle_detection()