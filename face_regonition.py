import cv2
from ultralytics import YOLO


def record_video(
	output_filename="output.avi", camera_id=0, fps=20.0, resolution=(640, 480)
):
	"""
	Record video from webcam and save to file

	Parameters:
	output_filename (str): Path to save the recorded video
	camera_id (int): Camera index (default 0 for primary webcam)
	fps (float): Frames per second for recording
	resolution (tuple): Width and height of the video frame
	"""
	# Initialize the webcam
	cap = cv2.VideoCapture(camera_id)

	# Set resolution
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	out = cv2.VideoWriter(output_filename, fourcc, fps, resolution)

	# Load YOLO model
	model = YOLO("yolo11s.pt")  # Load the YOLO model

	print(f"Recording started. Press 'q' to stop recording.")

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break

		# Object detection
		results = model.track(frame, persist=True)
		annotated_frame = results[0].plot()
		cv2.imshow("YOLO11 Tracking", annotated_frame)

		# Break the loop when 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	# Release everything when done
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	print(f"Recording saved to {output_filename}")


# Example usage
if __name__ == "__main__":
	record_video()
