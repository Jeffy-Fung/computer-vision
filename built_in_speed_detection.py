import cv2
from ultralytics import solutions
import os


def record_video(
	output_filename="outputs/output.mp4", camera_id=0, fps=20.0, resolution=(640, 480)
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
	
	# Create output directory if it doesn't exist
	os.makedirs(os.path.dirname(output_filename), exist_ok=True)
	out = cv2.VideoWriter(output_filename, fourcc, fps, resolution)

	# Load YOLO model
	speedestimator = solutions.SpeedEstimator(
		region=[(240, 0), (240, 640)],
		model="yolo11n.pt",
		show=True,
		classes=[0],  # Track only person class
		names={0: "person"},  # Map class ID to name
		view_img=True,  # Show the video stream
		line_thickness=2,  # Thickness of the detection line
		region_thickness=2,  # Thickness of the region line
	)

	print(f"Recording started. Press 'q' to stop recording.")

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("Failed to grab frame")
			break

		# Process frame with speed estimation
		results = speedestimator.process(frame)
		# Write the processed frame to the output file
		out.write(results.plot_im)

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
