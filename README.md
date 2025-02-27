# Hand Steering Control

This project utilizes OpenCV and MediaPipe to recognize hand gestures and convert them into vehicle control signals (steering, acceleration, braking). These signals are transmitted via the UDP protocol to Unity or another simulation system.

## Project Description
The system allows users to control a vehicle using hand gestures via a webcam:

- **Steering Wheel Rotation:** The angle between both hands is calculated to determine the steering direction.
- **Acceleration/Braking:** The distance between the hands determines whether the vehicle moves forward or backward.
- **Automatic Calibration:** The user clenches their fist to calibrate the initial hand distance.
- **UDP Communication:** Control data is sent via UDP sockets to an external application (e.g., Unity) for vehicle simulation.

## Key Features
- Hand gesture recognition using MediaPipe Hands.
- Calculation of steering angle and acceleration/braking levels based on hand position.
- Interactive UI displaying FPS, hand status, steering angle, and acceleration/braking level.
- Signal smoothing for more stable control.
- UDP data transmission to external applications.

## Requirements
### Hardware
- A webcam (built-in or external).
- A computer with at least a mid-range configuration.

### Software
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Socket programming (for UDP communication)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/hand-steering-control.git
   cd hand-steering-control
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python mediapipe numpy
   ```

## Usage
1. Run the main script:
   ```sh
   python main.py
   ```
2. Ensure the webcam is connected and positioned correctly.
3. Use hand gestures to control the vehicle.

## Future Improvements
- Support for additional gestures.
- Integration with more simulation platforms.
- Machine learning-based gesture recognition for improved accuracy.

## License
This project is licensed under the MIT License.

---
**Author:** Your Name  
**Contact:** your.email@example.com
