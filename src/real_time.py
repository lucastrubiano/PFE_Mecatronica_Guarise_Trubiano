import cv2

from FacialFeaturesExtractionSystem import FacialFeaturesExtractionSystem
from FatigueDetectionSystem import FatigueDetectionSystem
from AlertSystem import AlertSystem


if __name__ == "__main__":
    # Initialize the 3 subsystems
    facial_features_extraction = FacialFeaturesExtractionSystem()
    fatigue_detection_system = FatigueDetectionSystem()
    alert_system = AlertSystem()

    # Start video capture
    video_frame = cv2.VideoCapture(0)

    # Run the 3 subsystems in cascade
    while True:
        ret, frame = video_frame.read()
        if not ret:
            break

        landmarks = facial_features_extraction.run(frame)

        # Show landmarks points on frame
        for shape in landmarks:
            for x, y in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        fatigue_prediction = fatigue_detection_system.run(landmarks)

        cv2.putText(
            frame,
            "EAR: {:.2f}".format(fatigue_detection_system.get_avg_ear()),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "MAR: {:.2f}".format(fatigue_detection_system.get_avg_mar()),
            (300, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        alert_result = alert_system.run(fatigue_prediction)

        # Print fatigue detected text on frame
        if alert_result:
            cv2.putText(
                frame,
                "Fatigue Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Show frame
        cv2.imshow("Fatigue Detector", frame)

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture and close all windows when exiting
    cv2.destroyAllWindows()
    video_frame.release()
