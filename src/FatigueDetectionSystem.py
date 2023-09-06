from scipy.spatial import distance as dist

from config import EAR_THRESHOLD, MAR_THRESHOLD


class FatigueDetectionSystem:
    def __init__(self) -> None:
        """
        Initialize Fatigue Detection System
        """
        pass

    def __eye_aspect_ratio(self, eye):
        """
        Eye Aspect Ratio
        """
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def __mouth_aspect_ratio(self, mouth):
        """
        Mouth Aspect Ratio
        """
        mouth_width = dist.euclidean(mouth[0], mouth[6])
        mouth_height = dist.euclidean(mouth[14], mouth[18])

        mar = mouth_height / mouth_width

        return mar

    def get_avg_ear(self) -> float:
        """
        Get Average Eye Aspect Ratio
        """
        return self.avg_ear[0]

    def get_avg_mar(self) -> float:
        """
        Get Average Mouth Aspect Ratio
        """
        return self.avg_mar[0]

    def run(self, feautres_vector) -> None:
        """
        Run Fatigue Detection System
        """

        ear = self.closed_eyes_model(feautres_vector)

        mar = self.open_mouth_model(feautres_vector)

        fatigue_prediction = self.fatigue_predictor_model(ear, mar)

        return fatigue_prediction

    def closed_eyes_model(self, feautres_vector) -> float:
        """
        Closed Eyes Model
        """

        self.avg_ear = 0 * len(feautres_vector)

        for feature in feautres_vector:
            left_eye = feature[42:48]
            right_eye = feature[36:42]

            left_ear = self.__eye_aspect_ratio(left_eye)
            right_ear = self.__eye_aspect_ratio(right_eye)

            self.avg_ear.append((left_ear + right_ear) / 2.0)

        return self.avg_ear

    def open_mouth_model(self, feautres_vector) -> float:
        """
        Open Mouth Model
        """

        self.avg_mar = 0 * len(feautres_vector)

        for feature in feautres_vector:
            mouth = feature[48:68]

            mar = self.__mouth_aspect_ratio(mouth)

            self.avg_mar.append(mar)

        return self.avg_mar

    def fatigue_predictor_model(self, ear, mar) -> float:
        """
        Fatigue Predictor Model
        """

        fatigue_prediction = 0 * len(ear)

        for i in range(len(ear)):
            if ear[i] < EAR_THRESHOLD and mar[i] > MAR_THRESHOLD:
                fatigue_prediction[i] = 1

        return fatigue_prediction
