import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
path_images_from_camera = "data/data_faces_from_camera/"
shape_predictor_path = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
face_reco_model_path = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"

# Initialize Dlib components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

def return_128d_features(path_img):
    """Extract 128D features from a single image."""
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)

    logging.info("Processing image: %s", path_img)

    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        return np.array(face_descriptor)
    else:
        logging.warning("No face detected in image: %s", path_img)
        return None

def return_features_mean_personX(path_face_personX):
    """Compute the mean of 128D features for all images of a person."""
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)

    if photos_list:
        for photo in photos_list:
            photo_path = os.path.join(path_face_personX, photo)
            features_128d = return_128d_features(photo_path)
            if features_128d is not None:
                features_list_personX.append(features_128d)

    if features_list_personX:
        features_mean_personX = np.mean(features_list_personX, axis=0)
    else:
        features_mean_personX = np.zeros(128)
    return features_mean_personX

def main():
    person_list = sorted(os.listdir(path_images_from_camera))

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label"] + list(range(128)))  # Write header row with feature names
        for person in person_list:
            person_path = os.path.join(path_images_from_camera, person)
            features_mean_personX = return_features_mean_personX(person_path)

            person_name = person.split('_', 2)[-1] if '_' in person else person
            features_row = [person_name] + features_mean_personX.tolist()
            writer.writerow(features_row)

            logging.info("Processed features for %s", person_name)

    logging.info("Saved all features to data/features_all.csv")

if __name__ == '__main__':
    main()
