import cv2
import numpy as np
from openvino import Core

# -------------------- Load OpenVINO --------------------
core = Core()

# -------------------- Model Paths --------------------
face_model_path = r"face-detection-retail-0005.xml"
age_gender_model_path = r"age-gender-recognition-retail-0013.xml"

# -------------------- Read Models --------------------
face_model = core.read_model(face_model_path)
age_gender_model = core.read_model(age_gender_model_path)

# -------------------- Compile Models --------------------
face_compiled = core.compile_model(face_model, "CPU")
age_gender_compiled = core.compile_model(age_gender_model, "CPU")

# -------------------- Input/Output Layers ----------------q----
face_input_layer = face_compiled.input(0)
face_output_layer = face_compiled.output(0)

age_gender_input_layer = age_gender_compiled.input(0)
age_gender_outputs = age_gender_compiled.outputs

# -------------------- Gender Output Identification --------------------
# Usually:
# output 0 -> age_conv3
# output 1 -> prob (gender)
# But to be safe, we inspect names

age_output = None
gender_output = None

for out in age_gender_outputs:
    name = out.get_any_name()
    print("Output Layer:", name)
    if "age" in name.lower():
        age_output = out
    elif "prob" in name.lower() or "gender" in name.lower():
        gender_output = out

# Fallback if names not detected
if age_output is None:
    age_output = age_gender_outputs[0]
if gender_output is None:
    gender_output = age_gender_outputs[1]

# -------------------- Webcam --------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Webcam not opened")
    exit()

print("✅ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # -------------------- Face Detection Preprocessing --------------------
    face_input_image = cv2.resize(frame, (300, 300))
    face_input_image = face_input_image.transpose((2, 0, 1))  # HWC -> CHW
    face_input_image = np.expand_dims(face_input_image, axis=0)

    # -------------------- Run Face Detection --------------------
    face_results = face_compiled([face_input_image])[face_output_layer]

    # Output shape: [1, 1, N, 7]
    for detection in face_results[0][0]:
        confidence = float(detection[2])

        if confidence > 0.5:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            # Safe coordinates
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            face_crop = frame[ymin:ymax, xmin:xmax]

            if face_crop.size == 0:
                continue

            # -------------------- Age-Gender Preprocessing --------------------
            ag_input = cv2.resize(face_crop, (62, 62))
            ag_input = ag_input.transpose((2, 0, 1))  # HWC -> CHW
            ag_input = np.expand_dims(ag_input, axis=0)

            # -------------------- Run Age-Gender Model --------------------
            ag_results = age_gender_compiled([ag_input])

            age_pred = ag_results[age_output][0][0][0][0] * 100
            gender_pred = ag_results[gender_output][0]

            gender = "Female" if gender_pred[0] > gender_pred[1] else "Male"

            # -------------------- Draw Results --------------------
            label = f"{gender}, Age: {int(age_pred)}"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("OpenVINO Age-Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()