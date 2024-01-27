import cv2
import onnxruntime as ort
import onnxruntime.backend

model_path = "models/enet.onnx"
ort_sess = ort.InferenceSession(model_path)

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CV2 - gpu: CUDA is available")
else:
    print("CV2 - gpu: CUDA is not available")


print("Onnxruntime-gpu: ", ort.get_device())

