import numpy as np
import onnx
import onnxruntime as ort

class ONNXEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        self.__check_model(model_path)
        self.ort_session = self.__session_ort()

    @staticmethod
    def __check_model(model_path):
        print(model_path)
        detector_onnx = onnx.load(model_path)
        onnx.checker.check_model(detector_onnx)

    def __session_ort(self):
        ort_session = ort.InferenceSession(
            self.model_path, providers=self.EP_list
        )
        return ort_session
    
    @staticmethod
    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()
    
    def __call__(self, image):
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(image)}
        probs = self.ort_session.run(None,ort_inputs)[0]
        return probs