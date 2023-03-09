import argparse

import onnx
import onnxruntime
import torch
from easyocr import easyocr
import numpy as np

def verify_exported(model_onnx_path):
    detector_onnx = onnx.load(model_onnx_path)
    onnx.checker.check_model(detector_onnx)
    print(f"Model Inputs:\n {detector_onnx.graph.input}\n{'*'*80}")
    print(f"Model Outputs:\n {detector_onnx.graph.output}\n{'*'*80}")

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

def inference_validation_detector(model_onnx_path, dummy_input):
    ort_session = onnxruntime.InferenceSession(model_onnx_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    y_onnx_out, feature_onnx_out = ort_session.run(None, ort_inputs)
    return y_onnx_out, feature_onnx_out

def inference_validation_recognizer(model_onnx_path, dummy_input):
    ort_session = onnxruntime.InferenceSession(model_onnx_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    y_onnx_out = ort_session.run(None, ort_inputs)[0]
    return y_onnx_out

def export_onnx_detection(ocr_reader, in_shape_detection,detector_onnx_save_path, opset_version):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dummy_input = torch.rand(in_shape_detection)
        dummy_input = dummy_input.to(device)
        # forward pass
        with torch.no_grad():
            y_torch_out, feature_torch_out = ocr_reader.detector(dummy_input)
            input_names = ['image_input']
            output_names = ['output']
            dynamic_axes = {'image_input': {0: 'batch_size', 2: "height", 3: "width"},
                            'output': {0: 'batch_size', 1: "dim1", 2: "dim2"}
                            }
            torch.onnx.export(
                    ocr_reader.detector,
                    dummy_input,
                    detector_onnx_save_path,
                    export_params = True,
                    do_constant_folding = True,
                    opset_version = opset_version,
                    input_names = input_names,
                    output_names = output_names,
                    dynamic_axes = dynamic_axes,
                    verbose=False
                )
        # verify exported onnx model
        verify_exported(detector_onnx_save_path)

        # onnx inference validation
        y_onnx_out, feature_onnx_out = inference_validation_detector(detector_onnx_save_path, dummy_input)
        print(f"torch outputs: y_torch_out.shape={y_torch_out.shape} feature_torch_out.shape={feature_torch_out.shape}")
        print(f"onnx outputs: y_onnx_out.shape={y_onnx_out.shape} feature_onnx_out.shape={feature_onnx_out.shape}")
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            to_numpy(y_torch_out), y_onnx_out, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(
            to_numpy(feature_torch_out), feature_onnx_out, rtol=1e-03, atol=1e-05)

        print(f"Model exported to {detector_onnx_save_path} and tested with ONNXRuntime, and the result looks good!")


def export_onnx_recognizer(ocr_reader, in_shape_recognizer,recognizer_onnx_save_path, opset_version):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_list = [0]*10
    dummy_text_for_pred = torch.Tensor(dummy_list).to(torch.long).to(device)
    dummy_image = torch.rand(in_shape_recognizer).to(device)
    dummy_input = (dummy_image, dummy_text_for_pred)

    with torch.no_grad():
        y_torch_out = ocr_reader.recognizer(dummy_image, dummy_text_for_pred)

        input_names = ["image_input", "text_input"]
        output_names = ["output"]
        dynamic_axes = {'image_input' : {3 : 'batch_size_1_1'}}

        torch.onnx.export(
                ocr_reader.recognizer,
                dummy_input,
                recognizer_onnx_save_path,
                export_params=True,
                opset_version=opset_version,
                input_names= input_names,
                output_names = output_names,
                dynamic_axes = dynamic_axes
            )
        
        # verify exported onnx model
        verify_exported(recognizer_onnx_save_path)
        # onnx inference validation 
        y_onnx_out = inference_validation_recognizer(recognizer_onnx_save_path, dummy_image)
        print(f"torch outputs: y_torch_out.shape={y_torch_out.shape}")
        print(f"onnx outputs: y_onnx_out.shape={y_onnx_out.shape}")

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            to_numpy(y_torch_out), y_onnx_out, rtol=1e-03, atol=1e-05)

        print(f"Model exported to {recognizer_onnx_save_path} and tested with ONNXRuntime, and the result looks good!")

def export_onnx_model(
        detector_onnx_save_path,
        recognizer_onnx_save_path,
        in_shape_detection=[1, 3, 608, 800],
        in_shape_recognizer=[1, 1, 64, 192],
        lang_list=["en"],
        recog_network = 'english_g2',
        model_storage_directory=None,
        user_network_directory=None,
        download_enabled=True,
        dynamic=True,
        device="cpu",
        quantize=True,
        detector=True,
        recognizer=True,
        opset_version=11
    ):
    if dynamic is False:
        print('WARNING: it is recommended to use -d dynamic flag when exporting onnx')
    ocr_reader = easyocr.Reader(lang_list,
                                recog_network=recog_network,
                                gpu=False if device == "cpu" else True,
                                detector=detector,
                                recognizer=recognizer,
                                quantize=quantize,
                                model_storage_directory=model_storage_directory,
                                user_network_directory=user_network_directory,
                                download_enabled=download_enabled)

    # exporting detector if selected
    if detector:
        export_onnx_detection(ocr_reader, in_shape_detection, detector_onnx_save_path, opset_version)

    if recognizer:
        export_onnx_recognizer(ocr_reader, in_shape_recognizer, recognizer_onnx_save_path, opset_version)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang_list',
                        nargs='+', type=str,
                        default=["en"],
                        help='-l en ch_sim ... (language lists for easyocr)')
    parser.add_argument('-ds', '--detector_onnx_save_path', type=str,
                        default="detector_craft.onnx",
                        help="export detector onnx file path ending in .onnx" +
                        "Do not pass in this flag to avoid exporting detector")
    parser.add_argument('-rs', '--recognizer_onnx_save_path', type=str,
                        default="english_g2.onnx",
                        help="export recognizer onnx file path ending in .onnx" +
                        "Do not pass in this flag to avoid exporting recognizer")
    parser.add_argument('-rn', '--recog_network', type=str,
                        default="english_g2",
                        help=" Instead of standard mode, you can choose your own recognition network")
    parser.add_argument('-d', '--dynamic',
                        action='store_true',
                        help="Dynamic  input output shapes for detector")
    parser.add_argument('-isd', '--in_shape_detection',
                        nargs='+', type=int,
                        default=[1, 3, 608, 800],
                        help='-is 1 3 608 800 (bsize, channel, height, width)')
    parser.add_argument('-isr', '--in_shape_recognizer',
                        nargs='+', type=int,
                        default=[1, 1, 64, 192],
                        help='-is 1 1 64 192 (bsize, channel, height, width)')
    parser.add_argument('-m', '--model_storage_directory', type=str,
                        help="model storage directory for craft model")
    parser.add_argument('-u', '--user_network_directory', type=str,
                        help="user model storage directory")
    parser.add_argument('-o', '--opset_version', type=int,
                        default=11,
                        help="opset version onnx")
    
    args = parser.parse_args()
    dpath = args.detector_onnx_save_path
    rpath = args.recognizer_onnx_save_path
    args.detector_onnx_save_path = None if dpath == "None" else dpath
    args.recognizer_onnx_save_path = None if rpath == "None" else rpath

    if len(args.in_shape_detection) != 4 and len(args.in_shape_recognizer) != 4:
        raise ValueError(
            f"Input shape must have four values (bsize, channel, height, width) eg. 1 3 608 800")
    return args


def main():
    args = parse_args()
    export_onnx_model(detector_onnx_save_path=args.detector_onnx_save_path,
                    recognizer_onnx_save_path=args.recognizer_onnx_save_path,
                    in_shape_detection=args.in_shape_detection,
                    in_shape_recognizer=args.in_shape_recognizer,
                    lang_list=args.lang_list,
                    recog_network=args.recog_network,
                    model_storage_directory=args.model_storage_directory,
                    user_network_directory=args.user_network_directory,
                    dynamic=args.dynamic,
                    opset_version=args.opset_version)


if __name__ == "__main__":
    main()
