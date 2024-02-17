import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import cv2
import glob
from model import modelv1
import time
import onnxruntime as ort

import tensorrt as trt
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输

def preprocess_input(image):
    image /= 255.0
    return image
def out_to_rgb_np(out):
    CLASSES=('ignore','crack', 'spall', 'rebar')
    PALETTE=[[0, 0, 0],[0, 0, 255], [255, 0, 0], [0, 255, 0]]#bgr
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg


def get_miou_png(image,net,device):
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.to(device)
        start_time = time.time()
        pr = net(images)[0]
        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr.argmax(axis=-1)
        end_time = time.time()
        inference_time = end_time - start_time
        
        print('Inference time for', ':', inference_time, 'seconds')
    image =  Image.fromarray(np.uint8(pr))
    
    return image,inference_time

#使用pytorch——GPU推理
def pytorch_gpu():
    model_path = r"F:\Learning_materials\learn_cpp\project\crack_detection\model\best_epoch_weights.pth"
    num_classes     = 4
    name_classes    = ["background","spall", "crack", "rebar"]#/opt/data/private/muti_damage_detection/muti_seg/image/train.txt
    gt_dir          = "/opt/data/private/muti_damage_detection/seg/dataset/"
    pred_dir        = os.path.join(r"F:\Learning_materials\learn_cpp\project\crack_detection\model\result", 'detection-results')
    rgb_dir         = os.path.join(r"F:\Learning_materials\learn_cpp\project\crack_detection\model\result", 'detection_predict')

    # image_ids       = open(os.path.join("/opt/data/private/muti_damage_detection/muti_seg/dataset/","val.txt"),'r').read().splitlines() 
    image_ids = glob.glob(r"F:\Learning_materials\learn_cpp\project\crack_detection\img\*.jpg")

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)
    model = modelv1(4)#torch.cuda.is_available()
    net = model.eval()
    device = torch.device('cuda' if False else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    print('Load model.\n{} model, and classes loaded.'.format(model_path))
    totol_time = 0
    for image_id in tqdm(image_ids):
        
        image_path = image_id.split(",")[0]
        image_id2 = image_path.split("\\")[-1].split(".")[0]
        image       = Image.open(image_path)
        width, height = image.size
        target_size = max(width, height)
        resized_image = image.resize((target_size, target_size))
        cropped_image = resized_image.crop(((target_size - width) // 2, (target_size - height) // 2, (target_size + width) // 2, (target_size + height) // 2))
        image = image.resize((256, 256))
        image,inference_time       = get_miou_png(image,net,device)
        image.save(os.path.join(pred_dir, image_id2 + ".png"))
        image_data = np.array(image)
        image_data = out_to_rgb_np(image_data)
        cv2.imwrite(os.path.join(rgb_dir, image_id2 + ".png"), image_data)
        totol_time += inference_time
    print("mainInference time:", (totol_time)/10, "seconds")

#使用onnxruntime——GPU推理
def onnxruntime_gpu():
    ort_session = ort.InferenceSession('F:\Learning_materials\learn_cpp\project\crack_detection\model\model_softmax_argmax2.onnx',providers=["CUDAExecutionProvider"])
    image_ids = glob.glob(r"F:\Learning_materials\learn_cpp\project\crack_detection\img\*.jpg")

    totol_time = 0
    for i in range(10):
        input_shape = (1, 3, 256, 256)  # 定义输入形状torch.cuda.is_available()
        input_data = np.random.rand(*input_shape).astype(np.float32)
        start_time = time.time()
        output = ort_session.run(None, {'input': input_data})
        end_time = time.time()
        inference_time = end_time - start_time
        totol_time += (end_time - start_time)
        print('Inference time for', ':', inference_time, 'seconds')
    print("mainInference time:", (totol_time)/10, "seconds")

#使用tensorrt推理
def trt_gpu():
    logger = trt.Logger(trt.Logger.WARNING)
    # 创建runtime并反序列化生成engine
    with open("F:\Learning_materials\learn_cpp\project\crack_detection\model\model_softmax_argmax2.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    # 分配CPU锁页内存和GPU显存
    with engine.create_execution_context() as context:
        # 分配CPU锁页内存和GPU显存
        h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
    # 创建cuda流
    stream = cuda.Stream()

    image = cv2.imread(r"F:\Learning_materials\learn_cpp\project\crack_detection\img\rebar001131.jpg")
    # 进行模型所需的图片尺寸、颜色空间和像素值范围转换等预处理操作
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, [2, 0, 1])  # 将通道维度放到最前面
    # 将预处理后的图片数据复制到输入张量中
    np.copyto(h_input, image.ravel())
    # 创建context并进行推理
    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        start_time = time.time()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        end_time = time.time()
        # Return the host output. 该数据等同于原始模型的输出数据
        h_output = h_output.reshape((256, 256))
        print(h_output)
        a = np.unique(h_output)
        print(a)
        h_output[h_output == 0.e+00] = 0
        h_output[h_output == 4.e-45] = 3

        output_shape = h_output.shape
        print(output_shape)

        color_image = out_to_rgb_np(h_output)
        h_output =  Image.fromarray(np.uint8(h_output))
        a = np.unique(h_output)
        print(a)
        cv2.imwrite(os.path.join("F:\Learning_materials\learn_cpp\project\crack_detection\model\output.png"), color_image)
        inference_time = end_time - start_time
        print("Inference time: {:.3f} ms".format(inference_time * 1000))




