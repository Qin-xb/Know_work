# edge_mm

## edge_sam
### 1、edgesam_api.py 
启动api的主服务文件，默认的模型加载路径为：

encoder_onnx_path = "../model/edge_sam_3x_encoder.onnx"

decoder_onnx_path = "../model/edge_sam_3x_decoder.onnx"

### 2、predictor_onnx.py 
加载sam模型,使用cpu加速

### 3、transforms.py
将图像转换为模型预期的形式输入



