import numpy as np
from PIL import ImageDraw, Image
from predictor_onnx import SamPredictorONNX
import copy
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io

# 设置参数
encoder_onnx_path = "../model/edge_sam_3x_encoder.onnx"
decoder_onnx_path = "../model/edge_sam_3x_decoder.onnx"
predictor = SamPredictorONNX(encoder_onnx_path, decoder_onnx_path)
session_state = {
        'coord_list': [],
        'label_list': [],
        'box_list': [],
        'ori_image': None,
        'image_with_prompt': None,
        'feature': None,
        'label': "Positive"
    }
app = FastAPI()

def convert_box(xyxy):
    min_x = min(xyxy[0][0], xyxy[1][0])
    max_x = max(xyxy[0][0], xyxy[1][0])
    min_y = min(xyxy[0][1], xyxy[1][1])
    max_y = max(xyxy[0][1], xyxy[1][1])
    xyxy[0][0] = min_x
    xyxy[1][0] = max_x
    xyxy[0][1] = min_y
    xyxy[1][1] = max_y
    return xyxy

@app.post("/box")
def segment_with_box(inputs:dict):
    image = Image.open(inputs['path'])

    x1, y1, x2, y2 = inputs["x1"], inputs["y1"], inputs["x2"], inputs["y2"]

    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    session_state['box_list'] = [[x1, y1], [x2, y2]]
    session_state['ori_image'] = copy.deepcopy(image)
    session_state['image_with_prompt'] = copy.deepcopy(image)
    session_state['feature'] = predictor.set_image(np.array(image))
    print(f"box_list: {session_state['box_list']}")

    draw = ImageDraw.Draw(session_state['image_with_prompt'])

    image = session_state['image_with_prompt']

    box = convert_box(session_state['box_list'])
    xy = (box[0][0], box[0][1], box[1][0], box[1][1])
    draw.rectangle(
        xy,
        outline=box_color,
        width=box_outline
    )

    box_np = np.array(box)
    point_coords = box_np.reshape(2, 2)[None]
    point_labels = np.array([2, 3])[None]
    masks, _, _ = predictor.predict(
        features=session_state['feature'],
        point_coords=point_coords,
        point_labels=point_labels,
    )
    annotations = masks[:, 0, :, :]

    # 提取原图中的mask部分
    mask_np = annotations[0]  # 获取第一个mask
    extracted_region = np.array(image).copy()
    extracted_region[mask_np == 0] = 255  # 将非mask区域设置为白色

    # 转换为PIL图像
    result_image = Image.fromarray(extracted_region, 'RGB')

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='JPEG')  # 根据需要的格式保存
    img_byte_arr.seek(0)  # 重置字节流位置

    # 返回字节流
    return StreamingResponse(img_byte_arr, media_type='image/jpeg')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
