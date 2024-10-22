#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yangst
@email: yangst@knowdee.com
@time: 2024/9/29 17:55
"""

import pandas as pd
import streamlit as st
import requests
from PIL import Image
import json
import numpy as np
from streamlit_drawable_canvas import st_canvas
import os
import random

# 全局
text_embeddings = dict()
image_embeddings = dict()

image_captions_generated_by_cpm = dict()

os.makedirs('/data2/zhangdh/pictureFinding/tmp', exist_ok=True)
clip_embedding = np.load("/data2/zhangdh/pictureFinding/embedding/clip.npy")
e5_embedding = np.load("/data2/zhangdh/pictureFinding/embedding/e5.npy")

def box(x1, y1, x2, y2, path):

    data = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "path": path
    }
    url = "http://172.70.10.53:8008/box"

    res = requests.post(url, json=data)
    # with open(output_path, "wb") as f:
    #     f.write(res.content)
    return res.content


def image_index(query):
    data = {"query": query}
    url = "http://172.70.10.53:15003/imageSearch"

    res = requests.post(url, json=data).text
    res = json.loads(res)
    return res["data"]["images"]


def pic2embed(image_path):
    data = {"image": image_path}
    url = "http://172.70.10.53:15002/picture2embedding"

    res = requests.post(url ,json=data).text
    res =json.loads(res)
    return res['data']


# image命名
def gen_name(prefix="upload_"):
    while True:
        random_number = random.randint(10000, 99999)
        filename = f"{prefix}_{random_number}"
        if not os.path.exists(filename):
            return filename


def update_corpus():
    ...


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    pages = {
        "Image Search": search_image,
        "Image Upload": upload_image
    }
    page = st.sidebar.selectbox("Let's get started", options=list(pages.keys()))
    pages[page]()


def search_image():
    # st.markdown(
    #     """
    # Welcome to the demo of [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas).
    #
    # On this site, you will find a full use case for this Streamlit component, and answers to some frequently asked questions.
    #
    # :pencil: [Demo source code](https://github.com/andfanilo/streamlit-drawable-canvas-demo/)
    # """
    # )
    st.markdown(
        """
    Amazing Images

    * You just need to enter a description, and we will be responsible for finding similar images
    * If you want to upload a new image, please click the button [Image Upload]
   
    """
    )

    query = st.text_input("Now please tell me what kind of image you want to search for",
                  value="",
                  max_chars=None,
                  key=None,
                  type="default",
                  help=None,
                  autocomplete=None,
                  on_change=None,
                  placeholder=None)

    images = image_index(query)

    image_1, image_2 = st.columns(2)

    if query:

        with image_1:
            image = Image.open(images[0])
            new_image = image.resize((350, 250))
            st.image(new_image)
            # st.image(images[0], width=300)

        with image_2:
            image = Image.open(images[1])
            new_image = image.resize((350, 250))
            st.image(new_image)


def upload_image():
    
    # print(np.shape(e5_embedding))
    # print(np.shape(clip_embedding))
    image_name = gen_name()
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Upload images in the sidebar
    * Outline the entities you want to extract on the canvas
    * Choose whether to save the extracted entities to the database
    """
    )
    
    # Specify canvas parameters in application
    drawing_mode = "rect"
    stroke_width = 2
    
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = True
    
    # bg_width, bg_height = 512, 512
    if bg_image:
        _image = Image.open(bg_image)
        bg_width, bg_height = _image.width, _image.height
        bg_width, bg_height = _image.width, _image.height
        scale_width, scale_height = bg_width / 600, bg_height / 400
        
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            drawing_mode=drawing_mode,
            point_display_radius=0,
            display_toolbar=True,
            key="full_app",
        )
        s1 ,s2 = st.columns(2)
        with s1:
            if st.button('Click to save original image'):
                _image.save(f"/data2/zhangdh/pictureFinding/pictures/{image_name}.jpg")
                embed = pic2embed(f"/data2/zhangdh/pictureFinding/pictures/{image_name}.jpg")

                discription = embed['description']

                clip = embed['embedding_clip']
                clip = np.array([clip])
                clip_embedding = np.vstack((clip_embedding, clip))

                e5 = embed['embedding_e5']
                e5 = np.array([e5])
                e5_embedding = np.vstack((e5_embedding, e5))
                # print(np.shape(e5_embedding))
                # print(np.shape(clip_embedding))
                np.save('/data2/zhangdh/pictureFinding/embedding/clip.npy', clip_embedding)
                np.save('/data2/zhangdh/pictureFinding/embedding/e5.npy', e5_embedding)

                with open('/data2/zhangdh/pictureFinding/embedding/description.txt', 'a', encoding='utf-8') as file:
                    file.write(discription)
                with open('/data2/zhangdh/pictureFinding/embedding/images.txt', 'a', encoding='utf-8',newline='\n') as file:
                    file.write(f'/data2/zhangdh/pictureFinding/pictures/{image_name}.jpg'+'\n')
        # Do something interesting with the image data and paths
        if canvas_result.json_data is not None:
            mask_data = [{"left": _["left"], "top": _["top"], "width": _["width"], "height": _["height"]} for _ in
                        canvas_result.json_data["objects"]]
            objects = pd.json_normalize(mask_data)
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            num = len(canvas_result.json_data["objects"])
            # 访问最后一行
            if num >=1:
                last_row = canvas_result.json_data["objects"][-1]
                x1,y1,x2,y2 = int(last_row['left'])*scale_width, int(last_row['top'])*scale_height, (int(last_row["left"])+int(last_row['width']))*scale_width, (int(last_row["top"])+int(last_row['height']))*scale_height
                _image.save('/data2/zhangdh/pictureFinding/tmp/tmp.jpg')
                image_path = f"/data2/zhangdh/pictureFinding/tmp/tmp_crop.jpg"
                img = box(x1, y1, x2, y2, f"/data2/zhangdh/pictureFinding/tmp/tmp.jpg")
                if canvas_result.image_data is not None:
                    # tmp image 目录
                    with open(image_path, "wb" ) as f:
                        f.write(img)
                    with s2:
                        if st.button('Click to save cropped image'):
                            # 保存到库里的目录
                            save_path = f"/data2/zhangdh/pictureFinding/pictures/{image_name}_crop.jpg"
                            with open(save_path, "wb") as f:
                                f.write(img)
                            embed = pic2embed(save_path)

                            discription = embed['description']

                            clip = embed['embedding_clip']
                            clip = np.array([clip])
                            clip_embedding = np.vstack((clip_embedding, clip))

                            e5 = embed['embedding_e5']
                            e5 = np.array([e5])
                            e5_embedding = np.vstack((e5_embedding, e5))
                            # print(np.shape(e5_embedding))
                            # print(np.shape(clip_embedding))
                            np.save('/data2/zhangdh/pictureFinding/embedding/clip.npy', clip_embedding)
                            np.save('/data2/zhangdh/pictureFinding/embedding/e5.npy', e5_embedding)

                            with open('/data2/zhangdh/pictureFinding/embedding/description.txt', 'a', encoding='utf-8') as file:
                                file.write(discription)
                            with open('/data2/zhangdh/pictureFinding/embedding/images.txt', 'a', encoding='utf-8',newline='\n') as file:
                                file.write(f'/data2/zhangdh/pictureFinding/pictures/{image_name}_crop.jpg'+'\n')

                    st.image(image_path)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Knowdee Edge MM", page_icon="../img/4.jpg"
    )
    st.title("Knowdee Edge MM")
    st.sidebar.subheader("Functions")
    main()
