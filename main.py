import gradio as gr
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from ImagePGM import ImagePGM

HEIGHT = 400
SHAPE = (HEIGHT, HEIGHT)


def EQ_process_image(input_img):
    PIL_image = Image.fromarray(np.uint8(input_img)).convert('L')
    img_arr = np.asarray(PIL_image)
    img = ImagePGM()
    img.read_from_array(img_arr)
    img.apply_map(img.equalization_array())
    return img_arr, np.array(img.data)


def LT_process_image(input_img, p1x, p1y, p2x, p2y):
    # if p1x >= p2x:
    #     raise gr.Error("point1 x-coordinates must be smaller than point2 x-coordinates")

    PIL_image = Image.fromarray(np.uint8(input_img)).convert('L')
    img_arr = np.asarray(PIL_image)
    img = ImagePGM()
    img.read_from_array(img_arr)

    map = img.piecewise_linear((p1x, p1y), (p2x, p2y))
    img.apply_map(map)

    fig = go.Figure(data=go.Scatter(x=list(range(256)), y=map))
    return fig, img_arr, np.array(img.data)


imageUploaded = False


def LT_submit(*kwargs):
    global imageUploaded
    imageUploaded = True
    return LT_process_image(*kwargs)


def LT_change(*args):
    if not imageUploaded:
        raise gr.Error("You must upload an image first")
    return LT_process_image(*args)


with gr.Blocks() as demo:
    with gr.Tab("Equalization"):
        input_img = gr.Image(label="input_img")
        greet_btn = gr.Button("Submit")
        with gr.Row():
            output1 = gr.Image(label="grayscale image").style(height=HEIGHT)
            output2 = gr.Image(label="equalized image").style(height=HEIGHT)
        greet_btn.click(fn=EQ_process_image, inputs=input_img, outputs=[output1, output2])
    with gr.Tab("Linear Transformation"):
        with gr.Row():
            LT_input_img = gr.Image(label="input image")
            with gr.Blocks():
                with gr.Column():
                    p1_x = gr.Slider(label="point1 x-coordinate", minimum=0, maximum=255, step=1, value=100)
                    p1_y = gr.Slider(label="point1 y-coordinate", minimum=0, maximum=255, step=1, value=50)
                with gr.Column():
                    p2_x = gr.Slider(label="point2 x-coordinate", minimum=0, maximum=255, step=1, value=150)
                    p2_y = gr.Slider(label="point2 y-coordinate", minimum=0, maximum=255, step=1, value=200)

        LT_greet_btn = gr.Button("Submit")
        with gr.Row():
            LT_plot = gr.Plot()
            LT_output1 = gr.Image(label="grayscale image").style(height=HEIGHT)
            LT_output2 = gr.Image(label="Linearly Transformed image").style(height=HEIGHT)

        LT_greet_btn.click(fn=LT_submit, inputs=[LT_input_img, p1_x, p1_y, p2_x, p2_y],
                           outputs=[LT_plot, LT_output1, LT_output2])
        sliders = (p1_x, p1_y, p2_x, p2_y)
        for slider in sliders:
            slider.change(fn=LT_change, inputs=[LT_input_img, p1_x, p1_y, p2_x, p2_y],
                          outputs=[LT_plot, LT_output1, LT_output2])

# demo = gr.Interface(process_image, "image", ["image","image"])

if __name__ == "__main__":
    demo.launch()
