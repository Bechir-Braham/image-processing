import gradio as gr
import numpy as np
import plotly.graph_objects as go

from ColoredImage import ColoredImage

HEIGHT = 400
SHAPE = (HEIGHT, HEIGHT)


def EQ_process_image(input_img):
    img = ColoredImage().read_from_array(input_img).apply_equalization()
    return input_img, img.data


def LT_process_image(input_img, p1x, p1y, p2x, p2y):
    if p1x >= p2x:
        raise gr.Error("point1 x-coordinates must be smaller than point2 x-coordinates")

    img = ColoredImage().read_from_array(input_img)
    map = img.apply_linear_transformation(p1x, p1y, p2x, p2y)

    fig = go.Figure(data=go.Scatter(x=list(range(256)), y=map))
    return fig, input_img, np.array(img.data)


LT_imageUploaded = False


def LT_submit(*kwargs):
    global LT_imageUploaded
    LT_imageUploaded = True
    return LT_process_image(*kwargs)


def LT_change(*args):
    if not LT_imageUploaded:
        raise gr.Error("You must upload an image first")
    return LT_process_image(*args)


def SG_process_image(input_img, thresh_r, thresh_g, thresh_b):
    img = ColoredImage().read_from_array(input_img).apply_threshold(thresh_r, thresh_g, thresh_b)
    return input_img, img.data


SG_imageUploaded = False


def SG_submit(*kwargs):
    global SG_imageUploaded
    SG_imageUploaded = True
    return SG_process_image(*kwargs)


def SG_change(*args):
    if not SG_imageUploaded:
        raise gr.Error("You must upload an image first")
    return SG_process_image(*args)


with gr.Blocks() as demo:
    with gr.Tab("Equalization"):
        input_img = gr.Image(label="input_img")
        greet_btn = gr.Button("Submit")
        with gr.Row():
            output1 = gr.Image(label="input image").style(height=HEIGHT)
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
            LT_output1 = gr.Image(label="input image").style(height=HEIGHT)
            LT_output2 = gr.Image(label="Linearly Transformed image").style(height=HEIGHT)

        LT_greet_btn.click(fn=LT_submit, inputs=[LT_input_img, p1_x, p1_y, p2_x, p2_y],
                           outputs=[LT_plot, LT_output1, LT_output2])
        sliders = (p1_x, p1_y, p2_x, p2_y)
        for slider in sliders:
            slider.change(fn=LT_change, inputs=[LT_input_img, p1_x, p1_y, p2_x, p2_y],
                          outputs=[LT_plot, LT_output1, LT_output2])
    with gr.Tab("segmentation"):
        with gr.Row():
            SG_input_img = gr.Image(label="input image")
            SG_thresh_r = gr.Slider(label="red threshold", minimum=0, maximum=255, step=1, value=100)
            SG_thresh_g = gr.Slider(label="green threshold", minimum=0, maximum=255, step=1, value=50)
            SG_thresh_b = gr.Slider(label="blue threshold", minimum=0, maximum=255, step=1, value=200)
        SG_greet_btn = gr.Button("Submit")
        with gr.Row():
            SG_output1 = gr.Image(label="input image").style(height=HEIGHT)
            SG_output2 = gr.Image(label="Linearly Transformed image").style(height=HEIGHT)

        SG_greet_btn.click(fn=SG_submit, inputs=[SG_input_img, SG_thresh_r, SG_thresh_g, SG_thresh_b],
                           outputs=[SG_output1, SG_output2])
        sliders = (SG_thresh_r, SG_thresh_b, SG_thresh_g)
        for slider in sliders:
            slider.change(fn=SG_process_image, inputs=[SG_input_img, SG_thresh_r, SG_thresh_g, SG_thresh_b],
                          outputs=[SG_output1, SG_output2])

# demo = gr.Interface(process_image, "image", ["image","image"])

if __name__ == "__main__":
    demo.launch()
