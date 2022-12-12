import gradio as gr
import numpy as np
import pandas as pd
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


def SG_process_image(input_img, thresh_r, thresh_g, thresh_b, and_chk, or_chk):
    if and_chk and or_chk:
        raise gr.Error("Only one checkbox should be checked")
    if and_chk:
        thresh_r, thresh_g, thresh_b = [max(thresh_r, thresh_g, thresh_b)] * 3
    if or_chk:
        thresh_r, thresh_g, thresh_b = [min(thresh_r, thresh_g, thresh_b)] * 3
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


def SG2_process_image(input_img):
    img = ColoredImage().read_from_array(input_img)
    vals = img.apply_Otsu_segmentation()
    return input_img, img.data, *vals


def FL_add_noise(input_img):
    img = ColoredImage().read_from_array(input_img).add_noise()
    return input_img, img.data


def FL_apply_filter_with_noise(input_img, matrix):
    img = ColoredImage().read_from_array(input_img).add_noise()
    ret = img.data.copy()

    img.apply_filter(np.array(matrix).astype('float'))
    return ret, img.data


def FL_apply_filter(input_img, matrix):
    img = ColoredImage().read_from_array(input_img)
    ret = img.data.copy()

    img.apply_filter(np.array(matrix).astype('float'))
    return ret, img.data


def FL_apply_median(input_img):
    img = ColoredImage().read_from_array(input_img)
    img.apply_median()
    return input_img, img.data


def FL_apply_median_with_noise(input_img):
    img = ColoredImage().read_from_array(input_img).add_noise()
    ret = img.data.copy()
    img.apply_median()
    return ret, img.data


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
    with gr.Tab("Filters"):
        with gr.Row():
            FL_input_img = gr.Image(label="input_img")
            FL_df = gr.Dataframe(col_count=3, row_count=3, datatype="number", type="numpy", label="matrix")
        with gr.Row():
            FL_noise_btn = gr.Button("Add noise")
            FL_filter_btn = gr.Button("apply filter")
            FL_filter_with_noise_btn = gr.Button("apply filter with noise")
            FL_filter_median_btn = gr.Button("apply median")
            FL_filter_median_with_noise_btn = gr.Button("apply median with noise")
        with gr.Row():
            FL_output1 = gr.Image(label="noisy image").style(height=HEIGHT)
            FL_output2 = gr.Image(label="filtered image").style(height=HEIGHT)
        FL_noise_btn.click(fn=FL_add_noise, inputs=FL_input_img, outputs=[FL_output1, FL_output2])
        FL_filter_with_noise_btn.click(fn=FL_apply_filter_with_noise, inputs=[FL_input_img, FL_df],
                                       outputs=[FL_output1, FL_output2])
        FL_filter_btn.click(fn=FL_apply_filter, inputs=[FL_input_img, FL_df], outputs=[FL_output1, FL_output2])
        FL_filter_median_btn.click(fn=FL_apply_median, inputs=[FL_input_img], outputs=[FL_output1, FL_output2])
        FL_filter_median_with_noise_btn.click(fn=FL_apply_median_with_noise, inputs=[FL_input_img],
                                              outputs=[FL_output1, FL_output2])
        gr.Examples(
            examples=[
                pd.DataFrame(
                    {
                        "0": [1 / 9, 1 / 9, 1 / 9],
                        "1": [1 / 9, 1 / 9, 1 / 9],
                        "2": [1 / 9, 1 / 9, 1 / 9]
                    }
                ),
                pd.DataFrame(
                    {
                        "0": [-1, 0, 1],
                        "1": [-2, 0, 2],
                        "2": [-1, 0, 1]
                    }
                ),
                pd.DataFrame(
                    {
                        "0": [1 / 16, 2 / 16, 1 / 16],
                        "1": [2 / 16, 4 / 16, 2 / 16],
                        "2": [1 / 16, 2 / 16, 1 / 16]
                    }
                )

            ],
            inputs=FL_df,
        )
    with gr.Tab("Segmentation"):
        with gr.Tab("Manual Segmentation"):
            with gr.Row():
                with gr.Column():
                    SG_input_img = gr.Image(label="input image")
                with gr.Column():
                    SG_thresh_r = gr.Slider(label="red threshold", minimum=0, maximum=255, step=1, value=100)
                    SG_thresh_g = gr.Slider(label="green threshold", minimum=0, maximum=255, step=1, value=50)
                    SG_thresh_b = gr.Slider(label="blue threshold", minimum=0, maximum=255, step=1, value=200)
                with gr.Column():
                    SG_and_chk = gr.Checkbox(label="apply AND to thresholds")
                    SG_or_chk = gr.Checkbox(label="apply OR to thresholds")
            SG_greet_btn = gr.Button("Submit")
            with gr.Row():
                SG_output1 = gr.Image(label="input image").style(height=HEIGHT)
                SG_output2 = gr.Image(label="Segmented image").style(height=HEIGHT)

            SG_greet_btn.click(fn=SG_submit,
                               inputs=[SG_input_img, SG_thresh_r, SG_thresh_g, SG_thresh_b, SG_and_chk, SG_or_chk],
                               outputs=[SG_output1, SG_output2])
            sliders = (SG_thresh_r, SG_thresh_b, SG_thresh_g)
            for slider in sliders:
                slider.change(fn=SG_process_image,
                              inputs=[SG_input_img, SG_thresh_r, SG_thresh_g, SG_thresh_b, SG_and_chk, SG_or_chk],
                              outputs=[SG_output1, SG_output2])
        with gr.Tab("Otsu Segmentation"):
            SG2_input_img = gr.Image(label="input_img")
            SG2_button = gr.Button("Submit")
            with gr.Row():
                with gr.Column():
                    SG2_output1 = gr.Image(label="input image").style(height=HEIGHT)
                with gr.Column():
                    SG2_output2 = gr.Image(label="segmented image").style(height=HEIGHT)
                with gr.Column():
                    SG2_val_r = gr.Textbox(label="red threshold")
                    SG2_val_g = gr.Textbox(label="green threshold")
                    SG2_val_b = gr.Textbox(label="blue threshold")
                SG2_button.click(fn=SG2_process_image, inputs=SG2_input_img,
                                 outputs=[SG2_output1, SG2_output2, SG2_val_r, SG2_val_g, SG2_val_b])

# demo = gr.Interface(process_image, "image", ["image","image"])

if __name__ == "__main__":
    demo.launch()
