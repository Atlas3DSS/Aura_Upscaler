import uuid
import gradio as gr
from aura_sr import AuraSR
from PIL import Image
import requests
from io import BytesIO

# Initialize the AuraSR model
aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR")


def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)


def process_image(input_image):
    if input_image is None:
        return None

    # Resize the input image
    #get image size
    width, height = input_image.size
    #resize image without losing aspect ratio)
    if width > height:
        input_image = input_image.resize((512, int(512 * height / width)))
    else:
        input_image = input_image.resize((int(512 * width / height), 512))

    # Upscale the image using AuraSR
    upscaled_image = aura_sr.upscale_4x(input_image)
    #save the image with uuid
    filename = f"{str(uuid.uuid4())}.png"
    upscaled_image.save(filename)
    print(f"Saved image: {filename}")

    return [input_image, upscaled_image]


with gr.Blocks() as demo:
    gr.Markdown("""
    # AuraSR Image Upscaler

    Upload an image and the AuraSR model will upscale it by 4x.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            process_btn = gr.Button("Upscale Image")
        with gr.Column():
            output_image = gr.Image(label="Upscaled Image", type="pil")

    process_btn.click(fn=process_image,
                      inputs=[input_image],
                      outputs=[input_image, output_image])

demo.launch()
