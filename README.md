## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Developing an accessible and interactive user interface is essential for demonstrating the capabilities of a complex machine learning model like Stable Diffusion. The challenge is to efficiently integrate the Stable Diffusion API (in this case, via Hugging Face) with a user-friendly frontend framework like Gradio, allowing users to input text prompts and control generation parameters (like negative prompts, inference steps, and image dimensions) to produce and visualize generated images in real-time.

### DESIGN STEPS:

#### STEP 1:
Install libraries, configure the Hugging Face API key, and define a helper function for secure API calls.

#### STEP 2:
Create the main generate function to call the API and convert the base64 output into a PIL Image.

#### STEP 3:
Design a basic gr.Interface with a text prompt input and an image output to confirm core functionality.

#### STEP 4:
Update the generate function and gr.Interface to include controls for negative prompts and other parameters.

#### STEP 5:
Switch to gr.Blocks and use gr.Row/gr.Column to organize parameters and output visually.

#### STEP 6:
Use gr.Accordion to hide advanced options and launch the application with demo.launch(share=True).

### PROGRAM:

```
import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json

#Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))
prompt = "a cat in a park"

result = get_completion(prompt)
IPython.display.HTML(f'<img src="data:image/png;base64,{result}" />')
import gradio as gr 

#A helper function to convert the PIL image to base64
#so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt):
    output = get_completion(prompt)
    result_image = base64_to_pil(output)
    return result_image

gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[gr.Textbox(label="Your prompt")],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never",
                    examples=["the spirit of a tamagotchi wandering in the city of Vienna","a mecha robot in a favela"])

demo.launch(share=True, server_port=int(os.environ['PORT1']))
demo.close()
import gradio as gr 

#A helper function to convert the PIL image to base64 
# so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image
gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[
                        gr.Textbox(label="Your prompt"),
                        gr.Textbox(label="Negative prompt"),
                        gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                                 info="In how many steps will the denoiser denoise the image?"),
                        gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
                                  info="Controls how much the text prompt influences the result"),
                        gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
                        gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
                    ],
                    outputs=[gr.Image(label="Result")],
                    title="Image Generation with Stable Diffusion",
                    description="Generate any image with Stable Diffusion",
                    allow_flagging="never"
                    )

demo.launch(share=True, server_port=int(os.environ['PORT2']))
demo.close()
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    prompt = gr.Textbox(label="Your prompt")
    with gr.Row():
        with gr.Column():
            negative_prompt = gr.Textbox(label="Negative prompt")
            steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
            guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
            width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
            height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
            btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Image(label="Result")

    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])
gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT3']))
demo.close()
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt") #Give prompt some real estate
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit") #Submit button side by side!
    with gr.Accordion("Advanced options", open=False): #Let's hide the advanced options!
            negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
                    guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result") #Move the output up too
            
    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT4']))
gr.close_all()
```

### OUTPUT:

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/a453c95a-66e6-482e-96c4-d808dfc6b5ac" />

<img width="523" height="573" alt="image" src="https://github.com/user-attachments/assets/9b9a6985-5d39-455d-9aa0-09e8e891371e" />

<img width="520" height="567" alt="image" src="https://github.com/user-attachments/assets/da48d410-7523-47b2-85f6-9075da30f3e9" />

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/7089877c-9bc4-4889-86d8-7ea34caab294" />

<img width="526" height="573" alt="image" src="https://github.com/user-attachments/assets/ee403ed4-2d71-4de0-900f-66281379159d" />

<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/3422062c-9d4e-4a20-9067-54df7e3ca0b2" />

<img width="535" height="558" alt="image" src="https://github.com/user-attachments/assets/ebdbce0d-754f-4f66-a9df-1ce407622682" />

### RESULT:
Thus, a prototype application for image generation utilizing the Stable diffusion model is designed and deployed successfully.
