import gradio as gr


def greet(name):
    return f"Hello {name}!"

def classify_image(image):
    return "cat"  # Placeholder for actual image classification logic


with gr.Blocks() as demo:
    gr.Markdown("# Cat vs Dog Classifier")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(shape=(224, 224))
            classify_button = gr.Button("Classify")
        with gr.Column():
            output_text = gr.Textbox(label="Prediction")

    classify_button.click(fn=classify_image, inputs=image_input, outputs=output_text)

# Image input and output example
demo = gr.Interface(
    fn=greet, 
    inputs=gr.inputs.Image(shape=(224, 224)), 
    outputs="text"
)

demo.launch(debug=True)
demo.launch(share=True)
