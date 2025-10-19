import gradio as gr
from src.infer import inference_pipeline

# model_path = "checkpoints/ckpt_23_10_2025/best_cat_dog_classifier_model_20251019_122336.pth"


def classify_image(
        image_path: str
) -> str:
    """
    Classify the input image as cat or dog.
    """
    if image_path is None:
        return "Please upload an image."
    try:
        prediction = inference_pipeline(
            image_path=image_path, 
            model_path=model_path,
            hf=True
        )
        return f"Prediction: {prediction.capitalize()}"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🐶🐱 Cat vs Dog Classifier")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="filepath", 
                label="Input"
            )
            classify_button = gr.Button("🔍 Classify")
        with gr.Column():
            output_text = gr.Textbox(label="🧠 Prediction", placeholder="Result will appear here")

    classify_button.click(fn=classify_image, inputs=[image_input], outputs=[output_text])

demo.launch(debug=True)

