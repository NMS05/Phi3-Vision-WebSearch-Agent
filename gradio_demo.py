import gradio as gr
from search_agent import vlm_rag_agent

print("\n\t Loading Agent .... !\n")
AGENT = vlm_rag_agent(gradio_demo=True)

# Define the Gradio interface
def gradio_interface(image_url, query):
    response = AGENT.chat_with_agent(image_url, query)
    return response

# Create Gradio inputs
image_input = gr.Textbox(label="Image URL")
query_input = gr.Textbox(label="Text Query")
# Create Gradio output
response_output = gr.Textbox(label="Agent Response",scale=2)

# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[image_input, query_input],
    outputs=response_output,
    title="Phi3 Vision WebSearch Agent",
    description="Provide an image URL and a text query to chat with the model."
)

iface.launch(share=False, server_port=8080, server_name="127.0.0.1")