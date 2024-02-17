from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from transformers import AutoTokenizer
from gradio.themes.utils import colors, fonts, sizes
import subprocess
import psutil
from check_sources import create_prompt_with_source
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import LlamaRAMCache

# Load required environment variables with default fallback values
REPO_ID = os.getenv('REPO_ID', 'szymonrucinski/krakowiak-v2-7b-gguf')
FILENAME = os.getenv('FILENAME', 'krakowiak-v2-7b-gguf.Q2_K.bin')
TOKENIZER_NAME = os.getenv('TOKENIZER_NAME', 'mistralai/Mistral-7B-Instruct-v0.1')

# Download the model file from Hugging Face Hub
hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=".",
)

# Initialize the LLaMA model with specified parameters
llm = Llama(model_path=f"./{FILENAME}", n_threads=2, n_ctx=1024)

# Define the visual theme for the Gradio interface
theme = gr.themes.Monochrome(
    primary_hue="orange",
    secondary_hue="red",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

def get_system_memory():
    """Fetches and returns system memory usage statistics."""
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used = memory.used / (1024.0**3)  # Convert bytes to GB
    memory_total = memory.total / (1024.0**3)  # Convert bytes to GB
    return {
        "percent": f"{memory_percent}%",
        "used": f"{memory_used:.3f}GB",
        "total": f"{memory_total:.3f}GB",
    }

def generate(
    instruction: str,
    max_new_tokens: int,
    temp: float,
    top_p: float,
    top_k: int,
    rep_penalty: float,
    enable_internet_search: bool,
):
    """Generates a response based on the instruction provided by the user."""
    if enable_internet_search:
        # Use internet search to construct a prompt
        prompt = create_prompt_with_source(instruction)
    else:
        # Directly use the instruction as a prompt for the model
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
        chat = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    print(prompt)  # Debug: Print the prompt

    result = ""
    # Generate response using LLaMA model
    for x in llm(
        prompt,
        stop=['</s>'],
        stream=True,
        max_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=rep_penalty,
    ):
        result += x["choices"][0]["text"]
        yield result  # Yield the response incrementally

# Predefined example prompts for user convenience
examples = [
    "Jaki obiektyw jest idealny do portretów?",
    "Kiedy powinienem wybrać rower gravelowy a kiedy szosowy?",
    "Czym jest sztuczna inteligencja?",
    "Jakie są największe wyzwania sztucznej inteligencji?",
    "Napisz proszę co należy zjeść po ciezkim treningu?",
    "Mam zamiar aplikować na stanowisko menadżera w firmie. Sformatuj mój życiorys.",
]

def process_example(input: str):
    """Processes an example input by generating a response."""
    for x in generate(input, 256, 0.5, 0.9, 40, 1.0):
        pass  # Iterate through generator to get the last piece
    return x  # Return the final response

css = ".generating {visibility: hidden} \n footer {visibility: hidden}"  # Custom CSS for the Gradio interface

# Custom theme class for Gradio interface
class SeafoamCustom(Base):
    # Initialization with customized visual properties
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )
        # Set additional visual properties
        super().set(
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            input_background_fill="zinc",
            input_border_color="*secondary_300",
            input_shadow="*shadow_drop",
            input_shadow_focus="*shadow_drop_lg",
        )

seafoam = SeafoamCustom()  # Instantiate the custom theme

# Main Gradio Blocks interface setup
with gr.Blocks(theme=seafoam, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(""" ## 🤖 Krakowiak - Polski model językowy 🤖 \n
                        ### by [Szymon Ruciński](https://www.szymonrucinski.pl/) \n
                        Wpisz w poniższe pole i kliknij przycisk, aby wygenerować odpowiedzi na najbardziej nurtujące Cię pytania! 🤗 \n
                        ***W celu zapewnienia optymalnej wydajności korzystasz z modelu o zredukowanej liczbie parametrów. Jest on 4 razy mniejszy niż oryginalny i generuje odpowiedzi o znacząco niższej jakości.***
                     """)

# Final setup and launch of the Gradio app
if __name__ == "__main__":
    demo.queue(concurrency_count=1, max_size=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
