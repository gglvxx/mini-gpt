"""Interfață web pentru MiniGPT. Pornește cu: python app.py"""
import gradio as gr
import torch

from generate import load_model  # refolosim funcția deja scrisă în generate.py
from src.config import Config

# Modelul se încarcă O SINGURĂ dată, la pornirea aplicației (nu la fiecare click)
cfg = Config()
model, tokenizer, ck = load_model(cfg)


def generate_text(prompt, tokens, temperature, top_k, seed):
    """Funcție-generator: fiecare 'yield' trimite textul de până acum în browser (streaming)."""
    if seed >= 0:
        torch.manual_seed(int(seed))

    prompt = prompt or "\n"
    try:
        ids = tokenizer.encode(prompt)
    except ValueError as e:
        # gr.Error afișează mesajul frumos în interfață, fără să oprească aplicația
        raise gr.Error(f"Prompt invalid. {e}")

    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    text = prompt
    for i in range(int(tokens)):
        idx = model.generate(idx, max_new_tokens=1, temperature=temperature, top_k=int(top_k))
        text += tokenizer.decode(idx[0, -1:])
        if i % 5 == 0:  # actualizăm pagina la fiecare 5 caractere -> fluid, dar fără trafic inutil
            yield text
    yield text  # textul final complet


# --- Construcția interfeței ---
with gr.Blocks() as demo:
    gr.Markdown(
        f"# Mini-GPT\n"
        f"Model de limbaj Transformer antrenat de la zero pe Shakespeare · "
        f"**{model.num_params() / 1e6:.2f}M** parametri · val loss **{ck['val_loss']:.3f}**"
    )

    with gr.Row():
        # Coloana stângă: setări
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="ROMEO:", lines=2)
            tokens = gr.Slider(50, 1000, value=300, step=50, label="Caractere de generat")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                    label="Temperature (mic = sigur, mare = creativ)")
            top_k = gr.Slider(1, tokenizer.vocab_size, value=40, step=1, label="Top-k")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatoriu)")
            button = gr.Button("Generează", variant="primary")

        # Coloana dreaptă: rezultatul
        with gr.Column(scale=2):
            output = gr.Textbox(label="Text generat", lines=22)

    # Exemple gata făcute, la un click distanță
    gr.Examples(
        examples=[
            ["ROMEO:", 300, 0.8, 40, 42],
            ["KING HENRY:", 300, 0.5, 20, 7],
            ["First Citizen:", 300, 1.2, 65, 1],
        ],
        inputs=[prompt, tokens, temperature, top_k, seed],
    )

    # Legătura: click pe buton -> rulează generate_text -> rezultatul apare în output
    button.click(generate_text, inputs=[prompt, tokens, temperature, top_k, seed], outputs=output)


if __name__ == "__main__":
    demo.launch()