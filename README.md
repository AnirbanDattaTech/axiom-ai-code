# axiom-ai-code

The runnable code for [axiom-ai.tech](https://axiom-ai.tech/), a site about how AI systems are built, why they work, and where they stop, written in plain language from first principles.

When a post on the site has code, the code lives here: notebooks you can open, run and change. Nothing here needs a cloud account or a paid service. Most of it should run on an ordinary machine with a consumer GPU, and some of it on a CPU.

## How it's organised

Each folder's path is its post's path. A post at

    axiom-ai.tech/generative-ai/vision/02-vision-transformer/

has its code in

    generative-ai/vision/02-vision-transformer/

So far the tree looks like this:

    axiom-ai-code/
    ├── README.md
    ├── LICENSE              # Apache-2.0, for the code
    ├── LICENSE-CONTENT      # CC BY-NC-ND 4.0, for text, photographs and figures
    ├── NOTICE
    ├── requirements.txt
    └── generative-ai/
        └── vision/
            └── 02-vision-transformer/
                ├── README.md
                ├── 01-vision-transformer.ipynb
                ├── photo-01.jpg
                └── photo-02.jpg

Each post folder has its own README: which post it belongs to, what the code shows, what it needs, and what I saw when I ran it. Posts without code, like the prologue to the Vision series, have no folder here.

## Posts with code

**Generative AI**

- *Vision: Multimodal AI*, post 2: [What a vision model sees](generative-ai/vision/02-vision-transformer/). A photograph cut into 196 squares, and a Vision Transformer asked what it sees. With a beetle, a squirrel, and a list of a thousand words.

**Agentic AI**

- The *Agentic Flow* series will add its first folder here.

## Getting started

Tested with Python 3.12 on Windows. The steps are the same on macOS and Linux, apart from how an environment is activated.

1. **Clone the repository.**

       git clone https://github.com/AnirbanDattaTech/axiom-ai-code.git
       cd axiom-ai-code

2. **Create an environment,** with either venv or conda.

       python -m venv .venv
       .venv\Scripts\activate          # Windows
       source .venv/bin/activate       # macOS and Linux

       conda create -n axiom-ai python=3.12
       conda activate axiom-ai

3. **Install PyTorch and torchvision** using the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). It gives the right command for your system and your GPU, or for no GPU at all. Doing this first matters: the plain `pip install torch` may not give you a build that uses your GPU.

4. **Install the rest.**

       pip install -r requirements.txt

5. **Open a notebook** from its own folder, in Jupyter or VS Code, and run the cells in order.

If you use conda with VS Code, start VS Code from a terminal where the environment is already active. That way the notebook's kernel finds the environment's own libraries.

## A note on hardware

Everything here was written and tested on one NVIDIA RTX 3060 with 12 GB of memory. When something is too large for a machine like that, such as training a modern vision-language model, the post says so plainly, and the code here runs the part that does fit: a tokenizer, a single layer, one forward pass.

## If something doesn't run

Please open an issue with the error, your operating system, and your versions of Python and PyTorch. Each post's README lists the problems I met while writing it, and how they were solved, so it's worth a look first.

## Licence

This repository holds two kinds of material, under two licences.

**Code**, meaning the Python files and the code cells in the notebooks, is licensed under the [Apache License 2.0](LICENSE). You're welcome to use it, change it and build on it, for any purpose. If you share it, please keep the `LICENSE` and `NOTICE` files with it.

**Everything else**, meaning the text of the READMEs, the markdown cells in the notebooks, the photographs and the figures, is © 2026 Anirban Datta and licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (full text in [LICENSE-CONTENT](LICENSE-CONTENT)). You may share it unchanged, with credit, for non-commercial purposes. You're free to change the notebooks as much as you like for your own learning; the licence only asks that changed versions of the text aren't published.

Material by others, such as figures from papers, keeps its own licence, noted where it appears.

When crediting, please use *Anirban Datta, axiom-ai.tech*, with a link to the original page.