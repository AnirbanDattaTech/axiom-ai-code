# What a vision model sees

Companion code for the second post in the *Vision: Multimodal AI* series on [axiom-ai.tech](https://axiom-ai.tech/generative-ai/vision/02-vision-transformer/).

The Vision Transformer's paper has a small box in the corner of its first figure. It holds the model's answer: *bird*, *ball* or *car*, one word from a fixed list. This notebook opens that box. It takes a photograph, cuts it into squares the way the model does, and asks the model what it sees. Then it does the same with a second photograph, and looks at the list the answers come from.

## What's here

- `01-vision-transformer.ipynb`: the notebook. Six short cells, each with a note above it.
- `photo-01.jpg`: a beetle among leaves.
- `photo-02.jpg`: a palm squirrel in a tree, eating.

## What the notebook does

1. **Setup.** Imports, the folder, and the device.
2. **What the model sees.** The photograph is resized, cropped to a 224 × 224 square from the centre, and cut into a 14 × 14 grid of 16-pixel squares: 196 patches.
3. **From squares to tokens.** One convolution turns each square into a list of 768 numbers, a token. A class token joins at the front, and every token gets a learned position. The cell checks that these steps match what the library does itself.
4. **The answer.** The model gives a probability for each of its 1,000 labels, and we print the top five.
5. **A second photograph.** The same steps, on the squirrel.
6. **The list itself.** A count of the beetles and squirrels the list knows by name.

## What I saw

On the beetle, the model answered *leaf beetle* with 75.7%. Its other four guesses were insects too, which is a fair result for a beetle that covers about a dozen of the 196 squares.

On the squirrel, it answered *capuchin* (33.7%), then *marmoset*, *macaque* and *titi*. The nearest it came to the right word was *squirrel monkey*, at 3.6%. Cell 6 suggests why. The list names six beetles, and two squirrels, one of which is a monkey. The model can only answer with the words it has, so it chose the ones whose pictures looked most like this one: a small animal, upright in a tree, holding food with both hands. The squirrel would like it noted that he is not a capuchin.

Your numbers should match these closely. A different GPU, or a CPU, can change the last decimal place.

## Running it

Tested on Windows with Python 3.12, PyTorch 2.11 (CUDA 12.8) and torchvision 0.26, on an NVIDIA RTX 3060 with 12 GB. The code falls back to the CPU when there is no GPU.

1. Install PyTorch and torchvision for your machine, using the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). It gives the right command for your system and your GPU, or for no GPU at all.
2. From the repository root, install the rest: `pip install -r requirements.txt`.
3. Open the notebook from this folder, in Jupyter or VS Code, and run the cells in order.

The notebook reads the photographs from its own folder and saves its figures there. On the first run, torchvision downloads the model's weights, about 330 MB, and keeps them in its cache for next time.

### If the kernel stops without an error on Windows

This happened while writing the post. Two things helped:

- **Two copies of the OpenMP runtime.** PyTorch brings one, and the plotting stack can bring another. The first two lines of Cell 1 let them coexist. If you run the same code as a script, the message is `OMP: Error #15`.
- **A conda environment that isn't active.** If you use conda, start Jupyter or VS Code from a terminal where the environment is already active, so the kernel finds the environment's own libraries.

## References

- Alexey Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. https://arxiv.org/abs/2010.11929
- Olga Russakovsky et al. (2015). *ImageNet Large Scale Visual Recognition Challenge.* IJCV. https://arxiv.org/abs/1409.0575. The source of the 1,000 labels.
- torchvision: *Vision Transformer* models and weights. https://pytorch.org/vision/stable/models/vision_transformer.html

## The photographs

Both photographs are © 2026 Anirban Datta, licensed under CC BY-NC-ND 4.0, like the rest of the text here. See the licence section of the repository README.