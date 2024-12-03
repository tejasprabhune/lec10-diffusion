# Lecture 10: Diffusion!

In this short in-class assignment, we will implement simple vanilla U-Net
and Diffusion models trained on the MNIST dataset!

For setup, the easiest way is to install `uv` and run the following:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # If you have curl (windows/mac/linux) OR
wget -qO- https://astral.sh/uv/install.sh | sh      # If you don't have curl OR
brew install uv                                     # If you have homebrew

git clone https://github.com/tejasprabhune/lec10-diffusion
cd lec10-diffusion
```

To run any of the scripts, you can use `uv`:
```bash
uv run train.py --task unconditional_unet --data noisy_mnist --n_epochs 5
uv run train.py --task time_unet --data mnist --n_epochs 20
```

(i.e., just replace `python` with `uv run`, which will automatically install
the dependencies for you in a virtual environment.)

Here are the steps to complete this assignment:

1. Implement the `UnconditionalUNet` class in `diffuse/models/unet.py`.
2. Run `python train.py --task unconditional_unet --data noisy_mnist --n_epochs 5`
   to train the model and visualize the results.
3. Implement the `TimeConditionalUNet` class in `diffuse/models/unet.py`.
4. Implement the `ddpm_forward` and `ddpm_sample` functions in `diffuse/models/ddpm.py`.
   Read through the `DDPM` class which puts the U-Net and the functions together!
5. Run `python train.py --task time_unet --data mnist --n_epochs 20` to train the model
   and visualize the results.

You can get started by exploring the code!

We've provided a few sanity checks for code already written for you:

```bash
python -m unittest discover tests                   # for all tests
```

You are free to write your own tests for the code you write using the
test files as templates.

Good luck!

