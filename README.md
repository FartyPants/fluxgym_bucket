# Flux Gym with proper aspec ratio support and buckets

Enhanced Dead simple web UI for training FLUX LoRA **with LOW VRAM (12GB/16GB/20GB) support.**

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

# Here are my additions: 

- LORA resolution and resize is separated
- resolution is a tuple (width and Height)
- resize = 0 will not resize the images

Other changes from original pull requests
- proper calculation of step count (in original it would count also non image files, giving you wildly exgerrated step-count)
- UTF-8 for caption

My changes were done so it can correctly work with buckets. The original code not only merged resolution and resize into one parameter but it will always resize all images, making -enable_bucket counter-intuitive or kinda worthless

# Install
if you have working fluxgym, all you need to do is replace app.py from this repo into yours (same if you use Stability Matrix etc...)
That's all where the changes are.

![image](https://github.com/user-attachments/assets/0811780b-6193-4661-92f8-fb7fa2876d6e)


# Example 0 for simple no bucket

resize: 768

resolution width: 768

resolution height: 0

It will resize all images to 768 in the shortest side and crop from them to the square. If your images are not square (for example portrait aspect) the result LORA will love to crop heads and feet

# Example 1 for bucket with mostly square images or mix of square and non square
You should manually create the desired multi-resolution images. Don't just gobble random images in various random sizes - this will NOT work as you imagine. So say stick to 768 x 768, 768 x 1024, 1024 x 768 for 3 buckets. If you put random images that are seriously different than the resolution the result will be glorified garbage as the resizing will make it blurry (LORA seems to picks up on that part most).
You probably want to always use --bucket_no_upscale because upsaclaiing buckets makes things worse every time. 

resize: 0

resolution width: 768

resolution height: 0

--enable_bucket

--bucket_no_upscale

Setting resize 0 will **not** resize the input images and it will fit images to 768 * 768 pixel area (it means the buckets will all be created to fit the pixel area, so even if you have 768 x 1024, the bucket will be sized down to 640 x 864 and so your original images will be resized (as the area of 640 x 864 is close to 768 x 768). If you for example want area fit 768 x 1024 then you can set 896 as the square area (resolution width) because 896 x 896 is roughly same area as 768 x 1024

So this it's a good option if you have most images square and then add various odd aspect ratios or say you have 1/3 square 1/3 portrait and 1/3 landscape

Here is the math (train_util.py)
Original image size:

width = 768
height = 1024
aspect_ratio = 768 / 1024 = 0.75

Check pixel area:

768 × 1024 = 786,432 pixels > 589,824 → bucket needs downscaling

Use formulas from code:

resized_width = sqrt(max_area * aspect_ratio)
resized_height = max_area / resized_width

Plug in numbers:

resized_width = sqrt(589824 × 0.75) = sqrt(442368) ≈ 665.07
resized_height = 589824 / 665.07 ≈ 886.68

Round sizes:
Final resized dimensions before rounding to reso_steps:

width ≈ 665
height ≈ 887

Rounded to reso_steps (e.g., 64):
If reso_steps = 64, your bucketing logic might round down like:

bucket:
width = 640
height = 864


# Example 2 for bucket with non square images preserving the maximum quality and aspect

resize: 0

resolution width: 768

resolution height: 1024

--enable_bucket

--bucket_no_upscale

will create buckets to fit the pixel area of 768x1024 so if you have 768 x 1024 images they will be directly used in the bucket, same if you have 1024 x 768, they will be in 1024 bucket as the area is same.
This is a good option if your images are largely non-square. Setting resolution to the size of your images will ensure they will be used without cropping and resizing. 
For example if most or all of your images are 768 x 1024 use this option and your LORA will correctly use the aspect ratio without cutting heads and feet

# Important 
 Aspect-ratio-aware resizing and bucketing:

    If no_upscale = False:
    It looks for the closest predefined bucket (in aspect ratio) and scales the image up or down to fit.

    If no_upscale = True:
    It only downscales the image, to keep it within a maximum allowed area (self.max_area calculated from resultion width x resolution height), while preserving the aspect ratio.


2. Resolution rounding:

Uses helper function:

def round_to_steps(self, x):
    x = int(x + 0.5)
    return x - x % self.reso_steps

This rounds x to the nearest lower multiple of self.reso_steps.
3. Final bucket size:

    After resizing, it trims both width and height to be divisible by reso_steps, avoiding any padding.

max_area represents the maximum allowed pixel area for an image before it gets resized.

max_area = resolution_width * resolution_height  # for example, 1024 * 1024

It sets an upper bound on the number of pixels an image can occupy. If an image exceeds this area, it's resized down, keeping the aspect ratio, so that:

new_width * new_height <= max_area

By default kohya_ss will use hamming for scaling down and lanczos for scaling up.

You probably donm't really want to scale the images (especially not up) so the best is to make the area (= width x height) same for all aspects so: 768 x 1024 and 1024 x 768 and square 896 × 896 all have aboyt the same area


- **Frontend:** The WebUI forked from [AI-Toolkit](https://github.com/ostris/ai-toolkit) (Gradio UI created by https://x.com/multimodalart)
- **Backend:** The Training script powered by [Kohya Scripts](https://github.com/kohya-ss/sd-scripts)

FluxGym supports 100% of Kohya sd-scripts features through an [Advanced](#advanced) tab, which is hidden by default.

![screenshot.png](screenshot.png)

---


# What is this?

1. I wanted a super simple UI for training Flux LoRAs
2. The [AI-Toolkit](https://github.com/ostris/ai-toolkit) project is great, and the gradio UI contribution by [@multimodalart](https://x.com/multimodalart) is perfect, but the project only works for 24GB VRAM.
3. [Kohya Scripts](https://github.com/kohya-ss/sd-scripts) are very flexible and powerful for training FLUX, but you need to run in terminal.
4. What if you could have the simplicity of AI-Toolkit WebUI and the flexibility of Kohya Scripts?
5. Flux Gym was born. Supports 12GB, 16GB, 20GB VRAMs, and extensible since it uses Kohya Scripts underneath.

---

# News

- September 25: Docker support + Autodownload Models (No need to manually download models when setting up) + Support custom base models (not just flux-dev but anything, just need to include in the [models.yaml](models.yaml) file.
- September 16: Added "Publish to Huggingface" + 100% Kohya sd-scripts feature support: https://x.com/cocktailpeanut/status/1835719701172756592
- September 11: Automatic Sample Image Generation + Custom Resolution: https://x.com/cocktailpeanut/status/1833881392482066638

---

# Supported Models

1. Flux1-dev
2. Flux1-dev2pro (as explained here: https://medium.com/@zhiwangshi28/why-flux-lora-so-hard-to-train-and-how-to-overcome-it-a0c70bc59eaf)
3. Flux1-schnell (Couldn't get high quality results, so not really recommended, but feel free to experiment with it)
4. More?

The models are automatically downloaded when you start training with the model selected.

You can easily add more to the supported models list by editing the [models.yaml](models.yaml) file. If you want to share some interesting base models, please send a PR.

---

# How people are using Fluxgym

Here are people using Fluxgym to locally train Lora sharing their experience:

https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


# More Info

To learn more, check out this X thread: https://x.com/cocktailpeanut/status/1832084951115972653

# Install

## 1. One-Click Install

You can automatically install and launch everything locally with Pinokio 1-click launcher: https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


## 2. Install Manually

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```

Your folder structure will look like this:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
```

Now activate a venv from the root `fluxgym` folder:

If you're on Windows:

```
python -m venv env
env\Scripts\activate
```

If your're on Linux:

```
python -m venv env
source env/bin/activate
```

This will create an `env` folder right below the `fluxgym` folder:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
  /env
```

Now go to the `sd-scripts` folder and install dependencies to the activated environment:

```
cd sd-scripts
pip install -r requirements.txt
```

Now come back to the root folder and install the app dependencies:

```
cd ..
pip install -r requirements.txt
```

Finally, install pytorch Nightly:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or, in case of NVIDIA RTX 50-series (5090, etc.) you will need to install cu128 torch and update bitsandbytes to the latest:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -U bitsandbytes
```


# Start

Go back to the root `fluxgym` folder, with the venv activated, run:

```
python app.py
```

> Make sure to have the venv activated before running `python app.py`.
>
> Windows: `env/Scripts/activate`
> Linux: `source env/bin/activate`

## 3. Install via Docker

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```
Check your `user id` and `group id` and change it if it's not 1000 via `environment variables` of `PUID` and `PGID`. 
You can find out what these are in linux by running the following command: `id`

Now build the image and run it via `docker-compose`:
```
docker compose up -d --build
```

Open web browser and goto the IP address of the computer/VM: http://localhost:7860

# Usage

The usage is pretty straightforward:

1. Enter the lora info
2. Upload images and caption them (using the trigger word)
3. Click "start".

That's all!

![flow.gif](flow.gif)

# Configuration

## Sample Images

By default fluxgym doesn't generate any sample images during training.

You can however configure Fluxgym to automatically generate sample images for every N steps. Here's what it looks like:

![sample.png](sample.png)

To turn this on, just set the two fields:

1. **Sample Image Prompts:** These prompts will be used to automatically generate images during training. If you want multiple, separate teach prompt with new line.
2. **Sample Image Every N Steps:** If your "Expected training steps" is 960 and your "Sample Image Every N Steps" is 100, the images will be generated at step 100, 200, 300, 400, 500, 600, 700, 800, 900, for EACH prompt.

![sample_fields.png](sample_fields.png)

## Advanced Sample Images

Thanks to the built-in syntax from [kohya/sd-scripts](https://github.com/kohya-ss/sd-scripts?tab=readme-ov-file#sample-image-generation-during-training), you can control exactly how the sample images are generated during the training phase:

Let's say the trigger word is **hrld person.** Normally you would try sample prompts like:

```
hrld person is riding a bike
hrld person is a body builder
hrld person is a rock star
```

But for every prompt you can include **advanced flags** to fully control the image generation process. For example, the `--d` flag lets you specify the SEED.

Specifying a seed means every sample image will use that exact seed, which means you can literally see the LoRA evolve. Here's an example usage:

```
hrld person is riding a bike --d 42
hrld person is a body builder --d 42
hrld person is a rock star --d 42
```

Here's what it looks like in the UI:

![flags.png](flags.png)

And here are the results:

![seed.gif](seed.gif)

In addition to the `--d` flag, here are other flags you can use:


- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` also work. (Learn more about [Attention/Emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis))

## Publishing to Huggingface

1. Get your Huggingface Token from https://huggingface.co/settings/tokens
2. Enter the token in the "Huggingface Token" field and click "Login". This will save the token text in a local file named `HF_TOKEN` (All local and private).
3. Once you're logged in, you will be able to select a trained LoRA from the dropdown, edit the name if you want, and publish to Huggingface.

![publish_to_hf.png](publish_to_hf.png)


## Advanced

The advanced tab is automatically constructed by parsing the launch flags available to the latest version of [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts). This means Fluxgym is a full fledged UI for using the Kohya script.

> By default the advanced tab is hidden. You can click the "advanced" accordion to expand it.

![advanced.png](advanced.png)


## Advanced Features

### Uploading Caption Files

You can also upload the caption files along with the image files. You just need to follow the convention:

1. Every caption file must be a `.txt` file.
2. Each caption file needs to have a corresponding image file that has the same name.
3. For example, if you have an image file named `img0.png`, the corresponding caption file must be `img0.txt`.
