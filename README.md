# Flux Gym with proper aspec ratio support and buckets

Enhanced Dead simple web UI for training FLUX LoRA **with LOW VRAM (12GB/16GB/20GB) support.**

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

# Here are my additions: 

NEW: Fixed serious issue in original FlugGym where some images might be skipped due to poor logic in image processing . (Like you add 100 images, but it uses only 70) This new method is robust and clean .

- The app now spawns a new terminal for the Kohya process instead of redirecting the Kohya messages to the gradio web "termina log". This is more reliable, logical and you also can clearly see if the other process is running or not (for example if you previously by accident closed the FluxGym browser window, you gad no idea if the process continues. This way the FluxGym Gradio is not a parent of the process and the training continues, even if you close FluxGym - and writes to it's own terminal window. 
- added Batch parameter, clearly defined
- LORA resolution and resize is separated
- resolution is a tuple (width and Height)
- resize = 0 will not resize the images
- downsize only checkbox for resize 
  
![image](https://github.com/user-attachments/assets/5a7bbd4d-568f-49b8-9255-115470ad523e)


# Other features:
- save/load config (only settings, not the images). Export for kohya_ss config (might need some checking, some params are obvious, some are not ¯\\_(ツ)_/¯ )
  
  ![image](https://github.com/user-attachments/assets/d6279174-6c03-44b7-88c9-d5fba58e3b85)

- Simple set buckets or no buckets (will fill all the necessary params) Before clicking on the Set Bucket, setup Train Resolution width/height as the numbers will be also used in the config. Then press Quick: Enable Buckets which will then fill all the (few) necessary parameters for proper bucket handling. 
  
  ![image](https://github.com/user-attachments/assets/8099e9de-e337-488f-a9e4-e04ae019a8b4)


# Other changes from original pull requests
- proper calculation of step count (in original it would count also non image files, giving you wildly exgerrated step-count)
- UTF-8 for caption
- proper image/txt collection - the original method can very easily skip images because of how messy it was designed. (So you think you are training on 100 images, but it only collected 70 for example)

My changes were done so it can correctly work with buckets. The original code not only merged resolution and resize into one parameter but it will always resize all images, making -enable_bucket counter-intuitive or kinda worthless because it will lead in double resizing (potentially in both direction)

# Install
if you have working fluxgym, all you need to do is replace app.py from this repo into yours (same if you use Stability Matrix, Pinokio etc...)
That's all where the changes are.

# Example 0 for simple LORA without buckets

resize: 768

resolution width: 768

resolution height: 0

Resize and Resolution width should be the same, whatever it is 512, 768...
It will resize all images to 768 in the shortest side and crop from them to the square. If your images are not square (for example portrait aspect) the result LORA will love to crop heads and feet

# Quick Bucket button example

## Mainly Squares (30% or more)

**(I have images of various sizes, but many are squares)**

You only need to set: 

resolution width: 768

Press: Quick Enable Buckets

(it will use the resolution width to set max bucket size, it will zero resize, set enable bucket and no upsacale)

## Strict size example:

**(I have images mostly of certain sizes such a 1024 x 768 or 768 x 1024 not many squares)**

resolution width: 768

resolution height: 1024

Press: Quick Enable Buckets

(it will use the whichever is larger (width or height) to set max bucket size, it will zero resize, set enable bucket and no upsacale)

# Quick Bucket plus Resize down (faster Lora creation)

**(I have images of various sizes, but want to make Lora faster)**

say you have images of 860 x 1200, 1200 x 860, 768 x 768 etc...

If you want the best quality you would set width/height to 860 x 1200. But it will take forever. If you want faster Lora creation, you can do it by making training 768 x 768 

resolution width: 768
resolution height: 768 

Press: Quick Enable Buckets

(it will use the resolution width to set max bucket size, it will zero resize, set enable bucket and no upsacale and check Resize down only checkbox)

After you press the Quick Enable buckets, see the Resize is set to 0, but in this case we want to resize down the input images 

Set Resize to the same size as resolution width: 768
Set Downsize only checkbox (if you have any images that are < 768, it will not size them up, but create a bucket for them)

So this will resize the shortest dimension of all images to 768 and the buckets will be created accordingly with smaller sizes

Now start. 

# Avoid double resizing penalty

If you use Resize non zero it will resize images before processing further. But if you have buckets enabled, the buckets will also resize these images once more, potentially making them blurry. In fact in the original Flux Gym you could have flux gym resize image up and then bucket resize it down making the result blurry. 

So the solution is 

a) don't resize anything (Resize = 0) The buckets will resize the images as needed, as set by the Width and Height resolution (whichever is bigger).
b) if you must resize images, then choose the Width and the Height to be one that of most images after resize. So if you have most images 896 x 1152 and you set Resize to 768 then you need to set your resolution width to 768 and height to 986 (so scalled down 896 x 1152). This will create bucket with most images used directly without further second resizing in the training process.

# Example 1 for bucket with mostly square images or equal mix of square and non square images
You should manually create the desired multi-resolution images. Don't just gobble random images in various random sizes - this will NOT work as you imagine. So say stick to 768 x 768, 768 x 1024, 1024 x 768 for 3 buckets. If you put random images that are seriously different than the resolution the result will be glorified garbage as the resizing will make it blurry (LORA seems to picks up on that part most).
You probably want to always use --bucket_no_upscale because upsaclaing buckets makes things worse every time. 
In short: create the image set with only a few different image sizes, but also don't go crazy on the size itself (large size will not help, rather make things worse), keep it around the size they will be used as specified in Training Resolution (So 512, 768...)

So for example: 15 images 767x768, 5 images 768x1024, 5 images 1024x768, don't put random odd image sizes, prepare these in advance.

resize: 0

resolution width: 768

resolution height: 0

--enable_bucket

--bucket_no_upscale

--max_bucket_reso: 768 (set to the reslolution width or larger: 768)

(I know the script says min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, but actually it needs to be set as the max bucket size is calculated from it)

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
So for example: 20 images 768x1024, 5 images 1024x768

resize: 0

resolution width: 768

resolution height: 1024

--enable_bucket

--bucket_no_upscale

--max_bucket_reso set: 1024  (set to the reslolution width or height (whichever is larger) or larger)

(I know the script says min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, but actually it needs to be set as the max bucket size is calculated from it)

will create buckets to fit the pixel area of 768x1024 so if you have 768 x 1024 images they will be directly used in the bucket, same if you have 1024 x 768, they will be in another bucket unchanged as the area is the same.
This is a good option if your images are largely non-square. Setting resolution to the size of your images will ensure they will be used without cropping and resizing during the bucket creation. 
For example if most or all of your images are 768 x 1024 use this option and your LORA will correctly use the aspect ratio without cutting heads and feet and use the input images directly without resizing

# Some theory 
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

You probably donm't really want to scale the images up so use --no_bucket_upscale

# The original readme follows

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

!!! After that you can replace the original app.py with the app.py from this repo

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

!!! Replace app.y from this repo to the installed folder (that's all the changes I made)

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
