import os
# os.environ["USE_LIBUV"] = "0"

import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
import json

MAX_IMAGES = 150

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):

    # model license
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    # Update for the captioning_area
    # for _ in range(3):
    updates.append(gr.update(visible=True))
    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r', encoding='utf-8') as file:
                    corresponding_caption = file.read()

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))

    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)



#Finds the Shorter Side: The if width < height: condition correctly identifies whether the image is portrait or lancape.
#If it's a portrait (height is greater), it sets the shorter side (width) to the target size.
#If it's a landscape or square (width is greater or equal), it sets the shorter side (height) to the target size.

def resize_image(image_path, output_path, size, downscale_only):
    with Image.open(image_path) as img:
        width, height = img.size

        # --- MODE 1: BUCKETING (Downscale Only) ---
        if downscale_only:
            # Only resize if the image's shortest side is LARGER than the target size.
            if min(width, height) <= size:
                print(f"Skipping resize for {os.path.basename(image_path)} ({width}x{height}) as it is not larger than target size {size}.")
                # If the image isn't being resized, we still need to ensure it's at the output path.
                if image_path != output_path:
                    shutil.copy(image_path, output_path)
                return
            # If we proceed, it's a downscale operation.
            print_prefix = f"Downscaling (for bucketing) {os.path.basename(image_path)}"
        
        # --- MODE 2: STANDARD (Resize All) ---
        else:
            # For standard training, we resize everything, up or down.
            print_prefix = f"Resizing {os.path.basename(image_path)}"


        # Common resizing logic for both modes
        if width < height:
            # Portrait
            new_width = size
            new_height = int((size / width) * height)
        else:
            # Landscape or Square
            new_height = size
            new_width = int((size / height) * width)
            
        print(f"{print_prefix} from {width}x{height} to {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)


def create_dataset(destination_folder, size, downscale_only, *inputs):
    print("--- Starting Dataset Creation ---")
    all_files = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 1. Split the input files into images and text files
    image_files = [f for f in all_files if not f.lower().endswith('.txt')]
    txt_files = [f for f in all_files if f.lower().endswith('.txt')]

    # 2. Print the diagnostic info
    print(f"Found {len(image_files)} image file(s).")
    print(f"Found {len(txt_files)} provided caption file(s).")

    # 3. Copy all provided .txt files first, so they will exist
    print("\n--- Step 1: Copying all provided .txt caption files ---")
    for txt_path in txt_files:
        print(f"Copying provided caption: {os.path.basename(txt_path)}")
        shutil.copy(txt_path, destination_folder)

    # 4. Process the images
    print("\n--- Step 2: Processing image files ---")
    for index, image_path in enumerate(image_files):
        # Copy the image to the destination folder
        new_image_path = shutil.copy(image_path, destination_folder)
        image_file_name = os.path.basename(new_image_path)
        print(f"Processing image: {image_file_name}")

        # Resize the image if required
        if size > 0:
            resize_image(new_image_path, new_image_path, size, downscale_only)

        # Determine the expected path for the caption file
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))

        # Check if a caption file already exists (from the previous step)
        if os.path.exists(caption_path):
            print(f"  > Caption file '{caption_file_name}' already exists. Using it.")
            continue  # Move to the next image

        # If it doesn't exist, create one from the UI textbox
        print(f"  > Caption file not found. Creating '{caption_file_name}' from UI.")
        try:
            # Get the caption from the corresponding textbox on the UI
            original_caption = inputs[index + 1]
            with open(caption_path, 'w', encoding='utf-8') as file:
                # Use (original_caption or "") to prevent errors if a caption box is empty/None
                file.write(original_caption or "")
        except IndexError:
            # This is a safety net, but shouldn't happen with this new logic
            print(f"  > !! ERROR: Could not find a caption for image index {index}. Creating empty caption file.")
            with open(caption_path, 'w', encoding='utf-8') as file:
                file.write("")

    print("\n--- Dataset Creation Complete ---")
    return destination_folder

def run_captioning(images, concept_sentence, *captions):
    print(f"run_captioning")
    print(f"concept sentence {concept_sentence}")
    print(f"captions {captions}")
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        print(f"inputs {inputs}")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        print(f"generated_ids {generated_ids}")

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"generated_text: {generated_text}")
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        print(f"parsed_answer = {parsed_answer}")
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        print(f"caption_text = {caption_text}, concept_sentence={concept_sentence}")
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    # download unet
    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        gr.Info(f"Downloading base model: {base_model}. Please wait. (You can check the terminal for the download progress)", duration=None)
        print(f"download {base_model}")
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    # download vae
    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        gr.Info(f"Downloading vae")
        print(f"downloading ae.sft...")
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    # download clip
    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        gr.Info(f"Downloading clip...")
        print(f"download clip_l.safetensors")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    # download t5xxl
    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        print(f"download t5xxl_fp16.safetensors")
        gr.Info(f"Downloading t5xxl...")
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")


def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    base_model,
    output_name,
    resolutionX,
    resolutionY,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components
):

    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, sample_every_n_steps={sample_every_n_steps}")

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
      # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"


    #######################################################
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    if base_model == "flux-dev" or base_model == "flux-schnell":
        model_folder = "models/unet"
    else:
        model_folder = f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)

    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    sh = f"""accelerate launch {line_break}
  --num_processes=1 {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # check dirty
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # dirty
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolutionX,
  resolutionY,
  class_tokens,
  num_repeats
):
    if resolutionY == 0:
        resolutionY = resolutionX

    if resolutionX != resolutionY:
        resolution = f"[{resolutionX}, {resolutionY}]"
    else:
        resolution = f"{resolutionX}"

    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        # only count real image files
        img_exts = {'.jpg','.jpeg','.png','.bmp','.gif','.webp'}
        num_images = sum(
            1 for path in images
            if os.path.splitext(path)[1].lower() in img_exts
        )
        # num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(
    base_model,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    download(base_model)

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use Popen to run the command and capture output in real-time
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    yield from runner.run_command([command], cwd=cwd)
    yield runner.log(f"Runner: {runner}")

    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)


def update(
    base_model,
    lora_name,
    resolutionX,
    resolutionY,
    resize,
    downscale_only_value,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        base_model,
        output_name,
        resolutionX,
        resolutionY,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolutionX,
        resolutionY,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced():
    # if basic_args
    basic_args = {
        'pretrained_model_name_or_path',
        'clip_l',
        't5xxl',
        'ae',
        'cache_latents_to_disk',
        'save_model_as',
        'sdpa',
        'persistent_data_loader_workers',
        'max_data_loader_n_workers',
        'seed',
        'gradient_checkpointing',
        'mixed_precision',
        'save_precision',
        'network_module',
        'network_dim',
        'learning_rate',
        'cache_text_encoder_outputs',
        'cache_text_encoder_outputs_to_disk',
        'fp8_base',
        'highvram',
        'max_train_epochs',
        'save_every_n_epochs',
        'dataset_config',
        'output_dir',
        'output_name',
        'timestep_sampling',
        'discrete_flow_shift',
        'model_prediction_type',
        'guidance_scale',
        'loss_type',
        'optimizer_type',
        'optimizer_args',
        'lr_scheduler',
        'sample_prompts',
        'sample_every_n_steps',
        'max_grad_norm',
        'split_mode',
        'network_args'
    }

    # generate a UI config
    # if not in basic_args, create a simple form
    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':  # Skip the default help argument
            # if the dest is included in basic_args
            args_info[action.dest] = {
                "action": action.option_strings,  # Option strings like '--use_8bit_adam'
                "type": action.type,              # Type of the argument
                "help": action.help,              # Help message
                "default": action.default,        # Default value, if any
                "required": action.required       # Whether the argument is required
            }
    temp = []
    for key in args_info:
        temp.append({ 'key': key, 'action': args_info[key] })
    temp.sort(key=lambda x: x['key'])
    advanced_component_ids = []
    advanced_components = []
    for item in temp:
        key = item['key']
        action = item['action']
        if key in basic_args:
            print("")
        else:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    # radio
                    component = gr.Checkbox()
    #            elif action_type == "<class 'str'>":
    #                component = gr.Textbox()
    #            elif action_type == "<class 'int'>":
    #                component = gr.Number(precision=0)
    #            elif action_type == "<class 'float'>":
    #                component = gr.Number()
    #            elif "int_or_float" in action_type:
    #                component = gr.Number()
                else:
                    component = gr.Textbox(value="")
                if component != None:
                    component.interactive = True
                    component.elem_id = action['action'][0]
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                if action['help'] != None:
                    component.info = action['help']
            advanced_components.append(component)
            advanced_component_ids.append(component.elem_id)
    return advanced_components, advanced_component_ids


theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")



# fpham added to support advanced options ****************************************
# Note: These functions take values and return values, consistent with Gradio event handling
def fill_bucket_parameters_logic(resolution_x_value, resolution_y_value, resize_value, downscale_only, *advanced_component_values): # Takes values
    """Calculates new values for bucket-related parameters and resize."""
    global advanced_component_ids # Use global list to map values/positions
    input_advanced_values_map = dict(zip(advanced_component_ids, advanced_component_values))

    maxres = max(resolution_x_value, resolution_y_value) # Get max reso from input values
    output_advanced_values = []
    # Iterate through global component IDs to ensure output order matches outputs list
    for elem_id in advanced_component_ids:
        if elem_id == "--enable_bucket":
            output_advanced_values.append(True)
        elif elem_id == "--max_bucket_reso":
            output_advanced_values.append(maxres) # Set max reso to resolutionX or y value
        elif elem_id == "--bucket_no_upscale":
            output_advanced_values.append(True)
        else:
            # Keep the original value for others
            output_advanced_values.append(input_advanced_values_map.get(elem_id, None)) # Get from input values map

    new_resize_value = 0 # Set resize to 0 when enabling buckets
    new_downscale_only = True

    # Return the calculated new values corresponding to the outputs list [*advanced_components, resize]
    return (*output_advanced_values, new_resize_value, new_downscale_only)

def disable_buckets_logic(resolution_x_value, resize_value, downscale_only, *advanced_component_values): # Takes values
    """Calculates new values to disable buckets and sets resize to resolutionX."""
    global advanced_component_ids # Use global list to map values/positions
    input_advanced_values_map = dict(zip(advanced_component_ids, advanced_component_values))

    output_advanced_values = []
    # Iterate through global component IDs to ensure output order matches outputs list
    for elem_id in advanced_component_ids:
        if elem_id == "--enable_bucket":
            output_advanced_values.append(False)
        elif elem_id == "--bucket_no_upscale":
            output_advanced_values.append(False)
        else:
            # Keep the original value for others
            output_advanced_values.append(input_advanced_values_map.get(elem_id, None)) # Get from input values map

    new_resize_value = resolution_x_value # Set resize to resolutionX value
    new_downscale_only = False
    # Return the calculated new values corresponding to the outputs list [*advanced_components, resize]
    # To make the UI consistent, we should return the updated advanced values.
    # Assuming the intent was to update the checkboxes in the UI: outputs=[*advanced_components, resize]
    # If outputs is [resize], this function should only return new_resize_value.
    # Let's match the corrected outputs assumption (update checkboxes)
    return (*output_advanced_values, new_resize_value, new_downscale_only)


def save_parameters_logic(
    base_model_value, lora_name_value, resolutionX_value, resolutionY_value, resize_value, downscale_only_value, seed_value,
    workers_value, concept_sentence_value, learning_rate_value, network_dim_value, max_train_epochs_value,
    save_every_n_epochs_value, timestep_sampling_value, guidance_scale_value, vram_value, num_repeats_value,
    sample_prompts_value, sample_every_n_steps_value,
    filename_value, # Filename component value
    *advanced_component_values # Tuple of advanced parameter values
):
    """Saves all relevant parameter values to a JSON file in the configs folder."""

    global advanced_component_ids # Use global list to map values/positions

    # Collect basic parameters
    params = {
        "base_model": base_model_value,
        "lora_name": lora_name_value,
        "resolutionX": resolutionX_value,
        "resolutionY": resolutionY_value,
        "resize": resize_value,
        "downscale_only": downscale_only_value,
        "seed": seed_value,
        "workers": workers_value,
        "concept_sentence": concept_sentence_value,
        "learning_rate": learning_rate_value,
        "network_dim": network_dim_value,
        "max_train_epochs": max_train_epochs_value,
        "save_every_n_epochs": save_every_n_epochs_value,
        "timestep_sampling": timestep_sampling_value,
        "guidance_scale": guidance_scale_value,
        "vram": vram_value,
        "num_repeats": num_repeats_value,
        "sample_prompts": sample_prompts_value,
        "sample_every_n_steps": sample_every_n_steps_value,
    }

    # Collect advanced parameters using the global IDs and input values
    advanced_params = dict(zip(advanced_component_ids, advanced_component_values))
    params["advanced_parameters"] = advanced_params

    # Construct the full save path
    if not filename_value or not filename_value.strip():
        gr.Warning("Please provide a filename to save.")
        return # Nothing to return as outputs=None

    save_filename = filename_value.strip()
    if not save_filename.lower().endswith(".json"):
        save_filename += ".json"
    save_path = os.path.join("configs", save_filename)

    try:
        # Ensure the configs folder exists
        os.makedirs("configs", exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(params, f, indent=4)
        gr.Info(f"Parameters saved to {save_path}")
        # Refresh the dropdown after saving
        updated_json_files = get_json_files()
        return gr.update(choices=updated_json_files, value=save_filename) # Return update for dropdown
    except Exception as e:
        gr.Error(f"Error saving parameters to {save_path}: {e}")
        print(f"Error saving parameters: {e}")
        return gr.update() # Return no update for dropdown on failure


def load_parameters_logic(save_filename_value): # Only need the value from the filename component
    """Loads parameters from a JSON file and returns the values to update the Gradio interface."""

    global advanced_component_ids # Use global list for mapping

    if not save_filename_value or not save_filename_value.strip():
        gr.Warning("Please select or provide a filename to load.")
        # Return None or default values for all outputs
        # Need to return the correct number of outputs corresponding to outputs list
        num_basic_outputs = 18 # Count basic parameters in the outputs list
        num_advanced_outputs = len(advanced_component_ids)
        return (None,) * num_basic_outputs + (None,) * num_advanced_outputs # Return tuple of None

    save_path = os.path.join("configs", save_filename_value.strip())

    try:
        with open(save_path, "r") as f:
            params = json.load(f)

        # Get basic parameter values from the loaded dictionary
        # Use .get() with a default (like None) to handle missing keys gracefully
        base_model_val = params.get("base_model", None)
        lora_name_val = params.get("lora_name", None)
        resolutionX_val = params.get("resolutionX", None)
        resolutionY_val = params.get("resolutionY", None)
        resize_val = params.get("resize", None)
        downscale_only_val = params.get("downscale_only", None)
        seed_val = params.get("seed", None)
        workers_val = params.get("workers", None)
        concept_sentence_val = params.get("concept_sentence", None)
        learning_rate_val = params.get("learning_rate", None)
        network_dim_val = params.get("network_dim", None)
        max_train_epochs_val = params.get("max_train_epochs", None)
        save_every_n_epochs_val = params.get("save_every_n_epochs", None)
        timestep_sampling_val = params.get("timestep_sampling", None)
        guidance_scale_val = params.get("guidance_scale", None)
        vram_val = params.get("vram", None)
        num_repeats_val = params.get("num_repeats", None)
        sample_prompts_val = params.get("sample_prompts", None)
        sample_every_n_steps_val = params.get("sample_every_n_steps", None)

        # Get advanced parameter values from the loaded dictionary
        advanced_params = params.get("advanced_parameters", {})
        output_advanced_values = []
        # Iterate through the global component IDs to ensure output order matches the outputs list
        for elem_id in advanced_component_ids:
             # Get value from the loaded dictionary; default to None if not found
             output_advanced_values.append(advanced_params.get(elem_id, None))

        gr.Info(f"Parameters loaded from {save_path}")

        # Return all basic values followed by all advanced values
        # The order must exactly match the outputs list defined in the .click() event
        return (
            base_model_val, lora_name_val, resolutionX_val, resolutionY_val, resize_val, downscale_only_val,
            seed_val, workers_val, concept_sentence_val, learning_rate_val, network_dim_val,
            max_train_epochs_val, save_every_n_epochs_val, timestep_sampling_val, guidance_scale_val,
            vram_val, num_repeats_val, sample_prompts_val, sample_every_n_steps_val,
            *output_advanced_values # Unpack the collected advanced values into the return tuple
        )

    except FileNotFoundError:
        gr.Error(f"Config file not found: {save_path}")
        # Return None or default values for all outputs
        num_basic_outputs = 18
        num_advanced_outputs = len(advanced_component_ids)
        return (None,) * num_basic_outputs + (None,) * num_advanced_outputs
    except json.JSONDecodeError:
         gr.Error(f"Error decoding JSON file: {save_path}. File might be corrupted.")
         num_basic_outputs = 18
         num_advanced_outputs = len(advanced_component_ids)
         return (None,) * num_basic_outputs + (None,) * num_advanced_outputs
    except Exception as e:
        gr.Error(f"Error loading parameters from {save_path}: {e}")
        print(f"Error loading parameters: {e}")
        # Return None or default values for all outputs
        num_basic_outputs = 18
        num_advanced_outputs = len(advanced_component_ids)
        return (None,) * num_basic_outputs + (None,) * num_advanced_outputs



def get_json_files():
    """Returns a list of JSON files in the configs directory."""
    try:
        configs_dir = "configs"
        # Ensure the directory exists. exist_ok=True prevents error if it exists.
        os.makedirs(configs_dir, exist_ok=True)
        print(f"Ensured directory exists: {configs_dir}") # Debug print

        # List files *after* ensuring the directory exists
        files = [f for f in os.listdir(configs_dir) if f.lower().endswith(".json") and os.path.isfile(os.path.join(configs_dir, f))]
        files.sort()
        print(f"Found {len(files)} json config files in {configs_dir}: {files}") # Debug print
        return files
    except Exception as e:
        print(f"Error listing config files in {configs_dir}: {e}")
        return []


# --- New function to save as Kohya JSON ---
def save_as_kohya_json_logic(
    base_model_value, lora_name_value, resolutionX_value, resolutionY_value, resize_value, downscale_only_value, seed_value,
    workers_value, concept_sentence_value, learning_rate_value, network_dim_value, max_train_epochs_value,
    save_every_n_epochs_value, timestep_sampling_value, guidance_scale_value, vram_value, num_repeats_value,
    sample_prompts_value, sample_every_n_steps_value,
    filename_value, # Filename for the Kohya JSON
    *advanced_component_values # Tuple of advanced parameter values
):
    """Saves current parameters to a JSON file formatted for Kohya's sd-scripts (Flux training)."""

    global advanced_component_ids
    # Create a mapping from advanced component elem_id (with --) to its value
    advanced_values_map = dict(zip(advanced_component_ids, advanced_component_values))

    if not lora_name_value:
        gr.Warning("Please provide a LoRA name before saving as Kohya JSON.")
        return # No outputs

    if not filename_value or not filename_value.strip():
        gr.Warning("Please provide a filename to save the Kohya config.")
        return # No outputs

    # --- Define Kohya JSON Template Structure (Based on provided example) ---
    # Initialize with defaults that match the example or common Kohya values
    kohya_config = {
      "LoRA_type": "Flux1",
      # "LyCORIS_preset": "full", # Skip LyCORIS specific
      "adaptive_noise_scale": 0, # Default based on example
      "additional_parameters": "", # Not in UI
      "ae": resolve_path_without_quotes("models/vae/ae.sft"), # Fixed path
      "apply_t5_attn_mask": False, # Default based on example
      "async_upload": False, # Default based on example
      # Block specific params - Skip
      "bucket_no_upscale": False, # Default based on example
      "bucket_reso_steps": 64, # Default based on example
      # "bypass_mode": False, # Skip
      "cache_latents": False, # Default based on example
      "cache_latents_to_disk": True, # Default based on example
      "caption_dropout_every_n_epochs": 0, # Default based on example
      "caption_dropout_rate": 0, # Default based on example
      "caption_extension": ".txt", # Fixed
      "clip_l": resolve_path_without_quotes("models/clip/clip_l.safetensors"), # Fixed path
      "clip_skip": 1, # Default based on example
      "color_aug": False, # Default based on example
      # "constrain": 0, # Skip
      # Conv specific params - Skip
      "cpu_offload_checkpointing": False, # Default based on example
      "dataset_config": resolve_path_without_quotes(f"outputs/{slugify(lora_name_value)}/dataset.toml"), # Path to dataset config
      "debiased_estimation_loss": False, # Default based on example
      # "decompose_both": False, # Skip
      "dim_from_weights": False, # Default based on example
      "discrete_flow_shift": 3.1582, # Fixed from gen_sh
      # "dora_wd": False, # Skip
      # "double_blocks_to_swap": 0, # Skip
      # "down_lr_weight": "", # Skip
      "dynamo_backend": "no", # Default based on example
      "dynamo_mode": "default", # Default based on example
      "dynamo_use_dynamic": False, # Default based on example
      "dynamo_use_fullgraph": False, # Default based on example
      # "enable_all_linear": False, # Skip
      "enable_bucket": False, # Default based on example
      "epoch": 0, # Will be mapped from max_train_epochs
      "extra_accelerate_launch_args": "", # Skip
      "factor": -1, # Default based on example
      "flip_aug": False, # Default based on example
      "flux1_cache_text_encoder_outputs": False, # Will map from cache_text_encoder_outputs
      "flux1_cache_text_encoder_outputs_to_disk": True, # Will map from cache_text_encoder_outputs_to_disk
      "flux1_checkbox": True, # Appears UI specific
      "fp8_base": True, # Default based on example
      "fp8_base_unet": False, # Default based on example
      "full_bf16": False, # Default based on example
      "full_fp16": False, # Default based on example
      "gpu_ids": "", # Skip
      "gradient_accumulation_steps": 1, # Default based on example
      "gradient_checkpointing": True, # Default based on example
      "guidance_scale": 1.0, # Default based on example
      "highvram": True, # Default based on example (might need to map from advanced if different)
      "huber_c": 0.1, # Default based on example
      "huber_schedule": "snr", # Default based on example
      # Huggingface upload params - Skip
      # Img specific params - Skip
      "in_dims": "", # Skip
      "ip_noise_gamma": 0, # Default based on example
      "ip_noise_gamma_random_strength": False, # Default based on example
      "keep_tokens": 0, # Default based on example
      "learning_rate": 0.0005, # Default based on example
      "log_config": False, # Default based on example
      "log_tracker_config": "", # Skip
      "log_tracker_name": "", # Skip
      "log_with": "", # Skip
      "logging_dir": resolve_path_without_quotes("logs"), # Default Kohya log dir
      "loraplus_lr_ratio": 0, # Skip
      "loraplus_text_encoder_lr_ratio": 0, # Skip
      "loraplus_unet_lr_ratio": 0, # Skip
      "loss_type": "l2", # Default from gen_sh
      "lowvram": False, # Default based on example
      "lr_scheduler": "cosine_with_restarts", # Default based on example
      "lr_scheduler_args": "", # Default based on example
      "lr_scheduler_num_cycles": 3, # Default based on example
      "lr_scheduler_power": 1, # Default based on example
      "lr_scheduler_type": "", # Default based on example, redundant if lr_scheduler is set
      "lr_warmup": 0, # Default based on example
      "lr_warmup_steps": 0, # Default based on example
      "main_process_port": 0, # Default based on example
      "masked_loss": False, # Default based on example
      "max_bucket_reso": 1024, # Default based on example
      "max_data_loader_n_workers": 0, # Default based on example
      "max_grad_norm": 1.0, # Default based on example
      "max_resolution": "512,512", # Default based on example
      "max_timestep": 1000, # Default based on example
      "max_token_length": 75, # Default based on example
      "max_train_epochs": 0, # Default based on example (use 'epoch' instead)
      "max_train_steps": 0, # Default based on example
      "mem_eff_attn": False, # Default based on example
      "mem_eff_save": False, # Default based on example
      # Metadata params - Skip
      "mid_lr_weight": "", # Skip
      "min_bucket_reso": 256, # Default based on example
      "min_snr_gamma": 5, # Default based on example
      "min_timestep": 0, # Default based on example
      "mixed_precision": "bf16", # Default based on example
      "model_list": "custom", # Fixed
      "model_prediction_type": "raw", # Fixed from gen_sh
      "module_dropout": 0, # Default based on example
      "multi_gpu": False, # Default based on example
      "multires_noise_discount": 0.3, # Default based on example
      "multires_noise_iterations": 6, # Default based on example
      "network_alpha": 1, # Default based on example
      "network_dim": 1, # Default based on example
      "network_dropout": 0, # Default based on example
      "network_weights": "", # Skip
      "noise_offset": 0.05, # Default based on example
      "noise_offset_random_strength": False, # Default based on example
      "noise_offset_type": "Multires", # Default based on example
      "num_cpu_threads_per_process": 2, # Default based on example
      "num_machines": 1, # Default based on example
      "num_processes": 1, # Default based on example
      "optimizer": "AdamW8bit", # Default based on example
      "optimizer_args": "", # Default based on example
      "output_dir": resolve_path_without_quotes(f"outputs/{slugify(lora_name_value)}"), # Path to output folder
      "pretrained_model_name_or_path": "", # Will map later
      "prior_loss_weight": 1, # Default based on example
      "random_crop": False, # Default based on example
      "rank_dropout": 0, # Default based on example
      "rank_dropout_scale": False, # Default based on example
      "reg_data_dir": "", # Skip (no reg images in UI)
      "rescaled": False, # Default based on example
      "resume": "", # Default based on example
      "resume_from_huggingface": "", # Default based on example
      "sample_every_n_epochs": 0, # Default based on example
      "sample_every_n_steps": 0, # Default based on example
      "sample_prompts": "", # Default based on example
      "sample_sampler": "euler_a", # Default based on example
      "save_as_bool": False, # Default based on example
      "save_every_n_epochs": 1, # Default based on example
      "save_every_n_steps": 0, # Default based on example
      "save_last_n_epochs": 0, # Default based on example
      "save_last_n_epochs_state": 0, # Default based on example
      "save_last_n_steps": 0, # Default based on example
      "save_last_n_steps_state": 0, # Default based on example
      "save_model_as": "safetensors", # Fixed from gen_sh
      "save_precision": "bf16", # Default based on example
      "save_state": False, # Default based on example
      "save_state_on_train_end": False, # Default based on example
      "save_state_to_huggingface": False, # Default based on example
      "scale_v_pred_loss_like_noise_pred": False, # Default based on example
      "scale_weight_norms": 0, # Default based on example
      "sdxl": False, # Default (assuming not SDXL)
      "sdxl_cache_text_encoder_outputs": False, # Default based on example
      "sdxl_no_half_vae": False, # Default based on example
      "seed": 0, # Default based on example
      "shuffle_caption": False, # Default based on example
      # Single block params - Skip
      "skip_cache_check": False, # Default based on example
      "skip_until_initial_step": False, # Default based on example
      "split_mode": False, # Default based on example
      "split_qkv": False, # Default based on example
      "stop_text_encoder_training": 0, # Default based on example
      "t5xxl": resolve_path_without_quotes("models/clip/t5xxl_fp16.safetensors"), # Fixed path
      "t5xxl_lr": 0, # Default based on example
      "t5xxl_max_token_length": 512, # Default based on example
      "text_encoder_lr": 0, # Default based on example
      "timestep_sampling": "sigmoid", # Default based on example
      "train_batch_size": 2, # Default based on example
      "train_blocks": "all", # Default based on example
      "train_data_dir": resolve_path_without_quotes(f"datasets/{slugify(lora_name_value)}"), # Path to dataset folder
      # Train block indices - Skip
      "train_norm": False, # Default based on example
      "train_on_input": True, # Default based on example
      "train_t5xxl": False, # Default based on example
      "training_comment": "", # Default based on example
      # Txt specific params - Skip
      "unet_lr": 0.0005, # Default based on example
      "unit": 1, # Default based on example
      # "up_lr_weight": "", # Skip
      # "use_cp": False, # Skip
      # "use_scalar": False, # Skip
      # "use_tucker": False, # Skip
      "v2": False, # Default based on example
      "v_parameterization": False, # Default based on example
      "v_pred_like_loss": 0, # Default based on example
      "vae": "", # Default based on example (maybe for SD1.5?), use 'ae' for Flux
      "vae_batch_size": 0, # Default based on example
      # Validation params - Skip
      # Wandb params - Skip
      "weighted_captions": False, # Default based on example
      "xformers": "xformers", # Default based on example
      # Zero* params - Skip
    }

    # --- Populate Basic Parameters (Override template) ---
    # Ensure correct types where needed, default to template value if input is None
    kohya_config["seed"] = int(seed_value) if seed_value is not None else kohya_config.get("seed", 0)
    kohya_config["max_data_loader_n_workers"] = int(workers_value) if workers_value is not None else kohya_config.get("max_data_loader_n_workers", 0)

    # Handle learning_rate carefully - convert string to float
    try:
        if learning_rate_value is not None and learning_rate_value != "":
            kohya_config["learning_rate"] = float(learning_rate_value)
        # Else, keep the template default
    except (ValueError, TypeError):
        gr.Warning(f"Invalid learning rate value: '{learning_rate_value}'. Using default.")
        # Template default is kept if conversion fails

    kohya_config["network_dim"] = int(network_dim_value) if network_dim_value is not None else kohya_config.get("network_dim", 1)
    # Use 'epoch' field for total epochs based on example JSON
    kohya_config["epoch"] = int(max_train_epochs_value) if max_train_epochs_value is not None else kohya_config.get("epoch", 0)
    kohya_config["max_train_epochs"] = 0 # Set max_train_epochs to 0 if using epoch

    kohya_config["save_every_n_epochs"] = int(save_every_n_epochs_value) if save_every_n_epochs_value is not None else kohya_config.get("save_every_n_epochs", 1)
    kohya_config["timestep_sampling"] = timestep_sampling_value if timestep_sampling_value is not None else kohya_config.get("timestep_sampling", "sigmoid")
    kohya_config["guidance_scale"] = float(guidance_scale_value) if guidance_scale_value is not None else kohya_config.get("guidance_scale", 1.0)
    kohya_config["sample_prompts"] = sample_prompts_value if sample_prompts_value is not None else ""
    kohya_config["sample_every_n_steps"] = int(sample_every_n_steps_value) if sample_every_n_steps_value is not None else kohya_config.get("sample_every_n_steps", 0)
    kohya_config["max_resolution"] = f"{int(resolutionX_value) if resolutionX_value is not None else 512},{int(resolutionY_value) if resolutionY_value is not None and resolutionY_value > 0 else int(resolutionX_value) if resolutionX_value is not None else 512}"
    kohya_config["train_batch_size"] = 1 # Fixed as per dataset toml


    # Map specific paths based on base_model_value
    model_config = models.get(base_model_value) # Use .get for safety
    if model_config:
        model_file = model_config.get("file")
        repo = model_config.get("repo")
        if model_file and repo:
            if base_model_value in ["flux-dev", "flux-schnell"]: # Use list check
                model_folder = "models/unet"
            else:
                model_folder = f"models/unet/{repo}"
            kohya_config["pretrained_model_name_or_path"] = resolve_path_without_quotes(os.path.join(model_folder, model_file))
        else:
            gr.Warning(f"Base model '{base_model_value}' config in models.yaml is incomplete. Cannot set pretrained_model_name_or_path.")
            # Template placeholder is kept
    else:
        gr.Warning(f"Unknown base model: {base_model_value}. Cannot set pretrained_model_name_or_path.")
        # Template placeholder is kept


    # --- VRAM based settings (Override template defaults if not in advanced) ---
    # These are the defaults from gen_sh based on VRAM, applied IF NOT OVERRIDDEN in advanced options
    optimizer_type_vram = "adamw8bit"
    split_mode_vram = False
    lr_scheduler_vram = "constant_with_warmup" # Default for adafactor in gen_sh
    max_grad_norm_vram = 1.0 # Default for adamw8bit/general
    network_alpha_vram = int(network_dim_value) if network_dim_value is not None else kohya_config.get("network_dim", 1) # Default alpha == dim
    network_args_train_blocks_vram = "all" # Default

    if vram_value == "16G":
        optimizer_type_vram = "adafactor"
        max_grad_norm_vram = 0.0 # Adafactor specific default
        lr_scheduler_vram = "constant_with_warmup"
    elif vram_value == "12G":
        optimizer_type_vram = "adafactor"
        split_mode_vram = True
        network_args_train_blocks_vram = "single" # Note: this isn't a direct JSON key in example
        max_grad_norm_vram = 0.0 # Adafactor specific default
        lr_scheduler_vram = "constant_with_warmup"

    # Apply VRAM defaults (overwriting template)
    kohya_config["optimizer"] = optimizer_type_vram

    # Note: split_mode and network_args are command line args in gen_sh, not direct JSON keys in the example
    # If we want to represent them in the JSON, we'd need to add them to the template.
    # Let's add them to the template for better representation of the config.

    # Re-define template snippet for added keys
    kohya_config["split_mode"] = False # Added to template
    kohya_config["network_args"] = "" # Added to template (as a string)
    kohya_config["lr_scheduler"] = kohya_config.get("lr_scheduler", "cosine_with_restarts") # Keep template or default
    kohya_config["max_grad_norm"] = kohya_config.get("max_grad_norm", 1.0) # Keep template or default


    # Apply VRAM defaults (overwriting template)
    kohya_config["optimizer"] = optimizer_type_vram
    kohya_config["split_mode"] = split_mode_vram
    if network_args_train_blocks_vram:
        # Format network_args as a string expected by Kohya command line? Or try to parse?
        # Let's just add the flag as a string argument. It might need manual adjustment in Kohya.
        # Or better, find if Kohya JSON has a specific way to represent network_args.
        # The example doesn't show it. Let's skip mapping network_args for simplicity matching the example.
        pass # Skip network_args mapping for now

    # Set LR scheduler based on VRAM logic if not overridden by advanced
    if optimizer_type_vram == "adafactor":
        kohya_config["lr_scheduler"] = lr_scheduler_vram
    # For adamw8bit, the example uses "cosine_with_restarts". Let's use that as default for adamw8bit unless advanced overrides.
    # This is handled below by advanced overrides.

    # Set max_grad_norm based on VRAM logic if not overridden by advanced
    kohya_config["max_grad_norm"] = max_grad_norm_vram


    # Set network_alpha default based on calculated dim
    kohya_config["network_alpha"] = network_alpha_vram


    # --- Override with Advanced Parameters ---
    # Iterate through advanced component IDs. If the corresponding Kohya key exists (after stripping '--'), use the advanced value.
    for elem_id in advanced_component_ids:
        kohya_key = elem_id.lstrip('-') # Remove leading hyphens
        value = advanced_values_map.get(elem_id)

        # Check if this key exists in our kohya_config dictionary (which includes template + VRAM defaults)
        if kohya_key in kohya_config:
             # Convert numbers if the key's current value is a number type or the input looks like a number
             # Handle empty string or None input for numbers by skipping the update
            if isinstance(kohya_config.get(kohya_key), (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                try:
                    if value is not None and value != "":
                        # Convert to int or float based on the value or target type
                        if '.' in str(value):
                            kohya_config[kohya_key] = float(value)
                        elif isinstance(kohya_config.get(kohya_key), float):
                            kohya_config[kohya_key] = float(value) # Convert int strings to float if target is float
                        else:
                            kohya_config[kohya_key] = int(value)
                         # If value is None or empty string, we just don't update,
                         # keeping the VRAM-derived or template default value.
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert advanced param '{elem_id}' value '{value}' to number. Keeping previous value.")
            else:
                # For boolean or string types, just assign the value directly
                # Handle None values - remove the key if value is None? Or leave it? Let's leave it for now.
                kohya_config[kohya_key] = value

        # Handle specific renames/mappings (e.g. cache_text_encoder_outputs -> flux1_cache_text_encoder_outputs)
        # Add these checks *after* the generic mapping, as some might exist in both places
        if elem_id == "--cache_text_encoder_outputs":
            kohya_config["flux1_cache_text_encoder_outputs"] = advanced_values_map.get(elem_id) # Use the advanced value
        if elem_id == "--cache_text_encoder_outputs_to_disk":
            kohya_config["flux1_cache_text_encoder_outputs_to_disk"] = advanced_values_map.get(elem_id) # Use the advanced value
        # Add other renames/special mappings here if needed
        # e.g., if "--network_alpha" was in advanced, it would already map to "network_alpha"

    # --- Final Adjustments / Clean up ---
    # Ensure boolean values are actual booleans.
    # Iterate through keys that are booleans in the template.
    boolean_keys_in_template = {k for k, v in kohya_config.items() if isinstance(v, bool)}
    for key in boolean_keys_in_template:
        val = kohya_config.get(key) # Get value from current config state
        if isinstance(val, str):
            if val.lower() in ['true', '1']:
                 kohya_config[key] = True
            elif val.lower() in ['false', '0']:
                 kohya_config[key] = False
            else:
                 print(f"Warning: Advanced param '{key}' value '{val}' is not a boolean string. Keeping as is.")
        elif isinstance(val, (int, float)):
            kohya_config[key] = bool(val) # Convert non-zero numbers to True, zero to False
        # If val is already bool or None, keep it.

    # Convert some common numbers back to int if they were floats in the template but usually int
    int_keys = ["seed", "network_dim", "epoch", "save_every_n_epochs", "sample_every_n_steps", "max_data_loader_n_workers", "max_train_steps", "max_bucket_reso", "min_bucket_reso"]
    for key in int_keys:
        if key in kohya_config and isinstance(kohya_config[key], (int, float)):
            kohya_config[key] = int(kohya_config[key])

    # Remove None values if Kohya validation might complain (optional, JSON supports null)
    # kohya_config = {k: v for k, v in kohya_config.items() if v is not None}

    # --- Save to JSON File ---
    # Use 'configs_k' folder for consistency with other configs_k
    save_filename = filename_value.strip()
    if not save_filename.lower().endswith(".json"):
        save_filename += ".json"
    save_path = os.path.join("configs_k", save_filename)

    try:
        os.makedirs("configs_k", exist_ok=True) # Ensure configs_k folder exists
        with open(save_path, "w", encoding="utf-8") as f: # Specify encoding
            json.dump(kohya_config, f, indent=4)
        gr.Info(f"Kohya config saved to {save_path}")
    except Exception as e:
        gr.Error(f"Error saving Kohya config to {save_path}: {e}")
        print(f"Error saving Kohya config: {e}")
# end of fpham additions ******************

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row():
                json_files = get_json_files()
                save_filename_dropdown = gr.Dropdown(
                    choices=json_files,
                    label="Select Config File",
                    value=json_files[0] if json_files else None,  # Default value
                    allow_custom_value=True, # Allow typing a new name
                    interactive=True,
                )
                load_button = gr.Button("Load Config", interactive=True)
                save_filename = gr.Textbox(label="Config Filename", value="save.json", interactive=True)
                save_button = gr.Button("Save Config", interactive=True)
                save_kohya_button = gr.Button("Export as Kohya JSON", interactive=True)
            with gr.Row():
                gr.Markdown("Saves/Loads config data (all except images/captions) to/from JSON file in the `/configs` folder. Use 'Export as Kohya JSON' to export for Kohya's training UI in `/configs_k` folder")

            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>
        """, elem_classes="group_padding")
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    print(f"model_names={model_names}")
                    base_model = gr.Dropdown(label="Base model (edit the models.yaml file to add more to this list)", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    with gr.Row():
                        resize = gr.Number(value=512, precision=0, label="Resize Shortest Edge To ... (0 - no resize - for Buckets)", interactive=True, scale=3)
                        downscale_only = gr.Checkbox(value=False, label="Downscale Only (for Bucketing)", scale=1)

                    resolutionX = gr.Number(value=512, precision=0, label="Train Resolution width", interactive=True)
                    resolutionY = gr.Number(value=0, precision=0, label="Train Resolution height (0 = square)", interactive=True)
                    gr.Markdown("Note: For buckets, set Resize to 0, set --bucket_no_upscale, --enable_buckets, and set --max_bucket_reso to the Train resolution width or height whichever is larger.")
                    fill_advanced_button = gr.Button("Quick: Enable Buckets")
                    disable_buckets_button = gr.Button("Quick: Disable Buckets")
                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        images = gr.File(
                            file_types=["image", ".txt"],
                            label="Upload your images",
                            #info="If you want, you can also manually upload caption files that match the image names (example: img0.png => img0.txt)",
                            file_count="multiple",
                            interactive=True,
                            visible=True,
                            scale=1,
                        )
                    with gr.Group(visible=False) as captioning_area:
                        do_captioning = gr.Button("Add AI captions with Florence-2")
                        output_components.append(captioning_area)
                        #output_components = [captioning_area]
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=False, elem_id="start_training")
                    output_components.append(start)
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="--learning_rate", info="Learning Rate", value="8e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(label="--network_dim", info="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                    advanced_components, advanced_component_ids = init_advanced()
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            base_model,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])
    
    dataset_folder = gr.State()

    listeners = [
        base_model,
        lora_name,
        resolutionX,
        resolutionY,
        resize,
        downscale_only,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        *advanced_components
    ]

    # FPHAM added this -->
    # Connect the load button click event
    # Load logic takes filename value, returns many values
    # Outputs list includes basic components and unpacked advanced components
    load_button.click(
        fn=load_parameters_logic,
        inputs=[save_filename], # Input is just the filename value
        outputs=[
            base_model, lora_name, resolutionX, resolutionY, resize, downscale_only, seed, workers,
            concept_sentence, learning_rate, network_dim, max_train_epochs,
            save_every_n_epochs, timestep_sampling, guidance_scale, vram,
            num_repeats, sample_prompts, sample_every_n_steps,
            *advanced_components # Unpack the list of advanced components
        ]
    ).then( # Chain the update function after loading
        fn=update,
        inputs=listeners, # Pass all listener component values to update
        outputs=[train_script, train_config, dataset_folder] # Update script, config, and dataset_folder state
    )

    # Connect Quick Set Buttons
    # Fill buckets logic takes resolutionX, resize, and all advanced component values
    # It returns updated values for all advanced components and resize

    fill_advanced_button.click(
        fn=fill_bucket_parameters_logic,
        inputs=[resolutionX,resolutionY, resize, downscale_only, *advanced_components], # Pass values
        outputs=[*advanced_components, resize, downscale_only] # Update advanced components and resize
    ).then( # Chain the update function to regenerate script/config
         fn=update,
         inputs=listeners, # Pass all listener component values
         outputs=[train_script, train_config, dataset_folder] # Update script, config, and dataset_folder state
    )
    
    # Disable buckets logic takes resolutionX, resize, and all advanced component values
    # It returns updated values for all advanced components and resize (assuming update checkboxes)
    # If outputs=[resize] is desired, change fn to return only new_resize_value and remove *advanced_components from outputs.
    # Let's match the assumed intent of updating the checkboxes based on original code modifying component objects.
    disable_buckets_button.click(
        fn=disable_buckets_logic,
        inputs=[resolutionX, resize, downscale_only,*advanced_components], # Pass values
        outputs=[*advanced_components, resize, downscale_only] # Update advanced components and resize
    ).then( # Chain the update function
         fn=update,
         inputs=listeners, # Pass all listener component values
         outputs=[train_script, train_config, dataset_folder] # Update script, config, and dataset_folder state
    )

    # Connect the dropdown selection to the filename textbox
    save_filename_dropdown.change(
        fn=lambda x: x if x is not None else "", # Set textbox value to selected dropdown value
        inputs=save_filename_dropdown,
        outputs=save_filename
    )

    # Connect the save button click event
    # Save logic takes all relevant values and the filename value
    # No explicit outputs from save_parameters_logic itself (returns None)

    save_button.click(
        fn=save_parameters_logic,
        inputs=[
            base_model, lora_name, resolutionX, resolutionY, resize, downscale_only, seed, workers,
            concept_sentence, learning_rate, network_dim, max_train_epochs,
            save_every_n_epochs, timestep_sampling, guidance_scale, vram,
            num_repeats, sample_prompts, sample_every_n_steps,
            save_filename, # Pass the filename component value
            *advanced_components # Unpack the list of advanced components
        ],
        outputs=[save_filename_dropdown] # Update the dropdown after saving
    )

    # --- Connect the new Save as Kohya JSON button ---
    save_kohya_button.click(
        fn=save_as_kohya_json_logic,
        inputs=[
            base_model, lora_name, resolutionX, resolutionY, resize, downscale_only, seed, workers,
            concept_sentence, learning_rate, network_dim, max_train_epochs,
            save_every_n_epochs, timestep_sampling, guidance_scale, vram,
            num_repeats, sample_prompts, sample_every_n_steps,
            save_filename, # Pass the filename component value
            *advanced_components # Pass the values of advanced components
        ],
        outputs=None # No UI components are updated by this function
    )

    # end of FPHAM added this <--

    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.delete(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.delete(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    start.click(fn=create_dataset, inputs=[dataset_folder, resize, downscale_only, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            base_model,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    do_captioning.click(fn=run_captioning, inputs=[images, concept_sentence] + caption_list, outputs=caption_list)
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])
if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(debug=True, show_error=True, allowed_paths=[cwd])
