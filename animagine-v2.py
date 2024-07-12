import logging
from telegram import Update
from telegram.ext import Application, filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import torch
from diffusers import DiffusionPipeline
import datetime
import random
import threading
import queue
import asyncio
import time

# Initialize the lock
process_lock = threading.Lock()

torch.cuda.empty_cache()
#pipe initialization
pipe = DiffusionPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1", 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
)
pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

task_queue = queue.Queue()

def validate_input(input):
    if input == "":
        return False
    if input.isspace():
        return False
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

async def rf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input = update.message.text
    input = input.replace("/rf ", "")
    if not validate_input(input):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="bad input")
        return

    address = f"/home/penglaishan/aigen/animagine_3.1_{input}.jpg"
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=address)

async def stable_diffusion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input = update.message.text

    if not validate_input(input):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="bad input")
        return
    
    if input.startswith("/sd "):
        input = input.replace("/sd ", "")
        type = "sd"
    elif input.startswith("/sd2 "):
        input = input.replace("/sd2 ", "")
        type = "sd2"
    elif input.startswith("/sd_fast "):
        input = input.replace("/sd_fast ", "")
        type = "sd_fast"
    elif input.startswith("/sd2_fast "):
        input = input.replace("/sd2_fast ", "")
        type = "sd2_fast"
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="bad input")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = timestamp + "_" + str(random.randint(0, 1000))
    num_tasks = threading.active_count() - 1
    message = "Your task is added to the queue (queue size: "+ str(num_tasks) +") task ID: " + timestamp
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)
    worker_thread = threading.Thread(target=process_stable_diffusion_worker)
    if not worker_thread.is_alive():
        worker_thread.start()
    task_queue.put({"update": update, "input": input, "timestamp": timestamp, "context": context, "type": type})

def process_stable_diffusion_worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        with process_lock:
            asyncio.run(process_stable_diffusion(task))
        task_queue.task_done()

async def process_stable_diffusion(task):
    print("Processing task start")
    torch.cuda.empty_cache()
    global gpu_busy
    gpu_busy = True
    update = task["update"]
    input = task["input"]
    timestamp = task["timestamp"]
    context = task["context"]
    type = task["type"]

    if task["type"] == "sd":
        num_inference_steps = 28
        width = 1152
        height = 896
    elif task["type"] == "sd2":
        num_inference_steps = 28
        width = 896
        height = 1152
    elif task["type"] == "sd_fast":
        num_inference_steps = 20
        width = 1152
        height = 896
    elif task["type"] == "sd2_fast":
        num_inference_steps = 20
        width = 896
        height = 1152

    prompt = str(input)
    negative_prompt = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=num_inference_steps
    ).images[0]

    address_txt = f"/home/penglaishan/aigen/txt/{timestamp}.txt"
    open(address_txt, 'w').write(prompt)
    address = f"/home/penglaishan/aigen/animagine_3.1_{timestamp}.jpg"
    image.save(address, format='jpeg', quality=95)
    gpu_busy = False
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=address, caption="prompt: "+prompt+" / task ID: "+timestamp)

def main():
    app = Application.builder().token('CHANGE_TO_YOUR_TOKEN!').build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("rf", rf))
    app.add_handler(CommandHandler("sd", stable_diffusion))
    app.add_handler(CommandHandler("sd2", stable_diffusion))
    app.add_handler(CommandHandler("sd_fast", stable_diffusion))
    app.add_handler(CommandHandler("sd2_fast", stable_diffusion))
    
    app.run_polling()

if __name__ == "__main__":
    main()
