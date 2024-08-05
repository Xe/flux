<div align="center">
    <h1>Flux GPU Demo</h1>
    <p>Run <strong><a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell">Flux</a></strong> as a Replicate Cog on Fly.io!</p>
</div>

![Untitled](https://github.com/user-attachments/assets/fec35726-e1c2-48a3-904c-c8c50daa6b54)


Flux is one of the most advanced text-to-image model families yet. This demo exposes the Schenll or Dev model via a simple HTTP server, thanks to [Replicate Cog](https://github.com/replicate/cog). Cog is an open-source tool that lets you package machine learning models in a standard, production-ready container. When you're up and running, you can generate images using the `/predictions` endpoint. Images are automatically stored in object-storage on [Tigris](https://www.tigrisdata.com/) (you'll need to make sure you add a bucket to the app).

## Deploy to Fly.io

<!-- > [!IMPORTANT]  
> Before you deploy, you'll need to get access to the model on Hugging Face by filling out the form in the model [repo](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main). -->

```sh
fly apps create --name [APP_NAME]

fly storage create

fly secrets set HUGGING_FACE_HUB_TOKEN=<HUGGING_FACE_TOKEN>

cog push registry.fly.io/[APP_NAME]:latest --use-cuda-base-image false

```
Now replace `image` in your fly.toml, then:

```sh
fly deploy
```

## Example Request

```sh
curl --silent https://cog-flux.fly.dev/predictions/test \
    --request PUT \
    --header "Content-Type: application/json" \
    --data @- << 'EOF' | jq
{
    "input": {
        "prompt": "Starry night dreamscape with purple hot-air balloons",
        "aspect_ratio": "16:9",
        "guidance_scale": 3.5,
        "num_inference_steps": 50,
        "max_sequence_length": 512,
        "output_format": "png"
    }
}
EOF
```

Now view your image at `https://fly.storage.tigris.dev/[BUCKET_NAME]/[OUTPUT_IMAGE_NAME]`

## Having trouble?

Create an issue or ask a question here: https://community.fly.io/
