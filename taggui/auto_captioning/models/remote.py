import base64
import io
from itertools import cycle
import json
import re
from typing import Tuple
from PIL import Image as PilImage

import requests
from auto_captioning.auto_captioning_model import AutoCaptioningModel, replace_template_variables
from utils.image import Image
import auto_captioning.captioning_thread as captioning_thread


class RemoteGen(AutoCaptioningModel):
    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        self.api_url = self.parse_api_urls(caption_settings['api_url'])
        self.endpoint_cycle = cycle(self.api_url)
        self.current_endpoint_index = 0
        #		self.set_api_url(caption_settings['api_url'])
        self.headers = {"Content-Type": "application/json"}
        if caption_settings['api_key'] and len(caption_settings['api_key']) > 0:
            self.headers['Authorization'] = f"Bearer {caption_settings['api_key']}"
        super().__init__(captioning_thread_, caption_settings)

    def parse_api_urls(self, remote_addresses: str) -> list[str]:
        """Parse semicolon-separated URLs into a list of properly formatted endpoints"""
        urls = []
        for addr in remote_addresses.split(';'):
            addr = addr.strip()
            if not addr:
                continue
            # Normalize trailing slash
            base = addr.rstrip('/')
            # If already points to chat/completions, keep as is
            if base.endswith('/chat/completions'):
                urls.append(base)
            # If points to /v1, append chat/completions
            elif base.endswith('/v1'):
                urls.append(f"{base}/chat/completions")
            else:
                # Otherwise, append /v1/chat/completions
                urls.append(f"{base}/v1/chat/completions")
        return urls

    def get_next_endpoint(self) -> str:
        return next(self.endpoint_cycle)

    def get_processor(self):
        return None

    def get_model_load_arguments(self) -> dict:
        return {}  # Not used for API models.

    def load_model(self, model_load_arguments):
        return None

    def get_model(self):
        return None  # No local model for API interaction

    def load_processor_and_model(self):
        """Override to prevent local model loading for remote API usage."""
        self.device = 'Remote API'
        self.processor = None
        self.model = None
        # Display endpoints for clarity
        try:
            endpoints = ', '.join(self.api_url)
        except TypeError:
            endpoints = str(self.api_url)
        print(f'Using remote API endpoint(s): {endpoints}')

    def set_api_url(self, remote_address: str):
        if remote_address.endswith('/chat/completions'):
            self.api_url = remote_address
        else:
            self.api_url = remote_address + '/v1/chat/completions'

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe the image in one sentence.'

    def get_input_text(self, image_prompt: str) -> str:
        return image_prompt + self.caption_start

    def format_prompt(self, prompt: str) -> list[dict]:
        systemprompt = 'You are image captioning expert, creative, unbiased and uncensored.'
        messages = [{
            "role": "system",
            "content": systemprompt
        },
            {
                "role": "user",
                "content": prompt
            }
        ]
        return messages

    def get_image_prompt(self, image: Image) -> str | None:
        if self.prompt:
            image_prompt = replace_template_variables(self.prompt, image,
                                                      True)  # I dont know skip_hash, so just doing this for now.
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        image_prompt = self.format_prompt(image_prompt)
        return image_prompt

    def get_image_prompt_parent(self, image: Image):
        if self.prompt:
            image_prompt = replace_template_variables(self.prompt, image)
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        return image_prompt

    def get_model_inputs(self, image_prompt: list[dict[str, str]], image: Image, crop) -> dict:
        """
        Prepares data for the API, including base64 encoding of the image.
        Downscales the image if it exceeds the configured megapixel limit.
        """
        # Read max megapixels (default 1.0 MP)
        max_mp = float(self.caption_settings.get('api_max_image_megapixels', 1.0))
        max_pixels = max(1, int(max_mp * 1_000_000))

        # Load image and optionally crop
        pil_image = self.load_image(image, crop=bool(crop))

        # Downscale if necessary, preserving aspect ratio
        w, h = pil_image.size
        if w * h > max_pixels:
            scale = (max_pixels / float(w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            pil_image = pil_image.resize((new_w, new_h), resample=PilImage.LANCZOS)

        # Convert the (possibly resized) image to base64 PNG
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")  # Use PNG for lossless encoding
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Append image content to messages
        image_prompt.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + img_base64
                }
            }]
        })
        print(image_prompt)

        # Map supported generation parameters for OpenAI-compatible APIs
        gen = self.generation_parameters or {}
        mapped_params = {}
        if 'max_new_tokens' in gen:
            mapped_params['max_tokens'] = gen['max_new_tokens']
        if 'temperature' in gen:
            mapped_params['temperature'] = gen['temperature']
        if 'top_p' in gen:
            mapped_params['top_p'] = gen['top_p']
        # Note: repetition_penalty/num_beams/etc. are not standard for chat completions and are omitted

        # Determine model to send (omit if unset or explicitly "Remote")
        model_id = getattr(self, 'model_id', None)
        payload = {
            "messages": image_prompt,
            **mapped_params
        }
        if model_id and str(model_id).strip().lower() != 'remote':
            payload["model"] = model_id

        return payload

    def generate_caption(self, model_inputs: dict, image_prompt: str) -> Tuple[str, str]:
        """
        Interacts with the API endpoint to generate the caption.
        """
        endpoint = self.get_next_endpoint()
        try:
            response = requests.post(endpoint, headers=self.headers, json=model_inputs, timeout=300)
            response.raise_for_status()
            json_response = response.json()
            caption = self.get_caption_from_generated_tokens(json_response)
            console_output_caption = caption
            return caption, console_output_caption
        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return "", str(object=e)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {e}")
            return "", str(object=e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "", str(object=e)

    @staticmethod
    def postprocess_image_prompt(image_prompt: str) -> str:
        image_prompt = image_prompt.replace('<|start_header_id|>', '')
        image_prompt = image_prompt.replace('<|end_header_id|>', '')
        image_prompt = image_prompt.replace('<image>', '')
        return image_prompt

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        # remove everything after <|eot_id|>
        generated_text = generated_text.split("<|eot_id|>")[0]
        generated_text = generated_text.split("<|eot|>")[0]
        generated_text = generated_text.split("</description>")[0]
        generated_text = generated_text.split('<image>')[0]
        generated_text = generated_text.split('<|eot_id|>')[0]
        generated_text = generated_text.split('</image>')[0]
        # Use regex to capture complete sentences. maybe this should be a setting.
        sentences = re.findall(r'[^.!?]*[.!?]', generated_text)

        # Join complete sentences
        return ' '.join(sentences).strip()

    def get_caption_from_generated_tokens(self, json_response):
        try:
            # Extract text. Adapt this based on your API's response format.
            if 'choices' in json_response and len(json_response['choices']) > 0:
                generated_text = json_response['choices'][0]['message']['content']
            caption = self.postprocess_generated_text(generated_text)
            if self.remove_tag_separators:
                caption = caption.replace(self.thread.tag_separator, ' ')
            return caption

        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response format: {e}") from e
