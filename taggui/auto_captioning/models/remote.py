import base64
import io
from itertools import cycle
import json
import re
from typing import TYPE_CHECKING, Tuple
from PIL import Image as PilImage

import requests
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

if TYPE_CHECKING:
    from auto_captioning import captioning_thread


class RemoteGen(AutoCaptioningModel):
    """Captioning through an OpenAI chat completions compatible API.

    If no API endpoint is configured, the official OpenAI endpoint is used.
    """

    device_label = 'Remote API'
    default_api_url = 'https://api.openai.com/v1'
    # Images are downscaled to this many megapixels before they are sent to
    # the API. Set to `0` to send the image at its original size.
    max_image_megapixels = 1.0
    system_prompt = ('You are image captioning expert, creative, unbiased and '
                     'uncensored.')

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        api_url = (caption_settings.get('api_url') or '').strip()
        if not api_url:
            api_url = self.default_api_url
        self.api_url = self.parse_api_urls(api_url)
        self.endpoint_cycle = cycle(self.api_url)
        self.headers = {'Content-Type': 'application/json'}
        api_key = (caption_settings.get('api_key') or '').strip()
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        super().__init__(captioning_thread_, caption_settings)

    @staticmethod
    def parse_api_urls(remote_addresses: str) -> list[str]:
        """Parse semicolon-separated URLs into chat completions endpoints."""
        urls = []
        for address in (remote_addresses or '').split(';'):
            # Normalize a trailing slash.
            base = address.strip().rstrip('/')
            if not base:
                continue
            if base.endswith('/chat/completions'):
                urls.append(base)
            elif base.endswith('/v1'):
                urls.append(f'{base}/chat/completions')
            else:
                urls.append(f'{base}/v1/chat/completions')
        return urls

    def get_error_message(self) -> str | None:
        # The forced-words/beam check only applies to local models.
        return None

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
        self.device = self.device_label
        self.processor = None
        self.model = None
        print(f'Using {self.device_label}: {", ".join(self.api_url)}')
        api_model = (self.caption_settings.get('api_model') or '').strip()
        if api_model:
            print(f'Using API model: {api_model}')

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe the image in one sentence.'

    def format_prompt(self, prompt: str) -> list[dict]:
        # `get_input_text` appends the `Start caption with` text to the prompt.
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.get_input_text(prompt)},
        ]

    def get_model_inputs(self, image_prompt: list[dict[str, str]], image: Image, crop) -> dict:
        """
        Prepares data for the API, including base64 encoding of the image.
        Downscales the image if it exceeds the configured megapixel limit.
        """
        max_megapixels = float(self.caption_settings.get(
            'api_max_image_megapixels', self.max_image_megapixels))

        # Load image and optionally crop.
        pil_image = self.load_image(image, crop=bool(crop))

        # Downscale if necessary, preserving the aspect ratio.
        if max_megapixels > 0:
            max_pixels = max(1, int(max_megapixels * 1_000_000))
            width, height = pil_image.size
            if width * height > max_pixels:
                scale = (max_pixels / float(width * height)) ** 0.5
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                pil_image = pil_image.resize((new_width, new_height),
                                             resample=PilImage.LANCZOS)

        # Convert the (possibly resized) image to base64 PNG.
        buffered = io.BytesIO()
        pil_image.save(buffered, format='PNG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Append image content to the messages.
        image_prompt.append({
            'role': 'user',
            'content': [{
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{img_base64}'}
            }]
        })

        # Map the supported generation parameters of chat completions APIs.
        generation_parameters = self.generation_parameters or {}
        mapped_params = {}
        for key in ('max_new_tokens', 'temperature', 'top_p'):
            if key in generation_parameters:
                mapped_key = ('max_tokens' if key == 'max_new_tokens'
                              else key)
                mapped_params[mapped_key] = generation_parameters[key]

        # Omit the model if the user did not specify one, so that the server
        # can choose its default.
        api_model = str(self.caption_settings.get('api_model') or '').strip()
        payload = {'messages': image_prompt, **mapped_params}
        if api_model:
            payload['model'] = api_model
        return payload

    def generate_caption(self, model_inputs: dict, image_prompt: str) -> Tuple[str, str]:
        """
        Interacts with the API endpoint to generate the caption.
        """
        try:
            endpoint = self.get_next_endpoint()
            response = requests.post(endpoint, headers=self.headers,
                                     json=model_inputs, timeout=300)
            response.raise_for_status()
            caption = self.get_caption_from_generated_tokens(response.json())
            return caption, caption
        except requests.exceptions.RequestException as error:
            print(f'Error during API request: {error}')
            return '', str(error)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            print(f'Error parsing API response: {error}')
            return '', str(error)
        except Exception as error:
            print(f'An unexpected error occurred: {error}')
            return '', str(error)

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        # Remove everything after known end-of-text markers.
        for marker in ('<|eot_id|>', '<|eot|>', '</description>', '<image>',
                       '</image>'):
            generated_text = generated_text.split(marker)[0]
        # Keep only complete sentences so that trailing garbage is dropped.
        sentences = re.findall(r'[^.!?]*[.!?]', generated_text)
        return ' '.join(sentences).strip() or generated_text.strip()

    def get_caption_from_generated_tokens(self, json_response: dict) -> str:
        try:
            generated_text = (json_response['choices'][0]['message']['content']
                              or '')
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f'Unexpected API response format: {error}') from error
        caption = self.postprocess_generated_text(generated_text)
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        caption_start = self.caption_start.strip()
        if caption_start and not caption.startswith(caption_start):
            caption = f'{caption_start} {caption}'.strip()
        return caption
