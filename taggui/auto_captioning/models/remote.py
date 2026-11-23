import base64
import io
from itertools import cycle
import re
from typing import TYPE_CHECKING, Tuple

import requests
from PIL import Image as PilImage

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image
from utils.utils import elide_binary_data, truncate_text

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
    # A JPEG keeps the request small; PNG is used when the image has an alpha
    # channel that must be preserved.
    jpeg_quality = 90
    # Connect and read timeouts in seconds.
    request_timeout = (10, 300)
    maximum_error_length = 500

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
        for endpoint in self.api_url:
            if not endpoint.startswith(('http://', 'https://')):
                return ('The API endpoint must start with "http://" or '
                        f'"https://": {endpoint}')
        return None

    def get_next_endpoint(self) -> str:
        return next(self.endpoint_cycle)

    def get_endpoints_in_order(self) -> list[str]:
        """Return the endpoints, starting with the next one in the cycle."""
        start_endpoint = self.get_next_endpoint()
        start_index = self.api_url.index(start_endpoint)
        return self.api_url[start_index:] + self.api_url[:start_index]

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
        print(f'Using API model: {api_model or "(server default)"}')

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe the image in one sentence.'

    def format_prompt(self, prompt: str) -> list[dict]:
        # `get_input_text` appends the `Start caption with` text to the prompt.
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.get_input_text(prompt)},
        ]

    def get_max_image_megapixels(self) -> float:
        try:
            return float(self.caption_settings.get(
                'api_max_image_megapixels', self.max_image_megapixels) or 0)
        except (TypeError, ValueError):
            return self.max_image_megapixels

    def prepare_image(self, image: Image, crop: bool) -> PilImage:
        """Load the image and downscale it to the configured size limit."""
        pil_image = self.load_image(image, crop=bool(crop))
        max_megapixels = self.get_max_image_megapixels()
        if max_megapixels <= 0:
            return pil_image
        max_pixels = max(1, int(max_megapixels * 1_000_000))
        width, height = pil_image.size
        if width * height <= max_pixels:
            return pil_image
        scale = (max_pixels / float(width * height)) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return pil_image.resize(new_size, resample=PilImage.LANCZOS)

    def encode_image_as_data_uri(self, pil_image: PilImage) -> str:
        """Encode a PIL image as a data URI for the chat completions API."""
        has_alpha = (pil_image.mode in ('RGBA', 'LA', 'PA')
                     or 'transparency' in pil_image.info)
        output = io.BytesIO()
        if has_alpha:
            pil_image.save(output, format='PNG', optimize=True)
            mime_type = 'image/png'
        else:
            pil_image.convert('RGB').save(
                output, format='JPEG', quality=self.jpeg_quality,
                optimize=True)
            mime_type = 'image/jpeg'
        encoded = base64.b64encode(output.getvalue()).decode('ascii')
        return f'data:{mime_type};base64,{encoded}'

    def get_mapped_generation_parameters(self) -> dict:
        """Map the supported generation parameters of chat completions APIs."""
        generation_parameters = self.generation_parameters or {}
        mapped_parameters = {}
        for key in ('max_new_tokens', 'temperature', 'top_p'):
            value = generation_parameters.get(key)
            if value is None:
                continue
            mapped_key = 'max_tokens' if key == 'max_new_tokens' else key
            mapped_parameters[mapped_key] = value
        return mapped_parameters

    def get_model_inputs(self, image_prompt: list[dict[str, str]], image: Image,
                         crop) -> dict:
        """Prepare the chat completions payload including the image."""
        pil_image = self.prepare_image(image, crop)
        image_url = self.encode_image_as_data_uri(pil_image)

        # Send the instructions and the image as a single user message.
        messages = list(image_prompt)
        if messages:
            text = messages[-1].get('content') or ''
            user_content = []
            if text:
                user_content.append({'type': 'text', 'text': text})
            user_content.append({
                'type': 'image_url',
                'image_url': {'url': image_url}
            })
            messages[-1] = {'role': 'user', 'content': user_content}

        payload = {'messages': messages,
                   **self.get_mapped_generation_parameters()}
        # Omit the model if the user did not specify one, so that the server
        # can choose its default.
        api_model = str(self.caption_settings.get('api_model') or '').strip()
        if api_model:
            payload['model'] = api_model
        return payload

    def generate_caption(self, model_inputs: dict,
                         image_prompt: str) -> Tuple[str, str]:
        """Interact with the API endpoint to generate the caption."""
        errors = []
        for endpoint in self.get_endpoints_in_order():
            try:
                response = requests.post(endpoint, headers=self.headers,
                                         json=model_inputs,
                                         timeout=self.request_timeout)
            except requests.exceptions.Timeout:
                errors.append(f'{endpoint}: the request timed out.')
                continue
            except requests.exceptions.RequestException as error:
                errors.append(
                    f'{endpoint}: {self.get_request_error_message(error)}')
                continue
            if response.status_code >= 400:
                errors.append(
                    f'{endpoint}: {self.get_http_error_message(response)}')
                continue
            try:
                json_response = response.json()
            except ValueError as error:
                errors.append(
                    f'{endpoint}: the response is not valid JSON ({error}).')
                continue
            try:
                caption = self.get_caption_from_generated_tokens(json_response)
            except ValueError as error:
                errors.append(f'{endpoint}: {error}')
                continue
            console_output = caption + self.describe_token_usage(json_response)
            return caption, console_output
        error_message = self.sanitize_error_message('; '.join(errors))
        print(f'Error during API request: {error_message}')
        # Stop the captioning run instead of repeating a failing request for
        # every selected image.
        self.thread.is_error = True
        self.thread.error_message = error_message
        return '', error_message

    @classmethod
    def sanitize_error_message(cls, message: str) -> str:
        """Make an error message short and readable for the GUI."""
        message = elide_binary_data(str(message))
        message = ' '.join(message.split())
        return truncate_text(message, cls.maximum_error_length)

    @classmethod
    def get_request_error_message(cls, error: Exception) -> str:
        message = str(error).strip()
        if not message:
            return error.__class__.__name__
        return cls.sanitize_error_message(message)

    @classmethod
    def get_http_error_message(cls, response: requests.Response) -> str:
        """Extract a short error message from an unsuccessful response."""
        status = f'HTTP {response.status_code}'
        if response.reason:
            status += f' {response.reason}'
        details = ''
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get('error')
            if isinstance(error, dict):
                details = str(error.get('message') or '')
            elif isinstance(error, str):
                details = error
            if not details:
                details = str(body.get('message') or '')
        if not details:
            details = response.text or ''
        details = cls.sanitize_error_message(details)
        return f'{status}: {details}' if details else status

    @staticmethod
    def describe_token_usage(json_response: dict) -> str:
        """Describe the token usage of the response for the console."""
        usage = json_response.get('usage')
        if not isinstance(usage, dict):
            return ''
        parts = []
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        if isinstance(prompt_tokens, int):
            parts.append(f'{prompt_tokens} prompt')
        if isinstance(completion_tokens, int):
            parts.append(f'{completion_tokens} completion')
        if not parts:
            return ''
        return f' ({", ".join(parts)} tokens)'

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        generated_text = generated_text.strip()
        # Remove quotes that some models put around the whole caption.
        for opening_quote, closing_quote in (('"', '"'), ('“', '”'),
                                             ("'", "'"), ('`', '`')):
            if (len(generated_text) >= 2
                    and generated_text.startswith(opening_quote)
                    and generated_text.endswith(closing_quote)):
                generated_text = generated_text[
                    len(opening_quote):-len(closing_quote)].strip()
        # Remove everything after known end-of-text markers.
        for marker in ('<|eot_id|>', '<|eot|>', '</description>', '<image>',
                       '</image>'):
            generated_text = generated_text.split(marker)[0]
        # Keep only complete sentences so that trailing garbage is dropped.
        sentences = re.findall(r'[^.!?]*[.!?]', generated_text)
        return ' '.join(sentences).strip() or generated_text.strip()

    def get_caption_from_generated_tokens(self, json_response: dict) -> str:
        try:
            content = json_response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f'unexpected response format ({error})') from error
        if isinstance(content, list):
            # Some APIs return the content as a list of parts.
            content = ''.join(part.get('text') or ''
                              for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            raise ValueError('unexpected response format '
                             '(no message content)')
        caption = self.postprocess_generated_text(content)
        if not caption:
            raise ValueError('the API returned an empty caption')
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        caption_start = self.caption_start.strip()
        if caption_start and not caption.startswith(caption_start):
            caption = f'{caption_start} {caption}'.strip()
        return caption
