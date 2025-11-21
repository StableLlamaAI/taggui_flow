import base64
import io
import json
import re
from typing import Tuple

from auto_captioning.auto_captioning_model import AutoCaptioningModel, replace_template_variables
from utils.image import Image
import auto_captioning.captioning_thread as captioning_thread

try:
    from openai import OpenAI
    has_openai = True
except ImportError:
    has_openai = False
    print("OpenAI package not installed. Install with: pip install openai")


class OpenAIModel(AutoCaptioningModel):
    """Model class for interacting with OpenAI's API for image captioning."""

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        # Don't call super().__init__() yet - we need to set some things first
        self.thread = captioning_thread_
        self.thread_parent = captioning_thread_.parent()
        self.caption_settings = caption_settings
        self.model_id = caption_settings['model_id']
        self.prompt = caption_settings['prompt']
        self.skip_hash = caption_settings['skip_hash']
        self.caption_start = caption_settings['caption_start']
        self.remove_tag_separators = caption_settings['remove_tag_separators']
        self.generation_parameters = caption_settings['generation_parameters']

        if not has_openai:
            raise ImportError("OpenAI package is required. Install with: pip install openai")

        # Get API settings
        api_key = caption_settings.get('api_key', '')
        api_url = caption_settings.get('api_url', '').strip()

        # Initialize OpenAI client
        if api_url:
            # Custom endpoint (e.g., Azure OpenAI or compatible services)
            self.client = OpenAI(api_key=api_key, base_url=api_url)
        else:
            # Standard OpenAI endpoint
            self.client = OpenAI(api_key=api_key)

        # If model_id is still the placeholder (user hasn't selected a specific model),
        # default to gpt-4o for vision tasks
        if self.model_id in ('OpenAI', 'OpenAI compatible API'):
            self.model_id = 'gpt-4o'

        # Set device for display purposes
        self.device = 'OpenAI API'

        # No local processor or model needed
        self.processor = None
        self.model = None
        self.tokenizer = None

    def get_processor(self):
        return None

    def get_model_load_arguments(self) -> dict:
        return {}  # Not used for API models

    def load_model(self, model_load_arguments):
        return None

    def get_model(self):
        return None  # No local model for API interaction

    def load_processor_and_model(self):
        """Override to prevent loading from HuggingFace. OpenAI uses API only."""
        # Set device to indicate we're using an API
        self.device = 'OpenAI API'
        # No processor or model to load for API-based captioning
        self.processor = None
        self.model = None
        print(f'Using OpenAI API with model: {self.model_id}')

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'

    def get_input_text(self, image_prompt: str) -> str:
        """Combine prompt with caption start."""
        if image_prompt and self.caption_start:
            return f'{image_prompt} {self.caption_start}'
        return image_prompt or self.caption_start

    def get_image_prompt(self, image: Image) -> str:
        """Get the prompt for the image, with template variables replaced."""
        if self.prompt:
            image_prompt = replace_template_variables(self.prompt, image, self.skip_hash)
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        return image_prompt

    def encode_image_to_base64(self, image: Image, crop: bool) -> str:
        """Load and encode image to base64 for OpenAI API."""
        pil_image = self.load_image(image, crop)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def get_model_inputs(self, image_prompt: str, image: Image, crop: bool) -> dict:
        """
        Prepares data for the OpenAI API in the correct format.
        Returns a dict that will be used to call the API.
        """
        # Get the full prompt text
        input_text = self.get_input_text(image_prompt)

        # Encode image
        image_url = self.encode_image_to_base64(image, crop)

        # Build messages in OpenAI format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]

        return {
            "messages": messages,
            "image_prompt": image_prompt  # Store for later use
        }

    def generate_caption(self, model_inputs: dict, image_prompt: str) -> Tuple[str, str]:
        """
        Interacts with the OpenAI API to generate the caption.
        """
        try:
            # Extract messages from model_inputs
            messages = model_inputs["messages"]

            # Map generation parameters to OpenAI API parameters
            api_params = {
                "model": self.model_id,
                "messages": messages,
            }

            # Map common parameters to OpenAI equivalents
            gen_params = self.generation_parameters
            if 'max_new_tokens' in gen_params:
                api_params['max_tokens'] = gen_params['max_new_tokens']
            if 'temperature' in gen_params:
                api_params['temperature'] = gen_params['temperature']
            if 'top_p' in gen_params:
                api_params['top_p'] = gen_params['top_p']

            # Call OpenAI API
            response = self.client.chat.completions.create(**api_params)

            # Extract the generated text
            generated_text = response.choices[0].message.content

            # Post-process the generated text
            caption = self.postprocess_generated_text(generated_text)

            # Remove tag separators if requested
            if self.remove_tag_separators:
                caption = caption.replace(self.thread.tag_separator, ' ')

            console_output_caption = caption
            return caption, console_output_caption

        except Exception as e:
            error_msg = f"Error during OpenAI API request: {e}"
            print(error_msg)
            return "", error_msg

    @staticmethod
    def postprocess_generated_text(generated_text: str) -> str:
        """Clean up generated text by removing common artifacts."""
        # Remove common model artifacts
        generated_text = generated_text.strip()

        # Remove markdown code blocks if present
        generated_text = re.sub(r'^```.*?\n', '', generated_text)
        generated_text = re.sub(r'\n```$', '', generated_text)

        return generated_text
