from typing import TYPE_CHECKING

from auto_captioning.models.remote import RemoteGen

if TYPE_CHECKING:
    from auto_captioning import captioning_thread


class OpenAIModel(RemoteGen):
    """Captioning through OpenAI or an OpenAI-compatible API.

    The chat completions protocol is identical to `RemoteGen`, so this class
    only changes the defaults: the official OpenAI endpoint is used when no API
    endpoint is configured, the model defaults to `gpt-4o`, and images are sent
    at their original resolution.
    """

    device_label = 'OpenAI API'
    max_image_megapixels = 0
    system_prompt = ('You are an AI assistant specialized in describing images '
                     'accurately and concisely.')

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        caption_settings = dict(caption_settings)
        if not caption_settings.get('api_url', '').strip():
            caption_settings['api_url'] = 'https://api.openai.com/v1'
        super().__init__(captioning_thread_, caption_settings)
        # `model_id` is the placeholder when the user has not picked a specific
        # model yet.
        if self.model_id in ('', 'OpenAI', 'OpenAI compatible API'):
            self.model_id = 'gpt-4o'

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'
