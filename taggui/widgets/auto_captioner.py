import sys
from pathlib import Path

import requests
from PySide6.QtCore import (QModelIndex, QObject, QThread, Qt, QTimer, Signal,
                            Slot)
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QAbstractScrollArea, QCheckBox, QComboBox,
                               QDockWidget, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.models.remote import RemoteGen
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models_list import MODELS, get_model_class
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import ConsoleTextEdit, TallPushButton
from utils.enums import CaptionDevice, CaptionPosition
from utils.settings import DEFAULT_SETTINGS, settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import elide_binary_data, pluralize, truncate_text
from widgets.image_list import ImageList


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


class ApiModelFetcher(QThread):
    """Fetch the available models from an OpenAI-compatible endpoint."""

    models_fetched = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, api_url: str, api_key: str,
                 parent: QObject | None = None):
        super().__init__(parent)
        self.api_url = api_url
        self.api_key = api_key

    def run(self):
        try:
            endpoints = RemoteGen.parse_api_urls(
                self.api_url or RemoteGen.default_api_url)
            models_url = (endpoints[0].removesuffix('/chat/completions')
                          + '/models')
            headers: dict[str, str] = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            response = requests.get(models_url, headers=headers,
                                    timeout=(10, 30))
            if response.status_code >= 400:
                self.error_occurred.emit(
                    RemoteGen.get_http_error_message(response))
                return
            json_response = response.json()
            model_ids = sorted(
                item['id'] for item in json_response.get('data', [])
                if isinstance(item, dict) and item.get('id'))
            # Prefer vision-capable models, but show everything when no known
            # vision model names are returned (e.g. for local endpoints).
            vision_keywords = ('gpt-4', 'vision', 'multimodal', 'gemini')
            vision_model_ids = [
                model_id for model_id in model_ids
                if any(keyword in model_id.lower()
                       for keyword in vision_keywords)]
            self.models_fetched.emit(vision_model_ids or model_ids)
        except requests.exceptions.Timeout:
            self.error_occurred.emit('The request timed out.')
        except requests.exceptions.RequestException as error:
            self.error_occurred.emit(RemoteGen.get_request_error_message(error))
        except (ValueError, KeyError, TypeError) as error:
            self.error_occurred.emit(f'Unexpected response format: {error}')
        except Exception as error:
            self.error_occurred.emit(elide_binary_data(str(error)))


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        try:
            import bitsandbytes
            self.is_bitsandbytes_available = True
        except RuntimeError:
            self.is_bitsandbytes_available = False
        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.model_combo_box = FocusedScrollSettingsComboBox(key='model_id')
        # Use the full width of the dock so that long model identifiers are
        # readable.
        self.model_combo_box.setSizePolicy(QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Fixed)
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(MODELS)

        # Settings that are only relevant for API models. They are kept in a
        # separate container so that they can be hidden as a whole.
        self.api_settings_container = QWidget()
        api_settings_form = QFormLayout(self.api_settings_container)
        api_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        api_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.remote_address_line_edit = SettingsLineEdit(
            key='remote_address', default='')
        self.remote_address_line_edit.setPlaceholderText(
            'Empty: https://api.openai.com/v1')
        self.remote_address_line_edit.setClearButtonEnabled(True)
        self.remote_address_line_edit.setToolTip(
            'Base URL of the service, e.g. http://localhost:11434/v1.\n'
            'Leave it empty to use the official OpenAI endpoint.\n'
            'Separate multiple URLs with ";" to distribute the requests '
            'between them.')

        self.api_key_line_edit = SettingsLineEdit(key='api_key', default='')
        self.api_key_line_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_line_edit.setPlaceholderText('Empty for local servers')
        self.api_key_line_edit.setToolTip(
            'API key of the service. Local servers often do not need one.')
        api_key_container = QWidget()
        api_key_layout = QHBoxLayout(api_key_container)
        api_key_layout.setContentsMargins(0, 0, 0, 0)
        api_key_layout.addWidget(self.api_key_line_edit, 1)
        self.show_api_key_check_box = QCheckBox('Show')
        self.show_api_key_check_box.setToolTip('Show the API key in plain text')
        api_key_layout.addWidget(self.show_api_key_check_box)

        # The model can be picked from the fetched list or typed manually.
        # It is empty by default so that the `model` field can be omitted.
        self.api_model_combo_box = QComboBox()
        self.api_model_combo_box.setEditable(True)
        self.api_model_combo_box.setInsertPolicy(
            QComboBox.InsertPolicy.NoInsert)
        self.api_model_combo_box.setEditText(
            str(settings.value('api_model', defaultValue='', type=str)))
        self.api_model_combo_box.setToolTip(
            'Model name to request from the API, e.g. gpt-4o.\n'
            'Leave it empty to use the default model of the server.')
        api_model_line_edit = self.api_model_combo_box.lineEdit()
        if api_model_line_edit is not None:
            api_model_line_edit.setPlaceholderText(
                'Empty: server default, e.g. gpt-4o')
        self.api_model_container = QWidget()
        api_model_layout = QHBoxLayout(self.api_model_container)
        api_model_layout.setContentsMargins(0, 0, 0, 0)
        api_model_layout.addWidget(self.api_model_combo_box, 1)
        self.refresh_api_models_button = QPushButton('Refresh')
        self.refresh_api_models_button.setToolTip(
            'Load the list of available models from the API endpoint')
        api_model_layout.addWidget(self.refresh_api_models_button)

        self.api_max_image_megapixels_spin_box = (
            FocusedScrollSettingsDoubleSpinBox(
                key='api_max_image_megapixels', default=1.0, minimum=0,
                maximum=64))
        self.api_max_image_megapixels_spin_box.setSingleStep(0.5)
        self.api_max_image_megapixels_spin_box.setToolTip(
            'Images are downscaled to this many megapixels before they are '
            'sent to the API. Larger images cost more tokens and take longer '
            'to upload; 0 sends the original size.')

        self.api_status_label = QLabel()
        self.api_status_label.hide()

        api_settings_form.addRow('API endpoint',
                                 self.remote_address_line_edit)
        api_settings_form.addRow('API key', api_key_container)
        api_settings_form.addRow('API model', self.api_model_container)
        api_settings_form.addRow('Max image size (MP)',
                                 self.api_max_image_megapixels_spin_box)
        api_settings_form.addRow(self.api_status_label)

        self._api_model_fetcher: ApiModelFetcher | None = None
        self._api_model_fetch_timer = QTimer(self)
        self._api_model_fetch_timer.setSingleShot(True)
        self._api_model_fetch_timer.setInterval(500)
        self._api_model_fetch_timer.timeout.connect(self.fetch_api_models)

        # The prompt field grows with its content so that long prompts are
        # fully visible instead of being clipped after a few lines.
        self.prompt_text_edit = SettingsPlainTextEdit(
            key='prompt', minimum_line_count=3, maximum_line_count=12)
        self.prompt_text_edit.setTabChangesFocus(True)
        self.prompt_text_edit.setToolTip(
            'Instructions given to the model.\n'
            'Template variables: {tags}, {name} and {directory} (or '
            '{folder}).')
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self.skip_hash_container = QWidget()
        skip_hash_layout = QHBoxLayout()
        skip_hash_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        skip_hash_layout.setContentsMargins(0, 0, 0, 0)
        self.skip_hash_check_box = SettingsBigCheckBox(
            key='skip_hash', default=True)
        skip_hash_layout.addWidget(QLabel('Skip hash tags when inserting in prompt'))
        skip_hash_layout.addWidget(self.skip_hash_check_box)
        self.skip_hash_container.setLayout(skip_hash_layout)
        self.device_combo_box = FocusedScrollSettingsComboBox(key='device')
        self.device_combo_box.addItems(list(CaptionDevice))
        self.load_in_4_bit_container = QWidget()
        load_in_4_bit_layout = QHBoxLayout()
        load_in_4_bit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        load_in_4_bit_layout.setContentsMargins(0, 0, 0, 0)
        self.load_in_4_bit_check_box = SettingsBigCheckBox(
            key='load_in_4_bit', default=True)
        load_in_4_bit_layout.addWidget(QLabel('Load in 4-bit'))
        load_in_4_bit_layout.addWidget(self.load_in_4_bit_check_box)
        self.load_in_4_bit_container.setLayout(load_in_4_bit_layout)
        self.limit_to_crop_container = QWidget()
        limit_to_crop_layout = QHBoxLayout()
        limit_to_crop_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        limit_to_crop_layout.setContentsMargins(0, 0, 0, 0)
        self.limit_to_crop_check_box = SettingsBigCheckBox(
            key='limit_to_crop', default=True)
        limit_to_crop_layout.addWidget(QLabel('Limit to crop'))
        limit_to_crop_layout.addWidget(self.limit_to_crop_check_box)
        self.limit_to_crop_container.setLayout(limit_to_crop_layout)
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        basic_settings_form.addRow('Model', self.model_combo_box)
        basic_settings_form.addRow(self.api_settings_container)

        # Fetch the model list (debounced) when the API settings change.
        self.remote_address_line_edit.textChanged.connect(
            self.schedule_api_models_fetch)
        self.api_key_line_edit.textChanged.connect(
            self.schedule_api_models_fetch)
        self.refresh_api_models_button.clicked.connect(self.fetch_api_models)
        self.show_api_key_check_box.toggled.connect(
            self.set_api_key_visibility)

        self.prompt_label = QLabel('Prompt')
        basic_settings_form.addRow(self.prompt_label, self.prompt_text_edit)
        self.caption_start_label = QLabel('Start caption with')
        basic_settings_form.addRow(self.caption_start_label,
                                   self.caption_start_line_edit)
        basic_settings_form.addRow('Caption position',
                                   self.caption_position_combo_box)
        basic_settings_form.addRow(self.skip_hash_container)
        self.device_label = QLabel('Device')
        basic_settings_form.addRow(self.device_label, self.device_combo_box)
        basic_settings_form.addRow(self.load_in_4_bit_container)
        basic_settings_form.addRow(self.remove_tag_separators_container)
        basic_settings_form.addRow(self.limit_to_crop_container)

        self.wd_tagger_settings_form_container = QWidget()
        wd_tagger_settings_form = QFormLayout(
            self.wd_tagger_settings_form_container)
        wd_tagger_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        wd_tagger_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.show_probabilities_check_box = SettingsBigCheckBox(
            key='wd_tagger_show_probabilities', default=True)
        self.min_probability_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='wd_tagger_min_probability', default=0.4, minimum=0.01,
            maximum=1)
        self.min_probability_spin_box.setSingleStep(0.01)
        self.max_tags_spin_box = FocusedScrollSettingsSpinBox(
            key='wd_tagger_max_tags', default=30, minimum=1, maximum=999)
        tags_to_exclude_form = QFormLayout()
        tags_to_exclude_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        tags_to_exclude_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.tags_to_exclude_text_edit = SettingsPlainTextEdit(
            key='wd_tagger_tags_to_exclude', minimum_line_count=2,
            maximum_line_count=10)
        tags_to_exclude_form.addRow('Tags to exclude',
                                    self.tags_to_exclude_text_edit)
        wd_tagger_settings_form.addRow('Show probabilities',
                                       self.show_probabilities_check_box)
        wd_tagger_settings_form.addRow('Minimum probability',
                                       self.min_probability_spin_box)
        wd_tagger_settings_form.addRow('Maximum tags', self.max_tags_spin_box)
        wd_tagger_settings_form.addRow(tags_to_exclude_form)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        bad_forced_words_form = QFormLayout()
        bad_forced_words_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        bad_forced_words_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self.bad_words_line_edit.setClearButtonEnabled(True)
        self.forced_words_line_edit = SettingsLineEdit(key='forced_words')
        self.forced_words_line_edit.setClearButtonEnabled(True)
        bad_forced_words_form.addRow('Discourage from caption',
                                     self.bad_words_line_edit)
        bad_forced_words_form.addRow('Include in caption',
                                     self.forced_words_line_edit)
        self.min_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='min_new_tokens', default=1, minimum=1, maximum=99999)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=100, minimum=1, maximum=99999)
        self.beam_count_spin_box = FocusedScrollSettingsSpinBox(
            key='num_beams', default=1, minimum=1, maximum=99)
        self.length_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='length_penalty', default=1, minimum=-5, maximum=5)
        self.length_penalty_spin_box.setSingleStep(0.1)
        self.use_sampling_check_box = SettingsBigCheckBox(key='do_sample',
                                                          default=False)
        # The temperature must be positive.
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=50, minimum=0, maximum=200)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self.no_repeat_ngram_size_spin_box = FocusedScrollSettingsSpinBox(
            key='no_repeat_ngram_size', default=3, minimum=0, maximum=5)
        self.gpu_index_spin_box = FocusedScrollSettingsSpinBox(
            key='gpu_index', default=0, minimum=0, maximum=9)
        advanced_settings_form.addRow(bad_forced_words_form)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow('Minimum tokens',
                                      self.min_new_token_count_spin_box)
        advanced_settings_form.addRow('Maximum tokens',
                                      self.max_new_token_count_spin_box)
        advanced_settings_form.addRow('Number of beams',
                                      self.beam_count_spin_box)
        advanced_settings_form.addRow('Length penalty',
                                      self.length_penalty_spin_box)
        advanced_settings_form.addRow('Use sampling',
                                      self.use_sampling_check_box)
        advanced_settings_form.addRow('Temperature',
                                      self.temperature_spin_box)
        advanced_settings_form.addRow('Top-k', self.top_k_spin_box)
        advanced_settings_form.addRow('Top-p', self.top_p_spin_box)
        advanced_settings_form.addRow('Repetition penalty',
                                      self.repetition_penalty_spin_box)
        advanced_settings_form.addRow('No repeat n-gram size',
                                      self.no_repeat_ngram_size_spin_box)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow('GPU index', self.gpu_index_spin_box)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.addWidget(self.wd_tagger_settings_form_container)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        # Wire handlers
        self.model_combo_box.currentTextChanged.connect(
            self.update_visibility_for_model)
        self.device_combo_box.currentTextChanged.connect(
            self.set_load_in_4_bit_visibility)
        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)
        # Make sure the minimum new token count is less than or equal to the
        # maximum new token count.
        self.min_new_token_count_spin_box.valueChanged.connect(
            self.max_new_token_count_spin_box.setMinimum)
        self.max_new_token_count_spin_box.valueChanged.connect(
            self.min_new_token_count_spin_box.setMaximum)

        # Ensure the UI rows reflect the initially selected model.
        self.update_visibility_for_model(self.model_combo_box.currentText())
        if not self.is_bitsandbytes_available:
            self.load_in_4_bit_check_box.setChecked(False)

    def get_local_model_paths(self) -> list[str]:
        models_directory_path = settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if not models_directory_path:
            return []
        models_directory_path = Path(models_directory_path)
        print(f'Loading local auto-captioning model paths under '
              f'{models_directory_path}...')
        # Auto-captioning models have a `config.json` file.
        config_paths = set(models_directory_path.glob('**/config.json'))
        # WD Tagger models have a `selected_tags.csv` file.
        selected_tags_paths = set(
            models_directory_path.glob('**/selected_tags.csv'))
        model_directory_paths = [str(path.parent) for path
                                 in config_paths | selected_tags_paths]
        model_directory_paths.sort()
        print(f'Loaded {len(model_directory_paths)} model '
              f'{pluralize("path", len(model_directory_paths))}.')
        return model_directory_paths

    @Slot()
    def schedule_api_models_fetch(self):
        """Schedule a debounced fetch of the available models."""
        if not issubclass(get_model_class(self.model_combo_box.currentText()),
                          RemoteGen):
            return
        self._api_model_fetch_timer.start()

    @Slot()
    def fetch_api_models(self):
        """Fetch the available models from the configured API endpoint."""
        if not issubclass(get_model_class(self.model_combo_box.currentText()),
                          RemoteGen):
            return
        api_url = self.remote_address_line_edit.text().strip()
        api_key = self.api_key_line_edit.text().strip()
        # Avoid pointless requests for the official endpoint without a key.
        if not api_url and not api_key:
            self.set_api_status('')
            return
        if (self._api_model_fetcher is not None
                and self._api_model_fetcher.isRunning()):
            # Try again after the running request has finished.
            self._api_model_fetch_timer.start()
            return
        self.set_api_status('Loading the available models...')
        self.refresh_api_models_button.setEnabled(False)
        self._api_model_fetcher = ApiModelFetcher(api_url, api_key, self)
        self._api_model_fetcher.models_fetched.connect(self.set_api_models)
        self._api_model_fetcher.error_occurred.connect(
            self.show_api_models_error)
        self._api_model_fetcher.finished.connect(
            self._api_models_fetch_finished)
        self._api_model_fetcher.start()

    @Slot(list)
    def set_api_models(self, model_ids: list[str]):
        current_model = self.api_model_combo_box.currentText()
        self.api_model_combo_box.blockSignals(True)
        self.api_model_combo_box.clear()
        self.api_model_combo_box.addItems(model_ids)
        # The combo box is editable, so an empty field or a model name that is
        # not in the list is kept instead of selecting the first entry.
        self.api_model_combo_box.setEditText(current_model)
        self.api_model_combo_box.blockSignals(False)
        if model_ids:
            self.set_api_status(
                f'Found {len(model_ids)} '
                f'{pluralize("model", len(model_ids))}.')
        else:
            self.set_api_status(
                'The API did not report any models.', is_error=True,
                tooltip='The API did not report any models. Enter the model '
                        'name manually.')

    @Slot(str)
    def show_api_models_error(self, error_message: str):
        # Do not block usage just because the model list could not be loaded:
        # the model name can always be typed manually.
        error_message = truncate_text(elide_binary_data(error_message), 300)
        print(f'Could not load the model list: {error_message}')
        self.set_api_status(
            'Could not load the model list.', is_error=True,
            tooltip=f'Could not load the model list: {error_message}')

    def set_api_status(self, text: str, is_error: bool = False,
                       tooltip: str | None = None):
        """Show a short status message in the API settings."""
        if not text:
            self.api_status_label.clear()
            self.api_status_label.setToolTip('')
            self.api_status_label.hide()
            return
        self.api_status_label.setText(text)
        self.api_status_label.setToolTip(tooltip or text)
        if is_error:
            self.api_status_label.setStyleSheet('color: #c0392b;')
        else:
            self.api_status_label.setStyleSheet('')
        self.api_status_label.show()

    @Slot(bool)
    def set_api_key_visibility(self, is_visible: bool):
        self.api_key_line_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if is_visible
            else QLineEdit.EchoMode.Password)

    @Slot()
    def _api_models_fetch_finished(self):
        self.refresh_api_models_button.setEnabled(True)
        if self._api_model_fetcher is not None:
            self._api_model_fetcher.deleteLater()
            self._api_model_fetcher = None

    @Slot(str)
    def set_load_in_4_bit_visibility(self, device: str):
        model_class = get_model_class(self.model_combo_box.currentText())
        if issubclass(model_class, RemoteGen) or model_class is WdTagger:
            self.load_in_4_bit_container.setVisible(False)
            return
        is_load_in_4_bit_available = (self.is_bitsandbytes_available
                                      and device == CaptionDevice.GPU)
        self.load_in_4_bit_container.setVisible(is_load_in_4_bit_available)

    @Slot(str)
    def update_visibility_for_model(self, model_id: str):
        """Show only the settings that are relevant for the selected model."""
        model_class = get_model_class(model_id)
        is_api_model = issubclass(model_class, RemoteGen)
        is_wd_tagger_model = model_class is WdTagger
        is_local_model = not is_api_model and not is_wd_tagger_model

        self.wd_tagger_settings_form_container.setVisible(is_wd_tagger_model)

        # Basic captioning settings are shared by everything except WD Tagger.
        for widget in (self.prompt_label, self.prompt_text_edit,
                       self.skip_hash_container, self.caption_start_label,
                       self.caption_start_line_edit,
                       self.remove_tag_separators_container):
            widget.setVisible(not is_wd_tagger_model)

        # API settings.
        self.api_settings_container.setVisible(is_api_model)

        # Show the model default prompt as a hint for an empty prompt field.
        if is_wd_tagger_model:
            self.prompt_text_edit.setPlaceholderText('')
        else:
            # Some models define the default prompt as a class attribute,
            # others as a static method.
            default_prompt = getattr(model_class, 'default_prompt', '')
            if not default_prompt:
                try:
                    default_prompt = model_class.get_default_prompt()
                except TypeError:
                    default_prompt = ''
            default_prompt = str(default_prompt or '').strip()
            if default_prompt:
                self.prompt_text_edit.setPlaceholderText(
                    f'Empty: uses the default prompt "{default_prompt}"')
            else:
                self.prompt_text_edit.setPlaceholderText(
                    'Instructions given to the model')

        # Device settings are only relevant for local models.
        self.device_label.setVisible(is_local_model)
        self.device_combo_box.setVisible(is_local_model)
        self._set_form_row_visible(self.gpu_index_spin_box, is_local_model)
        if is_local_model:
            self.set_load_in_4_bit_visibility(
                self.device_combo_box.currentText())
        else:
            self.load_in_4_bit_container.setVisible(False)

        # Advanced settings: local models support all parameters, while API
        # models only support the token/temperature/top-p subset.
        show_advanced_settings = not is_wd_tagger_model
        self.horizontal_line.setVisible(show_advanced_settings)
        self.toggle_advanced_settings_form_button.setVisible(
            show_advanced_settings)
        if not show_advanced_settings:
            self.advanced_settings_form_container.hide()

        local_only_fields = (
            self.bad_words_line_edit, self.forced_words_line_edit,
            self.min_new_token_count_spin_box, self.beam_count_spin_box,
            self.length_penalty_spin_box, self.use_sampling_check_box,
            self.top_k_spin_box, self.repetition_penalty_spin_box,
            self.no_repeat_ngram_size_spin_box)
        for field in local_only_fields:
            self._set_form_row_visible(field, is_local_model)
        for field in (self.max_new_token_count_spin_box,
                      self.temperature_spin_box, self.top_p_spin_box):
            self._set_form_row_visible(field, show_advanced_settings)

        if is_api_model:
            self.schedule_api_models_fetch()

    def _set_form_row_visible(self, field_widget: QWidget, visible: bool):
        """Show or hide a form field together with its label."""
        for form_layout in self.findChildren(QFormLayout):
            for row in range(form_layout.rowCount()):
                field_item = form_layout.itemAt(
                    row, QFormLayout.ItemRole.FieldRole)
                if field_item is None or field_item.widget() is not field_widget:
                    continue
                label_item = form_layout.itemAt(
                    row, QFormLayout.ItemRole.LabelRole)
                if label_item is not None and label_item.widget() is not None:
                    label_item.widget().setVisible(visible)
                field_widget.setVisible(visible)
                return
        field_widget.setVisible(visible)

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        selected_model = self.model_combo_box.currentText()
        api_model = ''
        if issubclass(get_model_class(selected_model), RemoteGen):
            api_model = self.api_model_combo_box.currentText().strip()
            settings.setValue('api_model', api_model)

        return {
            'model_id': selected_model,
            'api_model': api_model,
            'api_url': self.remote_address_line_edit.text(),
            'api_key': self.api_key_line_edit.text(),
            'api_max_image_megapixels':
                self.api_max_image_megapixels_spin_box.value(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'skip_hash': self.skip_hash_check_box.isChecked(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'device': self.device_combo_box.currentText(),
            'gpu_index': self.gpu_index_spin_box.value(),
            'load_in_4_bit': self.load_in_4_bit_check_box.isChecked(),
            'limit_to_crop': self.limit_to_crop_check_box.isChecked(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'forced_words': self.forced_words_line_edit.text(),
            'generation_parameters': {
                'min_new_tokens': self.min_new_token_count_spin_box.value(),
                'max_new_tokens': self.max_new_token_count_spin_box.value(),
                'num_beams': self.beam_count_spin_box.value(),
                'length_penalty': self.length_penalty_spin_box.value(),
                'do_sample': self.use_sampling_check_box.isChecked(),
                'temperature': self.temperature_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'repetition_penalty': self.repetition_penalty_spin_box.value(),
                'no_repeat_ngram_size':
                    self.no_repeat_ngram_size_spin_box.value()
            },
            'wd_tagger_settings': {
                'show_probabilities':
                    self.show_probabilities_check_box.isChecked(),
                'min_probability': self.min_probability_spin_box.value(),
                'max_tags': self.max_tags_spin_box.value(),
                'tags_to_exclude':
                    self.tags_to_exclude_text_edit.toPlainText()
            }
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.is_captioning = False
        self.captioning_thread = None
        self.show_alert_when_finished = False
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.model_device_type: str | None = None
        self.is_model_loaded_in_4_bit = None

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = ConsoleTextEdit()
        set_text_edit_height(self.console_text_edit, 8)
        self.console_text_edit.setToolTip(
            'Captioning output. Image data is replaced by a placeholder.')
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        if (self.console_text_edit.add_output(text)
                and self.console_text_edit.isHidden()):
            self.console_text_edit.show()

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            alert = QMessageBox()
            alert.setIcon(QMessageBox.Icon.Critical)
            alert.setText('An error occurred during captioning.')
            alert.setInformativeText(
                'See the Auto-Captioner console for more information.')
            error_message = (self.captioning_thread.error_message or '').strip()
            if error_message:
                alert.setDetailedText(error_message)
        elif self.show_alert_when_finished:
            alert = QMessageBox()
            alert.setIcon(QMessageBox.Icon.Information)
            alert.setText('Captioning has finished.')
        else:
            return
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        self.show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.show_alert_when_finished = (
                confirmation_dialog.show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        models_directory_path = settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        models_directory_path = (Path(models_directory_path)
                                 if models_directory_path else None)
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator, models_directory_path)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
