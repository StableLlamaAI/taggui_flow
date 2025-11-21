import sys
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFontMetrics, QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QDockWidget, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.models.wd_tagger import WdTagger
from auto_captioning.models.remote import RemoteGen
from auto_captioning.models_list import MODELS, get_model_class
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionDevice, CaptionPosition
from utils.settings import DEFAULT_SETTINGS, settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
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
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(MODELS)

        #remote only:
        self.remote_address_line_edit = SettingsLineEdit(key='remote_address', default='http://localhost:5000')
        self.api_key_line_edit = SettingsLineEdit(key='api_key', default='')

        # Create OpenAI model selector (will be hidden by default)
        from PySide6.QtWidgets import QComboBox
        self.openai_model_combo_box = QComboBox()
        self.openai_model_combo_box.addItem('Loading models...')
        self.openai_model_combo_box.setEnabled(False)
        self.openai_model_combo_box.setVisible(False)  # Hidden by default

        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
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
        basic_settings_form.addRow('OAI Compatible Endpoint', self.remote_address_line_edit)
        basic_settings_form.addRow('API Key', self.api_key_line_edit)
        basic_settings_form.addRow('OpenAI Model', self.openai_model_combo_box)

        # Connect signals to fetch models when API settings change
        self.remote_address_line_edit.textChanged.connect(self.fetch_openai_models)
        self.api_key_line_edit.textChanged.connect(self.fetch_openai_models)

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
            key='wd_tagger_tags_to_exclude')
        tags_to_exclude_form.addRow('Tags to exclude',
                                    self.tags_to_exclude_text_edit)
        set_text_edit_height(self.tags_to_exclude_text_edit, 4)
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
            key='min_new_tokens', default=1, minimum=1, maximum=999)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=100, minimum=1, maximum=999)
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
            self.show_settings_for_model)
        # Update visibility dynamically whenever model changes
        self.model_combo_box.currentTextChanged.connect(self.update_visibility_for_model)
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

        self.show_settings_for_model(self.model_combo_box.currentText())
        # Ensure the UI rows reflect the initially selected model
        self.update_visibility_for_model(self.model_combo_box.currentText())
        self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())
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

    @Slot(str)
    def fetch_openai_models(self):
        """Fetch available models from OpenAI API endpoint."""
        from auto_captioning.models_list import get_model_class
        from auto_captioning.models.openai_model import OpenAIModel

        # Only fetch if OpenAI is selected
        current_model = self.model_combo_box.currentText()
        if get_model_class(current_model) != OpenAIModel:
            return

        api_url = self.remote_address_line_edit.text().strip()
        api_key = self.api_key_line_edit.text().strip()

        # Don't fetch if no API key
        if not api_key:
            self.openai_model_combo_box.clear()
            self.openai_model_combo_box.addItem('Enter API key to load models')
            self.openai_model_combo_box.setEnabled(False)
            return

        # Show loading state
        self.openai_model_combo_box.clear()
        self.openai_model_combo_box.addItem('Loading models...')
        self.openai_model_combo_box.setEnabled(False)

        # Fetch models in a separate thread to avoid blocking UI
        from PySide6.QtCore import QThread, Signal

        class ModelFetcher(QThread):
            models_fetched = Signal(list)
            error_occurred = Signal(str)

            def __init__(self, api_url, api_key):
                super().__init__()
                self.api_url = api_url
                self.api_key = api_key

            def run(self):
                try:
                    from openai import OpenAI

                    # Initialize client
                    if self.api_url:
                        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
                    else:
                        client = OpenAI(api_key=self.api_key)

                    # Fetch models
                    models = client.models.list()

                    # Filter for vision-capable models
                    vision_models = []
                    for model in models.data:
                        model_id = model.id
                        # Common vision-capable models
                        if any(keyword in model_id.lower() for keyword in
                               ['gpt-4', 'vision', 'gpt-4o', 'gpt-4-turbo']):
                            vision_models.append(model_id)

                    # If no vision models found, include all models
                    if not vision_models:
                        vision_models = [model.id for model in models.data]

                    # Sort models
                    vision_models.sort()

                    if vision_models:
                        self.models_fetched.emit(vision_models)
                    else:
                        self.error_occurred.emit('No models available')

                except Exception as e:
                    self.error_occurred.emit(str(e))

        def on_models_fetched(models):
            self.openai_model_combo_box.clear()

            # Restore previously selected model if it exists
            previous_model = settings.value('openai_selected_model', defaultValue='')

            for model in models:
                self.openai_model_combo_box.addItem(model)

            # Try to restore previous selection
            if previous_model:
                index = self.openai_model_combo_box.findText(previous_model)
                if index >= 0:
                    self.openai_model_combo_box.setCurrentIndex(index)

            self.openai_model_combo_box.setEnabled(True)

            # Clean up thread
            if hasattr(self, '_model_fetcher'):
                self._model_fetcher.deleteLater()

        def on_error(error_msg):
            self.openai_model_combo_box.clear()
            self.openai_model_combo_box.addItem(f'Error: {error_msg}')
            self.openai_model_combo_box.setEnabled(False)

            # Clean up thread
            if hasattr(self, '_model_fetcher'):
                self._model_fetcher.deleteLater()

        # Create and start fetcher thread
        self._model_fetcher = ModelFetcher(api_url, api_key)
        self._model_fetcher.models_fetched.connect(on_models_fetched)
        self._model_fetcher.error_occurred.connect(on_error)
        self._model_fetcher.start()

    def show_settings_for_model(self, model_id: str):
        from auto_captioning.models.openai_model import OpenAIModel

        wd_tagger_widgets = [self.wd_tagger_settings_form_container]
        device_related_widgets = [
            self.device_label,
            self.device_combo_box,
            self.load_in_4_bit_container
        ]
        advanced_settings_widgets = [
            self.horizontal_line,
            self.toggle_advanced_settings_form_button,
            self.advanced_settings_form_container
        ]
        basic_captioning_widgets = [
            self.prompt_label,
            self.prompt_text_edit,
            self.skip_hash_container,
            self.caption_start_label,
            self.caption_start_line_edit,
            self.remove_tag_separators_container
        ]

        model_class = get_model_class(model_id)
        is_wd_tagger_model = model_class == WdTagger
        is_api_model = model_class in [RemoteGen, OpenAIModel]

        # Show/hide WD Tagger specific widgets
        for widget in wd_tagger_widgets:
            widget.setVisible(is_wd_tagger_model)

        # Show/hide basic captioning widgets (for all non-WD-Tagger models)
        for widget in basic_captioning_widgets:
            widget.setVisible(not is_wd_tagger_model)

        # Show/hide device-related widgets (only for local models)
        for widget in device_related_widgets:
            widget.setVisible(not is_wd_tagger_model and not is_api_model)

        # Show/hide advanced settings (only for local models, not for API models)
        for widget in advanced_settings_widgets:
            widget.setVisible(not is_wd_tagger_model and not is_api_model)

        # Show/hide API-related widgets
        is_openai_model = model_class == OpenAIModel
        self.remote_address_line_edit.setVisible(is_api_model)
        self.api_key_line_edit.setVisible(is_api_model)
        self.openai_model_combo_box.setVisible(is_openai_model)

        # Fetch OpenAI models when OpenAI is selected
        if is_openai_model:
            self.fetch_openai_models()

        # Set load in 4-bit visibility only for local models
        if not is_wd_tagger_model and not is_api_model:
            self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())

    @Slot(str)
    def set_load_in_4_bit_visibility(self, device: str):
        model_id = self.model_combo_box.currentText()
        is_wd_tagger_model = get_model_class(model_id) == WdTagger
        if is_wd_tagger_model:
            self.load_in_4_bit_container.setVisible(False)
            return
        is_load_in_4_bit_available = (self.is_bitsandbytes_available
                                      and device == CaptionDevice.GPU)
        self.load_in_4_bit_container.setVisible(is_load_in_4_bit_available)

    @Slot(str)
    def update_visibility_for_model(self, model_id: str):
        # Determine model type
        model_class = get_model_class(model_id)
        try:
            from auto_captioning.models.remote import RemoteGen
        except Exception:
            RemoteGen = None
        try:
            from auto_captioning.models.openai_model import OpenAIModel
        except Exception:
            OpenAIModel = None

        is_api_model = (RemoteGen is not None and model_class == RemoteGen) or \
                       (OpenAIModel is not None and model_class == OpenAIModel)
        is_wd_tagger_model = model_class == WdTagger

        # Helper to hide both label and field by scanning all QFormLayouts
        from PySide6.QtWidgets import QFormLayout
        def _set_form_row_visible_for_field(field_widget, visible: bool):
            label_widget = None
            try:
                form_layouts = self.findChildren(QFormLayout)
                for form in form_layouts:
                    try:
                        rows = form.rowCount()
                    except Exception:
                        continue
                    for row in range(rows):
                        item_field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                        field = item_field.widget() if item_field is not None else None
                        if field is field_widget:
                            item_label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                            label_widget = item_label.widget() if item_label is not None else None
                            break
                    if label_widget is not None:
                        break
            except Exception:
                pass
            if label_widget is not None:
                label_widget.setVisible(visible)
            field_widget.setVisible(visible)

        # Toggle API-related rows (hide entire lines for local models)
        _set_form_row_visible_for_field(self.remote_address_line_edit, is_api_model)
        _set_form_row_visible_for_field(self.api_key_line_edit, is_api_model)
        _set_form_row_visible_for_field(self.openai_model_combo_box, is_api_model)

        # Toggle WD Tagger specific container
        self.wd_tagger_settings_form_container.setVisible(is_wd_tagger_model)

        # Toggle device/GPU rows (not relevant for API)
        self.device_label.setVisible(not is_api_model)
        self.device_combo_box.setVisible(not is_api_model)
        _set_form_row_visible_for_field(self.gpu_index_spin_box, not is_api_model)

        # Toggle 4-bit option:
        # - hidden for API
        # - hidden for WD tagger
        # - otherwise depends on device and bitsandbytes availability
        if is_api_model or is_wd_tagger_model:
            self.load_in_4_bit_container.setVisible(False)
        else:
            self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())

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
        from auto_captioning.models.openai_model import OpenAIModel

        selected_model = self.model_combo_box.currentText()

        # If OpenAI is selected, use the specific model from the combo box
        if get_model_class(selected_model) == OpenAIModel:
            if self.openai_model_combo_box.isEnabled() and self.openai_model_combo_box.count() > 0:
                openai_model = self.openai_model_combo_box.currentText()
                # Save the selected model for next time
                settings.setValue('openai_selected_model', openai_model)
                selected_model = openai_model

        return {
            'model_id': selected_model,
            'api_url': self.remote_address_line_edit.text(),
            'api_key': self.api_key_line_edit.text(),
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
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.model_device_type: str | None = None
        self.is_model_loaded_in_4_bit = None
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
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
            # Check if OpenAI model is selected and available
            from auto_captioning.models_list import get_model_class
            from auto_captioning.models.openai_model import OpenAIModel

            selected_model = self.caption_settings_form.model_combo_box.currentText()
            if get_model_class(selected_model) == OpenAIModel:
                if not self.caption_settings_form.openai_model_combo_box.isEnabled():
                    QMessageBox.warning(
                        self, 'OpenAI Model Not Available',
                        'Please wait for models to load or check your API key and endpoint.')
                    return

            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
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
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
