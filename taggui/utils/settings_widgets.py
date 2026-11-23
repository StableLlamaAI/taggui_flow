from math import ceil

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics, QTextDocument
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QLineEdit,
                               QPlainTextEdit, QSizePolicy, QSpinBox)

from utils.big_widgets import BigCheckBox
from utils.focused_scroll_mixin import FocusedScrollMixin
from utils.settings import DEFAULT_SETTINGS, settings


class AutoResizingPlainTextEdit(QPlainTextEdit):
    """
    A plain text edit that grows with its content instead of clipping it.

    The height is adjusted to the wrapped content between a minimum and a
    maximum number of lines. Beyond the maximum the text edit scrolls.
    """

    def __init__(self, minimum_line_count: int = 3,
                 maximum_line_count: int = 12):
        super().__init__()
        self.minimum_line_count = max(1, minimum_line_count)
        self.maximum_line_count = max(self.minimum_line_count,
                                      maximum_line_count)
        self._fixed_height = 0
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.document().contentsChanged.connect(self.update_height)
        self.document().documentLayout().documentSizeChanged.connect(
            self.update_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_height()

    def get_line_height(self) -> int:
        return QFontMetrics(self.font()).lineSpacing()

    def get_extra_height(self) -> int:
        margins = self.contentsMargins()
        return (margins.top() + margins.bottom() + self.frameWidth() * 2
                + 2 * self.document().documentMargin() + 2)

    def get_content_height(self) -> float:
        """
        Get the height of the wrapped content.

        A copy of the document is used so that the calculation does not depend
        on the internal layout state of the text edit.
        """
        document = QTextDocument()
        document.setDefaultFont(self.font())
        document.setDocumentMargin(self.document().documentMargin())
        document.setPlainText(self.toPlainText())
        document.setTextWidth(max(1, self.viewport().width()))
        height = document.size().height()
        document.deleteLater()
        return height

    def update_height(self):
        """Resize the text edit so that its content is visible."""
        line_height = self.get_line_height()
        if line_height <= 0 or self.viewport().width() <= 0:
            return
        content_line_count = ceil(self.get_content_height() / line_height)
        content_line_count = max(self.minimum_line_count,
                                 min(self.maximum_line_count,
                                     content_line_count))
        height = (content_line_count * line_height + self.get_extra_height())
        if abs(height - self._fixed_height) < 2:
            return
        self._fixed_height = height
        self.setFixedHeight(height)

    def sizeHint(self):
        hint = super().sizeHint()
        if self._fixed_height:
            hint.setHeight(self._fixed_height)
        return hint


class SettingsBigCheckBox(BigCheckBox):
    def __init__(self, key: str, default: bool | None = None, text: str | None = None):
        super().__init__(text)
        if not settings.contains(key):
            settings.setValue(key, default or DEFAULT_SETTINGS.get(key))
        self.setChecked(settings.value(key, type=bool))
        self.stateChanged.connect(
            lambda state: settings.setValue(
                key, state == Qt.CheckState.Checked.value))


class SettingsComboBox(QComboBox):
    def __init__(self, key: str, default: str | None = None):
        super().__init__()
        self.key = key
        if not settings.contains(key):
            settings.setValue(key, default or DEFAULT_SETTINGS.get(key))

    def addItems(self, texts: list[str]):
        setting: str = settings.value(self.key, type=str)
        super().addItems(texts)
        self.currentTextChanged.connect(
            lambda text: settings.setValue(self.key, text))
        if setting:
            self.setCurrentText(setting)


class FocusedScrollSettingsComboBox(FocusedScrollMixin, SettingsComboBox):
    pass


class FocusedScrollSettingsDoubleSpinBox(FocusedScrollMixin, QDoubleSpinBox):
    def __init__(self, key: str, default: float, minimum: float,
                 maximum: float):
        super().__init__()
        # The range must be set here so that the setting value is not clamped
        # by the default range.
        self.setRange(minimum, maximum)
        self.setValue(settings.value(key, default, type=float))
        self.valueChanged.connect(lambda value: settings.setValue(key, value))


class SettingsSpinBox(QSpinBox):
    def __init__(self, key: str, minimum: int, maximum: int, default: int | None = None):
        super().__init__()
        self.setRange(minimum, maximum)
        if not settings.contains(key):
            settings.setValue(key, default or DEFAULT_SETTINGS.get(key))
        self.setValue(settings.value(key, type=int))
        self.valueChanged.connect(lambda value: settings.setValue(key, value))


class FocusedScrollSettingsSpinBox(FocusedScrollMixin, SettingsSpinBox):
    pass


class SettingsLineEdit(QLineEdit):
    def __init__(self, key: str, default: str | None = None):
        super().__init__()
        if not settings.contains(key):
            settings.setValue(key, default or DEFAULT_SETTINGS.get(key, ''))
        self.setText(settings.value(key, type=str))
        self.textChanged.connect(lambda text: settings.setValue(key, text))


class SettingsPlainTextEdit(AutoResizingPlainTextEdit):
    def __init__(self, key: str, default: str | None = None,
                 minimum_line_count: int = 3,
                 maximum_line_count: int = 12):
        super().__init__(minimum_line_count, maximum_line_count)
        if not settings.contains(key):
            settings.setValue(key, default or DEFAULT_SETTINGS.get(key, ''))
        self.setPlainText(settings.value(key, type=str))
        self.textChanged.connect(lambda: settings.setValue(key,
                                                           self.toPlainText()))
