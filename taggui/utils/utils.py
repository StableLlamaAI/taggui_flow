import re
import sys
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

# Image data that some APIs echo back in their responses (or that ends up in
# error messages) is huge and useless inside the GUI, so it is replaced by a
# short placeholder before any text is displayed.
_DATA_URI_PATTERN = re.compile(
    r'data:[^\s,;]{0,100};base64,[A-Za-z0-9+/=\s]{100,}')
_BASE64_PATTERN = re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{200,}={0,2}')


def elide_binary_data(text: str, replacement: str = '[image data]') -> str:
    """Replace base64-encoded image data in a text with a short placeholder."""
    text = _DATA_URI_PATTERN.sub(replacement, text)
    return _BASE64_PATTERN.sub(replacement, text)


def truncate_text(text: str, maximum_length: int,
                  ellipsis: str = '...') -> str:
    """Shorten a text to at most `maximum_length` characters."""
    if len(text) <= maximum_length:
        return text
    return text[:maximum_length - len(ellipsis)].rstrip() + ellipsis


def get_resource_path(unbundled_resource_path: Path) -> Path:
    """
    Get the path to a resource, ensuring that it is valid even when the program
    is bundled with PyInstaller.
    """
    # PyInstaller stores the path to its temporary directory in `sys._MEIPASS`.
    base_path = getattr(sys, '_MEIPASS', Path(__file__).parent.parent.parent)
    resource_path = (Path(base_path) / unbundled_resource_path).resolve()
    return resource_path


def pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return f'{word}s'


def list_with_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


class ConfirmationDialog(QMessageBox):
    def __init__(self, title: str, question: str):
        super().__init__()
        self.setWindowTitle(title)
        self.setIcon(QMessageBox.Icon.Question)
        self.setText(question)
        self.setStandardButtons(QMessageBox.StandardButton.Yes
                                | QMessageBox.StandardButton.Cancel)
        self.setDefaultButton(QMessageBox.StandardButton.Yes)


def get_confirmation_dialog_reply(title: str, question: str) -> int:
    """Display a confirmation dialog and return the user's reply."""
    confirmation_dialog = ConfirmationDialog(title, question)
    return confirmation_dialog.exec()
