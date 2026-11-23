from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QFontMetrics, QTextCursor, QTextOption
from PySide6.QtWidgets import QCheckBox, QPlainTextEdit, QPushButton

from utils.utils import elide_binary_data


class BigPushButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        new_size = self.sizeHint() * 1.5
        self.setFixedSize(new_size)


class TallPushButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        new_height = int(self.sizeHint().height() * 1.5)
        self.setFixedHeight(new_height)


class BigCheckBox(QCheckBox):
    def __init__(self, text: str | None = None):
        super().__init__(text)
        # Size the indicator relative to the font so that it follows the system
        # font size and the display's pixel density.
        indicator_size = round(QFontMetrics(self.font()).height() * 1.1)
        self.setStyleSheet(
            f'QCheckBox::indicator '
            f'{{ width: {indicator_size}px; height: {indicator_size}px; }}')


class ConsoleTextEdit(QPlainTextEdit):
    """
    A read-only output console for long-running threads.

    Long lines are wrapped instead of overflowing horizontally, embedded image
    data is replaced by a placeholder, and separate output statements are
    visually separated by a blank line.
    """

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Keep the memory usage bounded during long captioning runs.
        self.setMaximumBlockCount(2000)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._pending_text = ''
        self._has_output = False
        self._replace_last_statement = False
        self._last_statement_line_count = 0

    def clear(self):
        super().clear()
        self._pending_text = ''
        self._has_output = False
        self._replace_last_statement = False
        self._last_statement_line_count = 0

    def add_output(self, text: str) -> bool:
        """
        Add a chunk of text that was written by `print`.

        Complete statements (i.e. text that ended with a newline) are appended
        as separate blocks. Returns `True` if the output changed.
        """
        if text == '\x1b[A':
            # ANSI escape sequence for moving the cursor up, which means that
            # the last statement should be replaced by the next one.
            self._replace_last_statement = True
            return False
        self._pending_text += text
        # `print` sends the text and the terminating newline separately, so
        # wait until the statement is complete.
        if not self._pending_text.endswith('\n'):
            return False
        statement = self._pending_text.rstrip('\n')
        self._pending_text = ''
        return self.append_statement(statement)

    def append_statement(self, statement: str) -> bool:
        lines = [line.rstrip() for line in statement.splitlines()]
        lines = [elide_binary_data(line) for line in lines if line.strip()]
        if not lines:
            return False
        if self._replace_last_statement:
            self._replace_last_statement = False
            self.remove_last_statement()
        if self._has_output and not self.document().isEmpty():
            self.appendPlainText('')
        for line in lines:
            self.appendPlainText(line)
        self._last_statement_line_count = len(lines)
        self._has_output = True
        # Keep the newest output visible.
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        return True

    def remove_last_statement(self):
        """Remove the blocks of the last appended statement."""
        document = self.document()
        if self._last_statement_line_count <= 0 or document.isEmpty():
            return
        block_count = min(self._last_statement_line_count,
                          document.blockCount())
        start_block_number = document.blockCount() - block_count
        # Also remove the empty separator block that precedes the statement.
        if start_block_number > 0:
            previous_block = document.findBlockByNumber(start_block_number - 1)
            if not previous_block.text():
                start_block_number -= 1
        start_block = document.findBlockByNumber(start_block_number)
        cursor = QTextCursor(document)
        cursor.setPosition(start_block.position())
        cursor.movePosition(QTextCursor.MoveOperation.End,
                            QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        if start_block_number > 0:
            # Remove the newline that separated the removed text.
            cursor.deletePreviousChar()
        self._last_statement_line_count = 0
        self._has_output = bool(self.toPlainText().strip())
