from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtWidgets import (QFrame, QPlainTextEdit, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem)

from utils.text_highlight import (draw_highlighted_text,
                                  get_style_and_widget, get_text_color)


class TextEditItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        # The text that is highlighted in the displayed tags, if any.
        self.highlight_text = None

    def paint(self, painter, option, index):
        # Add some left padding.
        option.rect.adjust(4, 0, 0, 0)
        if not self.highlight_text:
            super().paint(painter, option, index)
            return
        # Draw the background and the selection without the text, then draw
        # the text separately so that the search text can be highlighted.
        style, widget = get_style_and_widget(option)
        highlighted_option = QStyleOptionViewItem(option)
        self.initStyleOption(highlighted_option, index)
        text = highlighted_option.text
        highlighted_option.text = ''
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem,
                          highlighted_option, painter, widget)
        if not text:
            return
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, highlighted_option, widget)
        text_margin = (style.pixelMetric(
            QStyle.PixelMetric.PM_FocusFrameHMargin, highlighted_option,
            widget) + 1)
        draw_highlighted_text(
            painter, text_rect, text, self.highlight_text,
            highlighted_option.font, get_text_color(highlighted_option),
            text_margin, alignment=highlighted_option.displayAlignment,
            elide_mode=highlighted_option.textElideMode)

    def createEditor(self, parent, option, index):
        editor = QPlainTextEdit(parent)
        editor.setFrameStyle(QFrame.Shape.NoFrame)
        editor.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor.setStyleSheet('padding-left: 3px;')
        editor.index = index
        return editor

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(size.height() + 8)
        return size

    def eventFilter(self, editor, event: QEvent):
        if (event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)
            self.parent().setCurrentIndex(
                self.parent().model().index(editor.index.row(), 0))
            self.parent().selectionModel().select(
                self.parent().model().index(editor.index.row(), 0),
                QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.parent().setFocus()
            return True
        # This is required to prevent crashing when the user clicks on another
        # tag in the All Tags list.
        if event.type() == QEvent.FocusOut:
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)
            return True
        return False
