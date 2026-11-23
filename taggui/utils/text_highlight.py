"""Helpers for drawing item view text with a highlighted search text.

The text is laid out the same way as `QCommonStylePrivate::viewItemDrawText`
does it (word wrap, eliding and vertical alignment) so that items look
identical to the default item delegate painting.
"""

import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QPainter, QPalette,
                           QTextLayout, QTextOption)
from PySide6.QtWidgets import QApplication, QStyle, QWidget

HIGHLIGHT_BACKGROUND_COLOR = QColor('#ffd54f')

# Qt style options use the unicode line separator instead of a newline.
LINE_SEPARATOR = '\u2028'


def iter_matched_ranges(line_text: str, search_text: str):
    """Yield the (start, end) ranges of all matches within a line."""
    if not search_text:
        return
    for match in re.finditer(re.escape(search_text), line_text,
                             re.IGNORECASE):
        yield match.start(), match.end()


def get_style_and_widget(option) -> tuple[QStyle, QWidget | None]:
    """Get the style and widget that are used to draw an item view item."""
    widget = option.styleObject
    if not isinstance(widget, QWidget):
        widget = None
    style = widget.style() if widget is not None else QApplication.style()
    return style, widget


def get_text_color(option) -> QColor:
    """Get the color that the default delegate uses for the item text."""
    if not (option.state & QStyle.StateFlag.State_Enabled):
        color_group = QPalette.ColorGroup.Disabled
    elif not (option.state & QStyle.StateFlag.State_Active):
        color_group = QPalette.ColorGroup.Inactive
    else:
        color_group = QPalette.ColorGroup.Normal
    if option.state & QStyle.StateFlag.State_Selected:
        color_role = QPalette.ColorRole.HighlightedText
    else:
        color_role = QPalette.ColorRole.Text
    return option.palette.color(color_group, color_role)


def _lay_out_text(text: str, font: QFont, text_option: QTextOption,
                  line_width: float, max_height: float | None = None
                  ) -> tuple[QTextLayout, int]:
    """
    Lay out the lines of the text like
    `QCommonStylePrivate::viewItemTextLayout`. Return the layout and the index
    of the last line that is (partially) visible, or -1.
    """
    layout = QTextLayout(text, font)
    layout.setTextOption(text_option)
    layout.beginLayout()
    height = 0.0
    last_visible_line = -1
    line_index = 0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(line_width)
        line.setPosition(QPointF(0, height))
        height += line.height()
        # Assume that the height of the next line is the same as the height of
        # the current line.
        if max_height is not None and height + line.height() > max_height:
            next_line = layout.createLine()
            last_visible_line = line_index if next_line.isValid() else -1
            break
        line_index += 1
    layout.endLayout()
    return layout, last_visible_line


def _elide_text(text: str, font: QFont, text_option: QTextOption,
                text_rect, valign: Qt.AlignmentFlag,
                elide_mode: Qt.TextElideMode) -> tuple[str, QPointF]:
    """
    Elide the text like `QCommonStylePrivate::calculateElidedText`. Return the
    elided text and the position at which it should be painted.
    """
    font_metrics = QFontMetricsF(font)
    # When more than one line is displayed and the height only allows some of
    # the lines, it makes more sense to show the start of the text.
    vertical_alignment_optimization = bool(valign
                                           & Qt.AlignmentFlag.AlignVCenter)
    max_height = (text_rect.height() if vertical_alignment_optimization
                  else None)
    layout, last_visible_line = _lay_out_text(
        text, font, text_option, text_rect.width(), max_height)
    bounding_rect = layout.boundingRect()
    layout_rect = QStyle.alignedRect(Qt.LayoutDirection.LayoutDirectionAuto,
                                     valign, bounding_rect.size().toSize(),
                                     text_rect)
    paint_position = QPointF(text_rect.x(), layout_rect.top())
    height = 0.0
    elided_text = ''
    line_count = layout.lineCount()
    for line_index in range(line_count):
        line = layout.lineAt(line_index)
        height += line.height()
        # Skip lines that are above the visible rect.
        if height + layout_rect.top() <= text_rect.top():
            paint_position.setY(paint_position.y() + line.height())
            continue
        start = line.textStart()
        length = line.textLength()
        line_text = text[start:start + length]
        draw_elided = line.naturalTextWidth() > text_rect.width()
        elide_last_visible_line = last_visible_line == line_index
        if (not draw_elided and line_index + 1 < line_count
                and last_visible_line >= 0):
            next_line = layout.lineAt(line_index + 1)
            next_height = height + next_line.height() / 2
            # Elide when less than the next half line is visible.
            if (next_height + layout_rect.top()
                    > text_rect.height() + text_rect.top()):
                elide_last_visible_line = True
        if draw_elided or elide_last_visible_line:
            if elide_last_visible_line:
                if line_text.endswith(LINE_SEPARATOR):
                    line_text = line_text[:-1]
                line_text += '\u2026'
            line_text = font_metrics.elidedText(line_text, elide_mode,
                                                text_rect.width())
        elided_text += line_text
        if (line_index < line_count - 1
                and not elided_text.endswith(LINE_SEPARATOR)):
            elided_text += LINE_SEPARATOR
        # Stop when the text is below the visible text or when the last
        # visible line is reached.
        if (height + layout_rect.top() >= text_rect.bottom()
                or (last_visible_line >= 0
                    and last_visible_line == line_index)):
            break
    return elided_text, paint_position


def draw_highlighted_text(painter: QPainter, rect, text: str,
                          search_text: str, font: QFont, text_color: QColor,
                          text_margin: int,
                          alignment: Qt.AlignmentFlag =
                          Qt.AlignmentFlag.AlignLeft
                          | Qt.AlignmentFlag.AlignVCenter,
                          elide_mode: Qt.TextElideMode =
                          Qt.TextElideMode.ElideRight,
                          first_highlight_line: int = 0) -> None:
    """
    Draw the item text into `rect` with all occurrences of the search text
    highlighted. Matches before the line with the index `first_highlight_line`
    are not highlighted, e.g. because they are part of the file name and not
    of a tag.
    """
    text = text.replace('\n', LINE_SEPARATOR)
    # Remove the width padding like the default delegate does.
    text_rect = rect.adjusted(text_margin, 0, -text_margin, 0)
    if not text or text_rect.width() <= 0 or text_rect.height() <= 0:
        return
    text_option = QTextOption()
    text_option.setWrapMode(QTextOption.WrapMode.WordWrap)
    horizontal_alignment = (alignment
                            & Qt.AlignmentFlag.AlignHorizontal_Mask)
    if not horizontal_alignment:
        horizontal_alignment = Qt.AlignmentFlag.AlignLeft
    text_option.setAlignment(horizontal_alignment)
    elided_text, paint_position = _elide_text(
        text, font, text_option, text_rect, alignment, elide_mode)
    # Lay out the elided text so that the positions of the matches are known.
    layout, _ = _lay_out_text(elided_text, font, text_option,
                              text_rect.width())
    painter.save()
    painter.setClipRect(rect)
    if search_text:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(HIGHLIGHT_BACKGROUND_COLOR)
        for line_index in range(layout.lineCount()):
            if line_index < first_highlight_line:
                continue
            line = layout.lineAt(line_index)
            line_start = line.textStart()
            line_text = elided_text[line_start:line_start + line.textLength()]
            for match_start, match_end in iter_matched_ranges(line_text,
                                                              search_text):
                start_x = line.cursorToX(line_start + match_start)[0]
                end_x = line.cursorToX(line_start + match_end)[0]
                painter.drawRect(
                    QRectF(paint_position.x() + line.x() + start_x,
                           paint_position.y() + line.y(),
                           max(end_x - start_x, 1.0),
                           line.height()))
    painter.setPen(text_color)
    layout.draw(painter, paint_position)
    painter.restore()
