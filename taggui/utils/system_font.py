"""Determine the default UI font of the desktop.

Qt only applies the KDE font settings through the KDE platform theme plugin.
Applications that ship their own Qt (like the PySide6 wheels) do not include
that plugin, so Qt falls back to its generic 9 pt font on KDE, which is
smaller than the font that other applications use. Read the font from the
desktop's own configuration in that case.
"""

import os
import sys

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from utils.settings import (DEFAULT_SETTINGS, migrate_font_size_setting,
                            settings)

# The family of Qt's generic font on Unix. It is used to detect whether Qt
# applied a font of its own.
_GENERIC_FONT_FAMILY = 'Sans Serif'


def configure_font(app: QApplication):
    """
    Set the application font to the desktop font, scaled by the configured
    percentage.

    This must be called before any widget is created, because widgets inherit
    the font of their parent and `QApplication.setFont()` does not update
    widgets that already exist.
    """
    system_font = get_desktop_font(app.font()) or app.font()
    migrate_font_size_setting(system_font.pointSizeF())
    scale_percent = settings.value(
        'font_scale_percent',
        defaultValue=DEFAULT_SETTINGS['font_scale_percent'], type=int)
    font = QFont(system_font)
    if scale_percent != 100:
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * scale_percent / 100)
        else:
            font.setPixelSize(max(1, round(font.pixelSize() * scale_percent
                                           / 100)))
    # Point sizes are converted to pixels by Qt based on the pixel density of
    # the display, so high DPI displays work automatically.
    app.setFont(font)


def get_desktop_font(current_font: QFont) -> QFont | None:
    """
    Get the default UI font of the desktop.

    `current_font` is the font that Qt uses so far. `None` is returned if the
    font is unknown or if Qt is not on KDE, in which case `current_font` should
    be kept.
    """
    if not sys.platform.startswith('linux'):
        return None
    if 'kde' not in os.environ.get('XDG_CURRENT_DESKTOP', '').lower():
        return None
    font = _get_kde_font()
    if font is None and current_font.family() == _GENERIC_FONT_FAMILY:
        # KDE keeps the GTK font settings in sync with the desktop font, so
        # they are a fallback for when the KDE configuration does not contain
        # the font. They are only used if Qt did not apply a font of its own,
        # because they can be stale if the GTK configuration was not updated.
        font = _get_gtk_font()
    return font


def _get_kde_font() -> QFont | None:
    description = None
    for path in _config_paths('kdeglobals'):
        # KDE stores the font in the General group, but newer versions might
        # use the Fonts group.
        for group, key in (('General', 'font'), ('Fonts', 'General')):
            value = _read_ini_value(path, group, key)
            if value:
                description = value
    return _parse_font_description(description)


def _get_gtk_font() -> QFont | None:
    description = None
    for version in ('gtk-3.0', 'gtk-4.0'):
        for path in _config_paths(os.path.join(version, 'settings.ini')):
            value = _read_ini_value(path, 'Settings', 'gtk-font-name')
            if value:
                description = value
    return _parse_font_description(description)


def _config_paths(file_name: str) -> list[str]:
    """Get the configuration file paths ordered from lowest to highest
    priority."""
    config_home = (os.environ.get('XDG_CONFIG_HOME')
                   or os.path.expanduser('~/.config'))
    paths = [os.path.join(config_home, 'kdedefaults', file_name)]
    for config_dir in os.environ.get('XDG_CONFIG_DIRS', '/etc/xdg').split(':'):
        if config_dir:
            paths.append(os.path.join(config_dir, file_name))
    paths.append(os.path.join(config_home, file_name))
    return paths


def _read_ini_value(path: str, group: str, key: str) -> str | None:
    if not os.path.isfile(path):
        return None
    config = QSettings(path, QSettings.Format.IniFormat)
    config.beginGroup(group)
    value = config.value(key)
    config.endGroup()
    if isinstance(value, (list, tuple)):
        # INI values that contain commas are read as a list.
        value = ','.join(str(item) for item in value)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _parse_font_description(description: str | None) -> QFont | None:
    """Parse a KDE/Qt or a GTK/Pango font description."""
    if not description:
        return None
    description = description.strip()
    if ',' in description:
        # Qt/KDE use "family,pointSize,pixelSize,...", GTK sometimes uses
        # "family, size".
        fields = [field.strip() for field in description.split(',')]
        family = fields[0]
        numbers = fields[1:]
    else:
        # GTK/Pango style without a comma: "family [style] size".
        tokens = description.split()
        numbers = []
        while tokens and _is_number(tokens[-1]):
            numbers.insert(0, tokens.pop())
        family = ' '.join(tokens)
    if not family:
        return None
    point_size = pixel_size = None
    if len(numbers) >= 2:
        point_size = _to_int(numbers[0])
        pixel_size = _to_int(numbers[1])
    elif numbers:
        point_size = _to_int(numbers[-1])
    font = QFont(family)
    if point_size and point_size > 0:
        font.setPointSize(point_size)
    elif pixel_size and pixel_size > 0:
        font.setPixelSize(pixel_size)
    else:
        return None
    return font


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None
