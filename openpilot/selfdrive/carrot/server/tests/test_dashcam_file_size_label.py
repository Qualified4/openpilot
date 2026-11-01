from openpilot.selfdrive.carrot.server.features.dashcam.paths import file_size_label


def test_sizes_below_one_gigabyte_keep_one_decimal():
  assert file_size_label(512) == "512 B"
  assert file_size_label(1536) == "1.5 KB"
  assert file_size_label(int(12.34 * 1024 * 1024)) == "12.3 MB"


def test_gigabyte_sizes_show_three_decimals():
  assert file_size_label(1024 ** 3) == "1.000 GB"
  assert file_size_label(int(1.2345 * 1024 ** 3)) == "1.234 GB"
