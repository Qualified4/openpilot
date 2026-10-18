"""Serve a review folder on localhost with HTTP Range support (video seeking needs it).

Opening REVIEW.html directly as a file also works in Chrome; this is for a served copy.
Binds to 127.0.0.1 only.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import re


class RangeHandler(SimpleHTTPRequestHandler):
  def send_head(self):
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
    path = self.translate_path(self.path)
    if not match or not os.path.isfile(path):
      return super().send_head()
    size = os.path.getsize(path)
    start = int(match[1]) if match[1] else max(0, size - int(match[2] or 0))
    end = min(int(match[2]), size - 1) if match[1] and match[2] else size - 1
    if start >= size or start > end:
      self.send_error(416)
      return None
    f = open(path, 'rb')  # closed by SimpleHTTPRequestHandler after copyfile
    f.seek(start)
    self.send_response(206)
    self.send_header('Content-Type', self.guess_type(path))
    self.send_header('Accept-Ranges', 'bytes')
    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
    self.send_header('Content-Length', str(end - start + 1))
    self.end_headers()
    self.range_left = end - start + 1
    return f

  def copyfile(self, source, outputfile):
    left = getattr(self, 'range_left', None)
    if left is None:
      return super().copyfile(source, outputfile)
    while left > 0:
      chunk = source.read(min(1 << 16, left))
      if not chunk:
        break
      outputfile.write(chunk)
      left -= len(chunk)

  def end_headers(self):
    if 'Range' not in self.headers:
      self.send_header('Accept-Ranges', 'bytes')
    super().end_headers()


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('directory')
  p.add_argument('--port', type=int, default=8765)
  a = p.parse_args()
  ThreadingHTTPServer(('127.0.0.1', a.port), partial(RangeHandler, directory=a.directory)).serve_forever()
