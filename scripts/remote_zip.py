"""Read selected ZIP members using HTTP byte ranges, without fetching the archive."""
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RemoteZipIO(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.session.headers['Connection'] = 'close'
        r = self.session.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=30)
        r.raise_for_status()
        if r.status_code != 206:
            r.close()
            raise RuntimeError('Server does not support byte ranges')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.url = r.url
        r.close()
        self.pos = 0
        self.downloaded = 0
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b''
        start = self.pos
        for (a,b), value in self.cache.items():
            if a <= start and start + size <= b:
                self.pos += size
                return value[start-a:start-a+size]
        # Fetch enough for nearby local-header reads without an extra round trip.
        end = min(self.size, start + max(size, 65536))
        response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end-1}'}, timeout=90)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f'Expected HTTP 206, got {response.status_code}')
        data = response.content
        self.downloaded += len(data)
        if len(data) != end-start:
            raise IOError('Truncated range response')
        if len(data) <= 4*1024*1024:
            self.cache[start,end] = data
        self.pos += size
        return data[:size]
