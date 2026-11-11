import io
import re
import unittest
import zipfile
from selfwatch_v2.http_range import HTTPRangeReader


class Response(io.BytesIO):
    status=206
    def __init__(self,data,start,end,total,etag='test'):
        super().__init__(data)
        self.headers={'Content-Range':f'bytes {start}-{end}/{total}','ETag':etag}


class RangeReaderTests(unittest.TestCase):
    def test_zip_seek_and_repeated_blocks(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('frames/a.jpg',b'example image bytes'*20)
            archive.writestr('labels/a.json','{"boxes":[]}')
        payload=buffer.getvalue()
        def open_range(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
            return Response(payload[start:end+1],start,end,len(payload))
        with HTTPRangeReader('https://example.test/data.zip',opener=open_range,block_size=64,max_bytes=8192) as reader:
            with zipfile.ZipFile(reader) as archive:self.assertEqual(archive.read('labels/a.json'),b'{"boxes":[]}')
            reader.seek(0);reader.read(5);count=reader.requests
            reader.seek(0);reader.read(5)
            self.assertEqual(reader.requests,count)
            self.assertLess(reader.bytes_read,8192)

    def test_full_responses_and_wrong_ranges_are_rejected(self):
        def ignore(request,timeout):
            response=Response(b'x',0,0,10);response.status=200;return response
        with self.assertRaisesRegex(ValueError,'honor byte ranges'):HTTPRangeReader('https://example.test/x',opener=ignore)
        def wrong(request,timeout):return Response(b'x',1,1,10)
        with self.assertRaisesRegex(ValueError,'Unexpected HTTP'):HTTPRangeReader('https://example.test/x',opener=wrong)

    def test_budget_and_version_changes_fail_closed(self):
        def source(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
            return Response(b'x'*(end-start+1),start,end,100,etag='a' if end==0 else 'b')
        reader=HTTPRangeReader('https://example.test/x',opener=source,block_size=16,max_bytes=8)
        with self.assertRaisesRegex(ValueError,'download limit'):reader.read(1)
        reader=HTTPRangeReader('https://example.test/x',opener=source,block_size=16,max_bytes=100)
        with self.assertRaisesRegex(ValueError,'changed during'):reader.read(1)


if __name__=='__main__':unittest.main()
