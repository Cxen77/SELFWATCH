"""Bounded seekable HTTP payload reader for large public research archives."""
from collections import OrderedDict
import io
import re
import urllib.request


class HTTPRangeReader(io.RawIOBase):
    def __init__(self,url,max_bytes=384*1024**2,block_size=1024**2,cache_blocks=8,opener=None):
        super().__init__()
        if min(max_bytes,block_size,cache_blocks)<1:raise ValueError('Positive download/cache limits required')
        self.url=url;self.limit=max_bytes;self.block_size=block_size;self.cache_blocks=cache_blocks
        self.opener=opener or urllib.request.urlopen
        self.bytes_read=self.requests=self.position=0
        self.cache=OrderedDict();self.size=None;self.etag=None;self.modified=None
        self._fetch(0,0)

    def _fetch(self,start,end):
        if self.bytes_read+end-start+1>self.limit:raise ValueError('Archive payload download limit exceeded')
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'SELFWATCH-research/0.1'}
        if self.etag:headers['If-Range']=self.etag
        elif self.modified:headers['If-Range']=self.modified
        with self.opener(urllib.request.Request(self.url,headers=headers),timeout=45) as response:
            if response.status!=206:raise ValueError('Server did not honor byte ranges; refusing full archive download')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or tuple(map(int,match.groups()[:2]))!=(start,end):raise ValueError('Unexpected HTTP byte range')
            size=int(match.group(3));etag=response.headers.get('ETag');modified=response.headers.get('Last-Modified')
            if self.size is not None and (size!=self.size or (self.etag and etag!=self.etag)):
                raise ValueError('Remote archive changed during reading')
            self.size,self.etag,self.modified=size,etag,modified
            data=response.read(end-start+2)
            if len(data)!=end-start+1:raise ValueError('Truncated or oversized byte-range response')
        self.bytes_read+=len(data);self.requests+=1
        return data

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if target<0:raise ValueError('Invalid range-reader seek')
        self.position=target
        return target

    def read(self,size=-1):
        if self.closed:raise ValueError('Reader is closed')
        amount=max(0,self.size-self.position) if size is None or size<0 else min(size,max(0,self.size-self.position))
        if amount>64*1024**2:raise ValueError('Refusing an unbounded archive read')
        parts=[]
        while amount:
            index=self.position//self.block_size
            if index not in self.cache:
                start=index*self.block_size;end=min(self.size-1,start+self.block_size-1)
                self.cache[index]=self._fetch(start,end)
                while len(self.cache)>self.cache_blocks:self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            block=self.cache[index];offset=self.position-index*self.block_size
            count=min(amount,len(block)-offset)
            parts.append(block[offset:offset+count]);self.position+=count;amount-=count
        return b''.join(parts)
