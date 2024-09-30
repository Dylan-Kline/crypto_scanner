import asyncio
from collections import deque

class AsyncCappedList:

    def __init__(self, max_len):
        self.deque = deque(maxlen=max_len)
        self.lock = asyncio.Lock()

    async def append(self, item):
        async with self.lock:
            self.deque.append(item)

    async def extend(self, list):
        async with self.lock:
            self.deque.extend(list)

    async def get_list(self):
        async with self.lock:
            return list(self.deque)

    async def get_size(self):
        async with self.lock:
            return len(self.deque)
    
    async def get_last(self):
        async with self.lock:
            if len(self.deque) > 0:
                return self.deque[-1]
            return None
