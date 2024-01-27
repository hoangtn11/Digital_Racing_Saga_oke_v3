import queue
import time

def put_to_queue_no_wait_no_block(item, q):
    
    if q.full():
        try:
            q.get_nowait()
        except:
            pass

    try:
        q.put_nowait(item)
    except:
        pass

def get_fast(queue, block=True, timeout=None):
   
    with queue.not_empty:
        if not block:
            if not queue._qsize():
                return 
        elif timeout is None:
            while not queue._qsize():
                queue.not_empty.wait()
        elif timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        else:
            print("hahaa")
            endtime = time() + timeout
            while not queue._qsize():
                remaining = endtime - time()
                if remaining <= 0.0:
                    return 
                queue.not_empty.wait(remaining)
        item = queue._get()
        queue.not_full.notify()
        return item

