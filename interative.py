import Queue
import threading
import subprocess

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def getOutput(outQueue):
    outStr = ''
    try:
        while True: #Adds output from the Queue until it is empty
            outStr+=outQueue.get_nowait()

    except Queue.Empty:
        return outStr

p = subprocess.Popen("bash", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, universal_newlines=True)

outQueue = Queue()
errQueue = Queue()

outThread = Thread(target=enqueue_output, args=(p.stdout, outQueue))
errThread = Thread(target=enqueue_output, args=(p.stderr, errQueue))

outThread.daemon = True
errThread.daemon = True

outThread.start()
errThread.start()

someInput = raw_input("Input: ")

p.stdin.write(someInput)
errors = getOutput(errQueue)
output = getOutput(outQueue)
