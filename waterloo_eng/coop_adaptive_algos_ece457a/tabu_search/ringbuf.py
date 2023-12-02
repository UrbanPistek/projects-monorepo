#!python

class RingBuffer(object):

    def __init__(self, max_size=10):
        """Initialize the Ring Buffer with a max_size if set, otherwise
        max_size will elementsdefault to 10"""

        self.buffer = [None] * max_size
        self.head = 0
        self.tail = 0
        self.max_size = max_size

    def enqueue(self, item):
        """Insert an item at the back of the Ring Buffer"""

        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.max_size

    def contains(self, item):
        """Check if an item is present in the buffer"""

        # print(f"\nRing Buffer: {self.buffer}")

        # Empyt list
        if all(p == None for p in self.buffer):
            return False

        for i in range(len(self.buffer)):
            if self.buffer[i] == item: 
                # print ("Ring Buffer Contains Same Item") 
                # print(f"Ring Buffer: {self.buffer[i]} Input: {item}")
                return True
        
        return False

    def is_full(self):
        """Return True if the tail of the CircularBuffer is one before the head,
        otherwise return False"""
        return self.tail == (self.head-1) % self.max_size

    def back(self):
        """Return the item at the back of the Ring Buffer"""
        return self.buffer[self.tail]

    def front(self):
        """Return the item at the front of the Ring Buffer"""
        return self.buffer[self.head]

    def dequeue(self):
        """Return the item at the front of the Ring Buffer and remove it"""

        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.max_size
        return item