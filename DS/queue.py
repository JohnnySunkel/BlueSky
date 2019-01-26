class Queue:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
        
    def enqueue(self, item):
        self.items.insert(0, item)
        
    def dequeue(self):
        return self.items.pop()
    
    def size(self):
        return len(self.items)
    
    
# Test it
q = Queue()
q.isEmpty() # True
q.enqueue(4)
q.enqueue('dog')
q.enqueue(True)
q.size()
q.isEmpty() # False
q.enqueue(8.4)
q.dequeue() # 4
q.dequeue() # 'dog'
q.size() # 2
