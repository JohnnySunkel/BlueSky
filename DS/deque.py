class Deque:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    def addFront(self, item):
        self.items.append(item)
        
    def addRear(self, item):
        self.items.insert(0, item)
        
    def removeFront(self):
        return self.items.pop()
    
    def removeRear(self):
        return self.items.pop(0)
    
    def size(self):
        return len(self.items)
    
    
# Test it
d = Deque()
d.isEmpty() # True
d.addRear(4)
d.addRear('dog')
d.addFront('cat')
d.addFront(True)
d.size() # 4
d.isEmpty() # False
d.addRear(8.4)
d.removeRear() # 8.4
d.removeFront() # True
