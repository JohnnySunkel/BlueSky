class PriorityQueue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return not self.items

    def insert(self, item):
        self.items.append(item)

    def remove(self):
        max = 0
        for i in range(1, len(self.items)):
            if self.items[i] > self.items[max]:
                max = i
            item = self.items[max]
            del self.items[max]
            return item
