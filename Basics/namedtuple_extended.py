from collections import namedtuple

Car = namedtuple('Car', 'color mileage')

class MyCarWithMethods(Car):
    def hexcolor(self):
        if self.color == 'red':
            return '#ff0000'
        else:
            return '#000000'


# Alternative using _fields property
# Car = namedtuple('Car', 'color mileage')
# ElectricCar = namedtuple('ElectricCar', Car._fields + ('charge',))
