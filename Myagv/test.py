from pymycobot.myagv import MyAgv
import time

MA = MyAgv('/dev/ttyAMA2', 115200)

MA.go_ahead(50)
time.sleep(2)

MA.retreat(50)
time.sleep(2)

MA.pan_left(50)
time.sleep(2)

MA.pan_right(50)
time.sleep(2)

MA.clockwise_rotation(50)
time.sleep(2)

MA.counterclockwise_rotation(50)
time.sleep(2)

MA.stop()