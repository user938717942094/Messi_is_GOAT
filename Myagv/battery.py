from pymycobot.myagv import MyAgv
import time

agv = MyAgv('/dev/ttyAMA2', 115200)

print("Status:", agv.get_motors_current())

print("Data:", agv.get_battery_info()[0])
print("Data1:", agv.get_battery_info()[1])
print("Data2:", agv.get_battery_info()[2])