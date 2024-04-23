import time
import numpy as np
import matplotlib.pyplot as plt
from ikpy.chain import Chain
from ikpy.utils import plot
from pymycobot.mycobot import MyCobot
from scipy.spatial.transform import Rotation as R

urdf_file = r"C:\Users\oli\Music\DPOD\0_KG\code\3_vscode\KG_KAIROS_VSCODE\4_apr\mycobot_320_pi_2022"

def ikpy_test(urdf_file_path=None, position=None, angles_degrees=None, return_val=True, graph_print=False, info_print=True):
    arm_chain = Chain.from_urdf_file(urdf_file_path)
    
    target_position = position

    euler_angles_degrees = angles_degrees

    euler_angles_radians = np.radians(euler_angles_degrees)

    rotation_matrix = R.from_euler('xyz', euler_angles_radians).as_matrix()
    target_orientation = rotation_matrix

    ik = arm_chain.inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation,
        orientation_mode="all")

    ik_deg = np.degrees(ik).tolist()

    if return_val == True:
        return ik_deg[1:]
    
    if info_print == True:
        print("IK (Degrees):")
        print(ik_deg)

    if graph_print == True:
        fig, ax = plot.init_3d_figure()
        arm_chain.plot(ik, ax)
        ax.legend()
        plt.show()


mc = MyCobot('COM3',115200)
mc.send_angles([0,0,0,0,0,0],10)
time.sleep(5)
print("ang 0000")

# for i in range(1, 20):
#     a= i/100
#     target_1 = [a, 0, 0.5]
#     ang_1 = [0,90,0]
#     mc.send_angles(ikpy_test(urdf_file,target_1,ang_1),30)
#     time.sleep(0.5)
# cm단위
target_1 = [-0.3, 0.1, 0.4]
ang_1 = [0,0,45]
mc.send_angles(ikpy_test(urdf_file,target_1,ang_1),30)
print(target_1)