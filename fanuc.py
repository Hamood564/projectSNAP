import win32com.client
import time
import pandas as pd

# Load motion path data
motion_data = pd.read_csv("motion_profile.csv")

# Connect to Fanuc Robot
robot = win32com.client.Dispatch("FRRobot.FRCRobot")
robot.Connect("192.168.115.175")

# Initialize Position Register
pos_reg = robot.RegNumerics(1)  # PR[1]

for index, row in motion_data.iterrows():
    x, y, theta = row["X"], row["Y"], row["Theta"]

    # Send to Fanuc
    pos_reg(1).Value = x  # X
    pos_reg(2).Value = y  # Y
    pos_reg(6).Value = theta  # Rotation

    print(f"Sent Position: X={x}, Y={y}, Theta={theta}")
    time.sleep(0.2)

robot.Disconnect()
print("Motion Path Executed Successfully!")
