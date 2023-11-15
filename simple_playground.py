import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import Project2
import tkinter as tk
from tkinter import ttk
from matplotlib.animation import FuncAnimation



class Car():
    def __init__(self) -> None:
        self.radius = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction(self, state):
        '''
        此function為模擬時，給予車子隨機數字讓其走動。
        不需使用此function。
        '''
        return r.randint(0, self.n_actions-1)

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        '''
        請更改此處code，依照自己的需求撰寫。
        '''
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

def run_example(name):
    global states, position
    c, y = 0, 0
    # use example, select random actions until gameover
    p = Playground()
    while y < 35.0 and c < 100:
        temp = []
        states = []
        position = []
        state = p.reset()
        while not p.done:
            # print every state and position of the car
            x,y = p.car.getPosition('center').x,p.car.getPosition('center').y
            position.append([x, y])

            # select action randomly
            # you can predict your action according to the state here
            if(name == "train6dAll"):
                state.insert(0, y)
                state.insert(0, x)        
            action = Project2.MLP_prediction(state, name)
            
            state.append(action[0][0]-40)
            temp.append(state)

            # take action
            state = p.step(action)
            print(state, p.car.getPosition('center'))
            states.append(state)
        c += 1
    print(c)
    my_array = np.array(temp)

    # 指定保存的文件名和分隔符
    if name == "train4dAll":
        file_name = "track4D.txt"
    if name == "train6dAll":
        file_name = "track6D.txt"
    delimiter = ' '

    # 使用savetxt保存txt文件
    np.savetxt(file_name, my_array, fmt='%0.3f', delimiter=delimiter)
    return position

# 退出TK
def quit():
    root.quit()
    root.destroy()

# 啟動自走車
def Run():
    global index, x_data, y_data, data

    # 將動畫重新撥放
    index = 0
    x_data, y_data = [], []

    # 取得所選數據
    name = box.get()
    print(name)
    ax.set_title(name)

    data = np.array(run_example(name))
    
# 更新函數
def update(frame):
    global index, circle_center, states, position
    if index < len(data):
        x, y = data[index]
        x_data.append(x)
        y_data.append(y)
        scatter.set_offsets(np.column_stack([x_data, y_data]))

        # 移動車的位置
        circle_center = (x, y)
        circle.set_center(circle_center)

        # 更新距離資訊，座標位置
        text_to_display = (f'Front:{states[index][-3]:.3f} \n\nRight:{states[index][-2]:.3f} \n\nLeft  :{states[index][-1]:.3f} ')
        text_object.set_text(text_to_display)

        position_to_display = (f'X:{position[index][0]:.3f} \n\nY:{position[index][1]:.3f} ')
        text_oposition.set_text(position_to_display)

        # 更新索引
        index += 1
    return scatter, circle


if __name__ == "__main__":
    # 紀錄 track
    states = []

    # 起動時預設執行 train4dAll
    data = np.array(run_example("train4dAll"))

    # 初始化 Tkinter
    root = tk.Tk()
    root.title('self-propelled vehicle')
    
    # 初始化 Matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [], marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("train4dAll")

    # 初始化前左右的距離文字，座標位置
    text_object = ax.text(0, 37, '', fontsize=15, ha='center', va='center')
    text_oposition = ax.text(20, 2, '', fontsize=15, ha='center', va='center')

    # 將 Matplotlib 的 FigureCanvasTkAgg 嵌入Tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # 退出 Tkinter button
    button = tk.Button(master=root, text='Quit', command=quit)
    button.pack(side=tk.RIGHT)

    # comboboxText
    tk.Label(root, text='    ').pack(side=tk.LEFT)
    comboboxText = tk.StringVar()
    box = ttk.Combobox(root, textvariable=comboboxText, state='readonly')
    box['values'] = ["train4dAll", "train6dAll"]
    box.pack(side=tk.LEFT)
    box.current(0)

    # 訓練 button
    buttonRun = tk.Button(master=root, text='Train!', command=Run)
    buttonRun.pack(side=tk.LEFT)

    # playgroung 座標
    ground_data = [
        [18, 40],
        [30, 37],
        [-6, -3],
        [-6, 22],
        [18, 22],
        [18, 50],
        [30, 50],
        [30, 10],
        [6, 10],
        [6, -3],
        [-6, -3]
    ]

    # 繪製 playgroung
    ax.plot([ground_data[0][0], ground_data[1][0]], [ground_data[0][1], ground_data[1][1]], color="blue")
    ground_data = ground_data[2:-1]
    for i in range(len(ground_data)):
        ax.plot([ground_data[i][0], ground_data[(i+1)%8][0]], [ground_data[i][1], ground_data[(i+1)%8][1]], color="red")

    # 繪製直徑為6的車
    circle = Circle((0, 0), 3, edgecolor='b', facecolor='none', label='circle')
    ax.add_patch(circle)

    # 初始化數據
    x_data, y_data = [], []

    # 初始化數據索引
    index = 0

    # 定義動畫
    ani = FuncAnimation(fig, update, frames=len(data), interval=100)

    # 啟動Tkinter的主循環
    root.mainloop()
