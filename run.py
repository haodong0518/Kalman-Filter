import numpy as np
import math
import re
import matplotlib.pyplot as plt


def file_process(file):
    f = open(file, "r")
    data = f.read().rstrip()

    lines = data.split('\n')
    matrix = []

    # Iterate through lines and extract numbers from each line
    for line in lines:
        values = line.split()
        try:
            # Convert values to floats and add them to the matrix
            matrix.append([float(value) for value in values])
        except ValueError:
            pass
    return matrix

groundtruth = file_process('ds0_Groundtruth.dat')
M_odome  = file_process('ds0_Odometry.dat')
landmark = file_process('ds0_Landmark_Groundtruth.dat')
measure = file_process("ds0_Measurement.dat")
barcode = file_process('ds0_Barcodes.dat')

## Input the [time, forward veloity, angular velocity], state_0 is the groundtruth data
## return the point [t, px, py, heading]
def found_path(M_odome , state_0):
    path=[]
    #initial state:
    s1 = state_0[0][3]
    px = state_0[0][1]
    py = state_0[0][2]

    for i in range(len(M_odome)):
        # time difference
        if i == len(M_odome) - 1:
            dt=0
        else:
            dt = M_odome[i+1][0] - M_odome[i][0]


        #operations
        w1 = M_odome[i][-1] #angular velocity
        v1 = M_odome[i][-2] #forward velocity

        if w1 ==0:   
            dpx = v1*math.cos(s1)*dt
            dpy = v1*math.sin(s1)*dt

        elif w1 !=0:
            dpx = ((v1/w1)*(math.sin(s1+w1*dt) - math.sin(s1)))
            dpy = ((v1/w1)*((-1)*math.cos(s1+w1*dt) + math.cos(s1)))

        dtheta = w1 * dt

        #updata states
        px = px + dpx
        py = py + dpy
        s1 = s1 + dtheta
        
        if s1 > math.pi:
            s1 = s1 - 2*math.pi
        elif s1 < -math.pi:
            s1 = s1 + 2*math.pi
        pass
        
        t = M_odome[i][0]
        
        path.append([t, px,py,s1])
    return path   

# Find all the points based on the Odometry.dat
path_odo = found_path(M_odome,groundtruth)

# the input date type is [t, px,py,theta]
# output the list x, y for easier plotting

def plotting_data(data):
    x = []
    y=[]
    for i in range(len(data)):
        x.append(data[i][1])
    for j in range(len(data)):
        y.append(data[j][2])
    return x,y

## Find the the corresponded position in Odometry.dat matched with measurement data 
## Input measurement data [time, subject,range, bearing]
## pick positions in path from Odometry.dat, that match the time point in measurement file
def measure_data_pick(measurement_data, path_odome):
    corr = []
    for mea in measurement_data:
    ## looping the measurement time point 
        for posi in path_odome :
        ## looping the control time point
            if mea[0] < posi[0]:
                corr.append(posi)
                break
    return corr

## get rid of the measure with other robotics, only keep measure with landmarks
measure_mark = []
for vision in measure:
    for code in barcode:
        if vision[1] == code[1] and code[0]>5:
            measure_mark.append(vision)
            
# find the points[t,px,py,heading] from Odometry.dat that time is matched with measurement data.
predict = measure_data_pick(measure_mark,path_odo) ## time, px, py, theta
# find the opearations[t,forward velocity, angular velocity] from Odometry.dat that time is matched with measurement data.
match_odo = measure_data_pick(measure_mark, M_odome)

def return_code_p(code):
    for i in range(len(barcode)):
        if code == barcode[i][1]:
            subject = barcode[i][0]
            for j in range(len(landmark)):
                if landmark[j][0] == subject:
                    return landmark[j][1],landmark[j][2]
## input the landmark code, return its positions.


## give the theta calculated by math.atan2, and heading of each point, calculate its bearing
def find_bearing(theta , heading):
    b= theta - heading
    if abs(theta) > math.pi/2 and abs(heading) > math.pi/2 and theta*heading<0:
        if theta > heading:
            b = -(2*math.pi - abs(theta) -abs(heading))
        elif theta < heading:
            b = 2*math.pi - abs(theta) -abs(heading)
    return b

def weight_compute(alpha, beta, k ,n):
    lamda = ((alpha**2) * (n + k)) - n
    w_m = [lamda/(n+lamda)]
    w_c = [(lamda/(n+lamda)) + 1 - alpha**2 + beta]
    for i in range(1,(2*n)+1):
        w_m.append(1/(2*(n+lamda)))
        w_c.append(1/(2*(n+lamda)))
    return w_m, w_c

n = 3
#n + k =3
k = 1
alpha = 1
beta = 2
lamda = ((alpha**2) * (n + k)) - n
w_m,w_c = weight_compute(alpha, beta,k,n)
w_m = np.array(w_m).reshape(1,7)
w_c = np.array(w_c).reshape(1,7)
r = math.sqrt(n+lamda)

# give state [px,py,theta] 
# return next state [px,py,theta]
# i is the state in the whole iterations algorithm
def find_next_state_position(X0,i):
    if i == len(predict) - 1:
        dt=0
    else:
        dt = predict[i+1][0] - predict[i][0]
    #operations
    w1 = match_odo[i][-1] #angular velocity
    v1 = match_odo[i][-2] #forward velocity
    s1 = X0[-1]
    if w1 ==0:   
        dpx = v1*math.cos(s1)*dt
        dpy = v1*math.sin(s1)*dt

    elif w1 !=0:
        dpx = ((v1/w1)*(math.sin(s1+w1*dt) - math.sin(s1)))
        dpy = ((v1/w1)*((-1)*math.cos(s1+w1*dt) + math.cos(s1)))

    dtheta = w1 * dt

    #updata states
    px = X0[0] + dpx
    py = X0[1] + dpy
    s1 = X0[2] + dtheta

    if s1 > math.pi:
        s1 = s1 - 2*math.pi
    elif s1 < -math.pi:
        s1 = s1 + 2*math.pi
    pass

    return [px,py,s1]

# the is the function used in the algorithm, input the state, and for loop i, return the rangee and bearing to the landmark
def find_measure(X0,i):
    mea_position = []
    lx, ly = return_code_p(measure_mark[i][1])
    #print(f'subject {measure_mark[i][1]}')
    rangee = math.sqrt( (lx-X0[0])**2 + ( ly - X0[1])**2 )
    theta = math.atan2( (ly -X0[1]) , (lx - X0[0]) )
    # bearing = theta- X0[2]
    #print(f'bearing calculate {theta , X0[2]}')
    bearing = find_bearing(theta , X0[2])
    return [rangee, bearing]

def run_algorithm():
    miu1 = np.array(predict[0][1:])
    cov_x = np.array([
        [0.001,0,0],
        [0,0.001,0],
        [0,0,0.001]
    ])
    correct_path = []
    for i in range(len(predict)):
        if i < len(predict)-1:
            n=3
            X0 = np.array(miu1)
            for a in range(n):
                X0 = np.vstack((X0, miu1 + r*np.linalg.cholesky(cov_x)[:,a]))
            for b in range(n):
                X0 = np.vstack((X0, miu1 - r*np.linalg.cholesky(cov_x)[:,b]))
            #print(f'X0 {predict[i][0],X0}')
            ## step 3 X_t+1 = g(X_t-1)
            X1 = find_next_state_position(X0[0],i)
            for j in range(len(X0)):
                X1 = np.vstack((X1, find_next_state_position(X0[j],i)))
            X1=X1[1:]
            # X1 is the next state based on the last state
            miu_bar_t = np.dot(w_m, X1)
            #print(f'X1 {X1}')
            # Step 5
            R = np.array([
                [0.0001 , 0, 0],
                [0,0.0001,0],
                [0, 0 ,0.0001]
            ])
            cov_t = []
            cov_t.append(R)
            for k in range(len(X1)):
                a = (X1[k]-miu_bar_t).reshape(3,1)
                cov_t.append(w_c[0][k]*np.dot(a,a.T))
            cov_t = sum(cov_t)
            #print(f'cov_t {cov_t}')

            # Step 6
            X_t_bar = miu_bar_t
            for h in range(n):
                X_t_bar = np.vstack((X_t_bar , miu_bar_t + r*np.linalg.cholesky(cov_t)[:,h]))
            for g in range(n):
                X_t_bar = np.vstack((X_t_bar , miu_bar_t - r*np.linalg.cholesky(cov_t)[:,g]))
            #print(f'X_t_bar {X_t_bar}')
            #Step 7
            Z_t_bar = find_measure(X_t_bar[0],i)
            for u in range(len(X_t_bar)):
                Z_t_bar = np.vstack((Z_t_bar , find_measure(X_t_bar[u],i )))
            Z_t_bar = Z_t_bar[1:] #get rid of first row, for easy adding, I add first row twice
            #print(f'Z_t_bar{Z_t_bar}')
            # step 8
            z_mean = np.dot(w_m, Z_t_bar)

            Q = np.array([
            [0.1 , 0],
            [0,0.1],
            ])

            # Step 9
            S_t =[]
            S_t.append(Q)
            for q in range(len(Z_t_bar)):
                a = (Z_t_bar[q]-z_mean).reshape(2,1)
                S_t.append(w_c[0][q]*np.dot(a,a.T))
            S_t = sum(S_t)

            # Step 10
            cov_xz = []
            for e in range(2*n):
                a = (Z_t_bar[e]-z_mean).reshape(2,1)
                b =  (X_t_bar[e] - miu_bar_t).reshape(3,1)
                cov_xz.append(w_c[0][e]* np.dot(b,a.T))
            cov_xz = sum(cov_xz)

            S_t_inv = np.linalg.inv(S_t)
            K_t = np.dot (cov_xz, S_t_inv)
            z_t = np.array(measure_mark[i][2:]).reshape(2,1)
            #print(f'measure {z_t}')
            #print(f'subject {measure_mark[i][1]}')
            miu_t1 = miu_bar_t.reshape(3,1) + np.dot (K_t , z_t - z_mean.T)
            #print(np.dot (K_t , z_t - z_mean.T).shape)
            #print(K_t@S_t@K_t.T)
            var_t  = cov_t - K_t@S_t@K_t.T
            #print(f'miu{miu_t1}')
            #print(f'var {var_t}')

            correct_path.append(miu_t1)
            miu1 = miu_t1.reshape(1,3)    
    return correct_path

## For question 2 
def question2():
    q2 =[[0,0,0],
    [1,0.5,0],
    [2,0,-1/(2*math.pi)],
    [3,0.5,0],
    [4,0,1/(2*math.pi)],
    [5,0.5,0],
    [6,0,0]]
    
    state_0 = [[0,0,0,0]] #time, X, Y , Theta
    
    path = found_path(q2,state_0)
    x2 = []
    y2=[]
    heading= []
    for i in range(len(path)):
        x2.append(path[i][1])
    for j in range(len(path)):
        y2.append(path[j][2])
    for k in range(len(path)):
        heading.append(path[k][-1])
    
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x2,y2,'r',label='dead-reckoned path')
    for i in range(1,len(x2),2):
        plt.quiver(x2[i],y2[i],math.cos(heading[i]),math.sin(heading[i]),color='b',width=0.002)
    ax = fig.add_subplot(111) 
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    plt.title('question2 estimated trajectory')
    plt.legend()
    plt.show()

                   
                   
# For question 3
def question3():        
    gx = []
    gy=[]
    for i in range(len(groundtruth)):
        gx.append(groundtruth[i][1])
    
    for j in range(len(groundtruth)):
        gy.append(groundtruth[j][2])
    
    fig = plt.figure(figsize=(8, 4))
    x,y = plotting_data(path_odo)
    plt.plot(x,y,'r',label='dead-reckoned path')
    plt.plot(gx,gy,'g',label='groundtruth path')
    plt.title('Estimated Trajectory')
    ax = fig.add_subplot(111) 
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    plt.legend()
    plt.show()
    
def question6():
    r_p = [[2,3,0],[0,3,0],[1,-2,0]]
    landmark_subs = [6,13,17]
    positions = []
    distance = []
    bearing = []
    for mark in landmark:
        if mark[0] in landmark_subs:
            positions.append(mark[1:3])
    for i in range(len(r_p)):
        distance.append(math.sqrt((positions[i][0] - r_p[i][0])**2 + (positions[i][1] - r_p[i][1])**2))
        
    for i in range(len(r_p)):
        theta = math.atan2(   (positions[i][1] -r_p[i][1] ),(positions[i][0] - r_p[i][0] )  )
        heading = r_p[i][-1]
        b= theta - heading
        if abs(theta) > math.pi/2 and abs(heading) > math.pi/2 and theta*heading<0:
            if theta > heading:
                b = -(2*math.pi - abs(theta) -abs(heading))
            elif theta < heading:
                b = 2*math.pi - abs(theta) -abs(heading)
        bearing.append(b)
    return distance, bearing

def plot_full_algorithm():
    fig = plt.figure(figsize=(8, 4))
    correct_path = run_algorithm()
    x = []
    y = []
    for i in range(len(correct_path)):
        x.append(correct_path[i][0])
    for j in range(len(correct_path)):
        y.append(correct_path[j][1])
      
    gx,gy= plotting_data(groundtruth)
    fx,fy=plotting_data(predict)
    plt.plot(x,y,'b',label='corrected path')
    plt.plot(fx,fy,'r',label = 'dead-reckoned') 
    plt.plot(gx,gy,'g',label='groundtruth path')
    plt.title('Estimated Trajectory and Corrected Trajectory by UKF')
    ax = fig.add_subplot(111) 
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    for i in range(1,len(correct_path),50):
        plt.quiver(x[i],y[i],math.cos(correct_path[i][2]),math.sin(correct_path[i][2]),color='b',width=0.002)
    plt.legend()
    plt.show()
                   

def run():
    
    ## Foe question 2
    print("The plot for Q2")
    question2()

            
    ## For question 3
    print("The plot for Q3")
    question3()
    
    
    
    ##For Question 6 
    print("Question6")
    d, theta = question6()
    print("range:")
    print(d)
    print("headings")   
    print(theta)
    

    ## For the full filter
    plot_full_algorithm()

run()
    
    