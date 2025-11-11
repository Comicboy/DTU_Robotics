import numpy as np

def calculateMatrix(theta,d,a,alpha):
    A = np.array([[np.cos(theta) , -np.sin(theta)*np.cos(alpha) , np.sin(theta)*np.sin(alpha) , a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                  [0,np.sin(alpha),np.cos(alpha),d],
                  [0,0,0,1]], dtype=float)
    return A#np.array([[cosd(theta) , -sind(theta)*cosd(alpha) , sind(theta)*sind(alpha) , a*cosd(theta)],[sind(theta) cosd(theta)*cosd(alpha) -cosd(theta)*sind(alpha) a*sind(theta)],[0 sind(alpha) cosd(alpha) d],[0 0 0 1]])

def forwards_kinematics(theta_1,theta_2,theta_3,theta_4):
    T_01 = calculateMatrix(theta_1,50,0,np.pi/2)
    T_12 = calculateMatrix(theta_2,0,93,0)
    T_23 = calculateMatrix(theta_3,0,93,0)
    T_34 = calculateMatrix(theta_4,0,50,0)
    
    T_03=T_01@T_12@T_23
    T_04=T_01@T_12@T_23@T_34
    
    T_35=np.array([[np.cos(theta_4),-np.sin(theta_4),0,35],[np.sin(theta_4),np.cos(theta_4),0,45],[0,0,1,0], [0,0,0,1]],dtype=float)
    T_05 = T_01@T_12@T_23@T_35

    return T_03,T_04,T_05


'''
def inverse_kinematics(phi):
    # Constants
    d1 = 0.05
    a2 = 0.093
    a3 = 0.093
    a4 = 0.05

    radius = 0.032
    p_0c = np.array([0.15, 0, 0.12])

    x_04 = 0.0

    # Assume phi is defined (you need to assign a value, e.g., phi = 30)
    # phi = 0  # example angle in degrees

    # Compute positions
    p_04 = p_0c + radius * np.array([0, np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))])
    o_04 = p_04

    # Geometry calculations
    d_4z = a4 * x_04
    u = np.sqrt(a4**2 - d_4z**2)

    theta1 = np.rad2deg(np.arctan2(o_04[1], o_04[0]))
    x43 = u * np.cos(np.deg2rad(theta1))
    y43 = u * np.sin(np.deg2rad(theta1))

    xc = o_04[0] - x43
    yc = o_04[1] - y43
    zc = o_04[2] - d_4z

    r = np.sqrt(xc**2 + yc**2)
    s = zc - d1

    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    s3 = np.sqrt(1 - c3**2)  # remember the other solution: -np.sqrt(1 - c3**2)
    theta3 = np.rad2deg(np.arctan2(s3, c3))

    theta2 = np.rad2deg(np.arctan2(s, r) - np.arctan2(a3 * s3, a2 + a3 * c3))

    alfa = np.rad2deg(np.arcsin(x_04))
    theta4 = alfa - theta2 - theta3

    q = np.array([theta1, theta2, theta3, theta4])
    return q
'''

def inverseKinematics(x,o):
    d4=50
    x_c=np.round(o[0]-x[0]*d4,4)
    y_c=np.round(o[1]-x[1]*d4,4)
    z_c=np.round(o[2]-x[2]*d4,4)
    d1=50
    #1 = atan2(x_c,y_c);
    q0 = np.arctan2(y_c,x_c)
    r_sq=x_c**2+y_c**2
    s=z_c-d1
    c2=np.round((r_sq+s*s-93*93-93*93)/(2*93*93),4)
    #print("c2 =", c2)
    #q3 = [atan2(c3,sqrt(1-c3^2)) atan2(c3,-sqrt(1-c3^2))];
    q2 = [np.atan2(np.sqrt(1-c2*c2),c2),np.atan2(-np.sqrt(1-c2*c2),c2)]
    #print("q2 =", np.rad2deg(q2))
    #q2 = atan2(sqrt(r_sq),s)-atan2(93+93*c3,93*sin(q3(1)));
    q1 = [np.atan2(s,np.sqrt(r_sq))-np.atan2(93*np.sin(q2[0]),93+93*c2),np.atan2(s,np.sqrt(r_sq))-np.atan2(93*np.sin(q2[1]),93+93*c2)]
    #print("q1 =", np.rad2deg(q1))
    #q3 = [np.arctan2(np.round(np.sqrt(1-x[2]**2),4),x[2]), np.atan2(x[1]*d4,x[0]*d4)]
    q3 = [np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))-q1[0]-q2[0],np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))-q1[1]-q2[1]]
    #print("q3 = ", np.rad2deg(q3))
    #q4 = np.atan2(x(3),np.sqrt(1-x(3)^2))
    #q4 = atan2(sqrt(1-x(3)^2),x(3));
    #q4 = atan2(sqrt(1-(sin(q1)*x(1)-cos(q1)*x(2))^2),sin(q1)*x(1)-cos(q1)*x(2));
    #q4 = atan2(x(2),x(1));
    return [q0,q1,q2,q3]

if __name__ == "__main__":
    # Test the functions
    T_03,T_04, T_05 = forwards_kinematics(np.deg2rad(0), np.deg2rad(45), np.deg2rad(-45), np.deg2rad(0))

    #print("T_04:\n", T_04)

    q = inverseKinematics(T_04[0:3,0],T_04[0:3,3])

    #q = inverseKinematics([1,0,0], [150,0,50])

    print("q = ", np.rad2deg(q[0]), np.rad2deg(q[1][0]), np.rad2deg(q[2][0]), np.rad2deg(q[3][0]))
    T_test,T_test_2,_ = forwards_kinematics(q[0], q[1][0], q[2][0], q[3][0])

    print("DDifference = ",np.round(T_test_2 - T_04,3))

    #print("T_03:\n", T_test)
    #print("T_04:\n", T_test_2)

    #print("Joint angles (degrees):", inverse_kinematics(0))