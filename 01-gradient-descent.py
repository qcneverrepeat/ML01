# coding:utf-8

# ======================
# y = 5x**2 + 6*x + 9
def Y(x):
    # x = float(x)
    return 5*(x**2) + 6*x + 9
def D(x):
    # x = float(x)
    return 10*x + 6
# initialization
x = 0
# gradient descent
for i in range(50):
    x -= D(x)*0.1
# output
print(x,Y(x))

# ======================
# z = 4x^2 - 6y^3 + 2y
def Z(x,y):
    return 4*(x**2) - 6*(y**3) + 2*y
def D2(x):
    return 8*x
def D3(y):
    return - 18*(y**2) + 2
x,y = 0,0
for i in range(100):
    x -= D2(x)*0.1
    y -= D3(y)*0.1
print(x,y,Z(x,y))

# ======================
# z = 4x^2 + 6y^3 + 2xy + 2x
# must be a convex problem
def Z1(x,y):
    return 4*(x**2) + 6*(y**3) + 2*y*x + 2*x
def Dx(x,y):
    return 8*x + 2*y + 2
def Dy(x,y):
    return 18*(y**2) + 2*x
x,y = 0,0
z1 = z2 = Z1(x,y)
for i in range(500):
    x -= Dx(x,y)*0.01
    y -= Dy(x,y)*0.01
    z1,z2 = z2,Z1(x,y)
    print('i=%.0f,    x=%.4f,    y=%.4f,   z=%.4f'%(i,x,y,Z1(x,y)))
    # 迭代终止条件,331次收敛
    if z1 == z2:
        break

# 占位符%s既可以表示字符串str,还可以表示整数int,浮点数float;
# 占位符%d既可以表示整数int,还可以表示浮点数float(去除整数部分)
# 占位符%f既可以表示浮点数float,还可以表示整数int(默认保留6位小数)
# 注意:若想自主保留n位小数,可将其表示位%.nf
