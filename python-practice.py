# 8/9
# 語法練習
print("Hello world")
input()
# 變數(不能以數字開頭)
x = "Hello world"
print(x)
# type為型態(資料類型)
a = "Hello world"
b = 1
c = 3.14
d = True
print(type(a))  # string字串
print(type(b))  # integer整數
print(type(c))  # float浮點數
print(type(d))  # Boolean布林值
# "="為指派
x = 1
y = 1
z = x+y
print(z)
# 語法練習
a = int(input("輸入數字a:"))
b = int(input("輸入數字b:"))
c = a+b
print(c)
# 字串格式化
person = "Marry"
year = 20
height = 168.2
print("{} is {} years old, she is {} cm tall.".format(person, year, height))  # 新式
print("%s is %d years old, she is %f cm tall." %
      (person, year, height))      # 舊式
# 判斷式
eng = int(input("insert english score:"))
math = int(input("insert math score:"))
if eng >= 60 and math >= 60:
    print("因為英文得{}分和數學得{}分所以去吃海底撈".format(eng, math))
elif eng >= 60 or math >= 60:
    print("因為英文得{}分和數學得{}分所以去吃肯德基".format(eng, math))
else:
    print("罰站一小時")
# 迴圈
for i in range(0, 11):
    if i % 2 == 0:
        print(i)
    else:
        print()
m = 0
while m <= 10:
    if m % 2 == 0:
        print(m)
    else:
        print()
    m += 1
# list
sampleA = [23, 45, 12, 76, 235, 6785, 2423, 3453, 6452]
sampleB = [22, 44, 66, 88]
for i in sampleA:
    print(i, end=' ')
sampleA.append(40)   # 新增
sampleA.remove(45)   # 移除
sampleA.extend(sampleB)
print(sampleA)
# 分數加總(平均)
score = [90, 89, 78, 63]
sum = 0
for i in score:
    sum += i
print("總和:", sum)
print("平均:", sum/len(score))
# 函式(function)
dog1 = "小白"
dog1point = 0
dog1stepsize = 1


def dogrun(dogname, currentpoint, stepsize):
    print(dogname, "從位置", currentpoint)
    currentpoint += stepsize
    print("跑到位置", currentpoint, "了")
    return currentpoint


for i in range(3):
    dog1point = dogrun(dog1, dog1point, dog1stepsize)
