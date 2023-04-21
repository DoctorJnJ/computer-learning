x = int(input())
y = int(input())
class jisuan():
    def add(self, x, y):

        z = x + y
        return z

    def reduce(self, x, y):

        z = x - y
        return z

    def chen(self, x, y):

        z = x * y
        return z

    def chu(self, x, y):
        if y == 0:
            print("Error")
            return None
        z = x / y
        return z

jisuan_1 = jisuan()
results = []
results.append([jisuan_1.add(x, y)])
results.append(jisuan_1.reduce(x, y))
results.append(jisuan_1.chen(x, y))
results.append(jisuan_1.chu(x, y))
print(results)

print("加法", jisuan_1.add(x, y))
print("减法", jisuan_1.reduce(x, y))
print("乘法", jisuan_1.chen(x, y))
print("除法", jisuan_1.chu(x, y))
