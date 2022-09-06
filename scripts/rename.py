import os
path = "G:/HgWork/医疗废弃物/imgs/"
# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
print(len(f))
n = 0
i = 0
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = f[n]

    # 设置新文件名
    newname = str(n+1) + '.jpeg'
    # 用os模块中的rename方法对文件改名
    os.rename(path+oldname, path+newname)
    print(oldname, '======>', newname)

    n += 1
