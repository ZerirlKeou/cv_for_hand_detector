import pandas as pd

i = 0
file = pd.read_csv("cmd.csv")
df = pd.DataFrame(file)
print(df)
docu = df[i:i + 1]
print(docu)
list_cmd = [docu['input'][i], docu['output'][i]]
print(list_cmd)
# change_img = []
# for x in range(0, 2, 1):
#     # if str(docu['Img_id' + str(x)][i]) != str(0):
#         # print(docu['Img_id' + str(x)][i])
#         # print(f'{path}\\{str(docu["""Img_id0"""][i]) + ".png"}')
