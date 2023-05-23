
with open('../data/train.json', 'r', encoding='utf-8') as f:
    con=f.read().splitlines()

print(len(con))

length=3705
sum=0
max_len=0
min_len=9999

for i in con:
    i=eval(i)
    t=len(i['整编内容'])
    if max_len<t:
        max_len=t
    if t<10:
        print(i['整编内容'])
    else:
        if min_len>t:
            min_len=t
    sum+=t

print('共{}条简讯数据,平均简讯长度为{} ,最长简讯长度{},最短简讯长度{}'.format(length,sum//length,max_len,min_len))
