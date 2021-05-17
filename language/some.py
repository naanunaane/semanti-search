import redis
import pandas as pd 
py_df  = pd.read_csv('python.csv')
r = redis.StrictRedis(host='localhost', port=12000, db=0,password="codesearch")



k1 = py_df.iloc[0]['docstring']
v1 = py_df.iloc[0]['function']


langs = ['python','go','javascript','java']

for lang in langs:
    lang_df = pd.read_csv(lang+'.csv')
    for idx in range(len(lang_df)):
        key = lang_df.iloc[idx]['docstring']
        value = lang_df.iloc[idx]['function']
        codestring = key+'<CODESTRING>'+value
        r.set(lang+'_'+str(idx),codestring)

print(r.get("python_1").decode("utf-8"))
print(r.get("java_45").decode("utf-8"))



