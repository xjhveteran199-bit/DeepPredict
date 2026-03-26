src = open(r'C:\Users\XJH\DeepPredict\deeppredict_web.py', encoding='utf-8').read()
start = src.find('def on_file_upload(')
end = src.find('\ndef plot_bland_altman(')
func = src[start:end]
idx = func.find('success =')
print(func[idx:idx+1000])
