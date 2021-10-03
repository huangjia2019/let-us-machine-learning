import numpy as np #导入NumPy
from flask import Flask, request, render_template #导入Flask相关包
import pickle #导入模块序列化包

app = Flask(__name__) 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home(): # 默认启动页面
    return render_template('index.html') # 启动index.html

@app.route('/predict',methods=['POST'])
def predict(): # 启动预测页面

    features = [int(x) for x in request.form.values()] # 输入特征
    label = [np.array(features)] # 标签
    prediction = model.predict(label) # 预测结果

    output = round(prediction[0], 2) #输出预测结果

    return render_template('index.html', #预测浏览量
                           prediction_text='浏览量 {}'.format(int(output)))

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__": # 启动程序
    app.run(debug=True)