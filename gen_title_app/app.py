from flask import Flask, render_template, request,jsonify
from gen_titile import PegasusPredictor


app = Flask(__name__)

def load_model():
    # 模型加载代码
    predictor = PegasusPredictor("static/models/pegasus-238M")

    return predictor

def create_app():
    app.config['model'] = load_model()
    return app


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/restart_model', methods=['POST'])
def restart_model():
    try:
        model = load_model()
        app.config['model'] = model
        return '模型重新加载成功'
    except Exception as e:
        return str(e)



@app.route('/summary', methods=['POST'])
def summarize():
    text = request.form['text']
    model = app.config['model']
    summary = model.predict(text,max_length=80)[0]
    return jsonify({'summary': summary})
    #return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app=create_app()
    app.run()
