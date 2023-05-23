from flask import Flask, render_template, request, jsonify
from gen_summary import BartPredictor,T5Predictor
from translations.xiaoniu import translate_doc


app = Flask(__name__)

def load_model():
    # 模型加载代码
    predictor = BartPredictor("static/models/bart-base-summary")
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
    print("获取的原文为", text)
    summary = model.predict(text, max_length=100)[0]
    print("摘要结果为", summary)
    return jsonify({'summary': summary})

@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    print("获取的原文为",text)
    translated_text = translate_doc(text)
    print("翻译结果为",translated_text)
    return jsonify({'translation': translated_text})

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8085)
