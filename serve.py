from flask import Flask, request, render_template
from src.ar_main import main

global model
model = None

# Khởi tạo flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any string works here'
app.static_folder = 'static'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    selected = request.form.get('comp_select')
    checkbox = request.form.get('check_box')
    print(checkbox, selected)
    if selected:
        main(selected, checkbox)
    return render_template('index.html')

if __name__ == "__main__":
    print("App run!")
    # Load model
    app.run(debug=False, threaded=False)
