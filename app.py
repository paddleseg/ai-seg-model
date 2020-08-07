from flask import Flask
from flask import request, abort, jsonify, send_file
from werkzeug.utils import secure_filename
import seg
import os
import util

app = Flask(__name__)

args = {}
args['image_shape'] = [192, 192]
args['model_dir'] = "/app/myModel"
args['save_dir'] = util.DEST_FOLDER
args['background_image_path'] = util.TRANSPARENT


@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>This Is A AI Prediction Service</h1>'


@app.route('/upload', methods=['POST'])
def upload():
    # print(request.environ)
    file = request.files['image']
    filename = os.path.join(util.SRC_FOLDER + secure_filename(file.filename))
    file.save(filename)
    # result = seg.init([filename])

    args['image_path'] = filename
    # args['background_image_path'] = "/data/input/black.jpg"

    result = seg.infer(args)
    if result['succ']:
        # return result['path']
        return send_file(result['path'], secure_filename(result['path'])), 200

    return jsonify(error=str('seg error')), 499


@app.route('/result', methods=['GET'])
def getResult():
    file = request.args.get('file')
    filePath = os.path.join(util.DEST_FOLDER + secure_filename(file))
    print(secure_filename(filePath))
    # # buffer = os.open(filePath)
    try:
        return send_file(filePath, attachment_filename=file)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    # seg.infer(args)
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)
